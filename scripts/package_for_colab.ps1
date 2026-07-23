#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Packages ProtIntel source code + pre-computed ESM-2 embeddings into
    protintel_colab_package.zip ready for upload to Google Colab.

.DESCRIPTION
    Includes:
      - src/              (corrected codebase with label parsing fix)
      - tests/            (regression test suite)
      - configs/          (training, model, data YAML configs)
      - train.py, evaluate.py, infer.py
      - scripts/generate_embeddings.py
      - requirements.txt, pyproject.toml
      - datasets/raw/     (CullPDB + CB513 .npy.gz files)
      - datasets/processed/embeddings/  (pre-computed ESM-2 .pt files)
      - colab/ProtIntel_GPU_Training.ipynb

    Excludes:
      - .venv/, .git/, __pycache__/, *.pyc
      - datasets/processed/embeddings is INCLUDED (large but needed)

.NOTES
    Run from project root: .\scripts\package_for_colab.ps1
    Output: protintel_colab_package.zip (~10+ GB due to embeddings)
#>

param(
    [string]$OutputPath = "protintel_colab_package.zip",
    [switch]$SkipEmbeddings
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot | Split-Path -Parent
Set-Location $ProjectRoot

Write-Host "=" * 60
Write-Host "ProtIntel Colab Package Builder"
Write-Host "=" * 60
Write-Host "Project root : $ProjectRoot"
Write-Host "Output zip   : $OutputPath"
Write-Host ""

# Verify required files exist
$RequiredFiles = @(
    "train.py",
    "evaluate.py",
    "configs/training.yaml",
    "configs/model.yaml",
    "configs/data.yaml",
    "src/models/protintel_model.py",
    "src/data/protein_dataset.py",
    "datasets/raw/cullpdb+profile_6133_filtered.npy.gz",
    "datasets/raw/cb513+profile_split1.npy.gz",
    "colab/ProtIntel_GPU_Training.ipynb"
)

Write-Host "Verifying required files..."
$Missing = @()
foreach ($f in $RequiredFiles) {
    if (Test-Path $f) {
        Write-Host "  [OK] $f"
    } else {
        Write-Host "  [MISSING] $f" -ForegroundColor Red
        $Missing += $f
    }
}

if ($Missing.Count -gt 0) {
    Write-Error "ABORT: $($Missing.Count) required file(s) missing. Fix before packaging."
    exit 1
}

# Count embeddings
$EmbDir = "datasets/processed/embeddings"
$EmbFiles = @(Get-ChildItem -Path $EmbDir -Filter "*.pt" -ErrorAction SilentlyContinue)
Write-Host ""
Write-Host "Embeddings found: $($EmbFiles.Count) .pt files in $EmbDir"
if ($EmbFiles.Count -lt 100) {
    Write-Warning "Only $($EmbFiles.Count) embedding files found. Run scripts/generate_embeddings.py first!"
}

# Build the zip using .NET System.IO.Compression
Write-Host ""
Write-Host "Building zip archive: $OutputPath ..."
$StartTime = Get-Date

if (Test-Path $OutputPath) { Remove-Item $OutputPath -Force }

Add-Type -AssemblyName System.IO.Compression.FileSystem

$ZipPath = Join-Path $ProjectRoot $OutputPath
$Zip = [System.IO.Compression.ZipFile]::Open($ZipPath, 'Create')

function Add-FileToZip {
    param([System.IO.Compression.ZipArchive]$Archive, [string]$FilePath, [string]$EntryName)
    $entry = $Archive.CreateEntry("protintel/$EntryName", [System.IO.Compression.CompressionLevel]::Fastest)
    $entryStream = $entry.Open()
    $fileStream = [System.IO.File]::OpenRead($FilePath)
    $fileStream.CopyTo($entryStream)
    $fileStream.Close()
    $entryStream.Close()
}

function Add-DirToZip {
    param([System.IO.Compression.ZipArchive]$Archive, [string]$DirPath, [string[]]$Excludes = @())
    $BaseDir = (Resolve-Path $DirPath).Path
    Get-ChildItem -Path $DirPath -Recurse -File | ForEach-Object {
        $RelPath = $_.FullName.Substring($BaseDir.Length + 1).Replace('\', '/')
        # Apply exclusion patterns
        $skip = $false
        foreach ($ex in $Excludes) {
            if ($_.FullName -like $ex) { $skip = $true; break }
        }
        if (-not $skip) {
            $EntryName = "$DirPath/$RelPath"
            Add-FileToZip -Archive $Archive -FilePath $_.FullName -EntryName $EntryName
        }
    }
}

$Excludes = @('*__pycache__*', '*.pyc', '*.pyo', '*.egg-info*', '*.DS_Store', '*.log')

# Add source directories
Write-Host "  Adding src/ ..."
Add-DirToZip -Archive $Zip -DirPath "src" -Excludes $Excludes

Write-Host "  Adding configs/ ..."
Add-DirToZip -Archive $Zip -DirPath "configs"

Write-Host "  Adding tests/ ..."
Add-DirToZip -Archive $Zip -DirPath "tests" -Excludes $Excludes

Write-Host "  Adding scripts/ ..."
Add-DirToZip -Archive $Zip -DirPath "scripts" -Excludes $Excludes

Write-Host "  Adding colab/ ..."
Add-DirToZip -Archive $Zip -DirPath "colab"

# Add root-level files
foreach ($f in @("train.py", "evaluate.py", "infer.py", "requirements.txt", "pyproject.toml")) {
    if (Test-Path $f) {
        Write-Host "  Adding $f ..."
        Add-FileToZip -Archive $Zip -FilePath $f -EntryName $f
    }
}

# Add raw datasets
Write-Host "  Adding datasets/raw/ ..."
Add-DirToZip -Archive $Zip -DirPath "datasets/raw"

# Add embeddings (large ??? main payload)
if (-not $SkipEmbeddings) {
    Write-Host "  Adding datasets/processed/embeddings/ ($($EmbFiles.Count) files) ??? this may take a while ..."
    $i = 0
    foreach ($f in $EmbFiles) {
        $RelPath = $f.FullName.Substring($ProjectRoot.Length + 1).Replace('\', '/')
        Add-FileToZip -Archive $Zip -FilePath $f.FullName -EntryName $RelPath
        $i++
        if ($i % 500 -eq 0) { Write-Host "    ... $i / $($EmbFiles.Count) embeddings added" }
    }
}

$Zip.Dispose()

$Elapsed = (Get-Date) - $StartTime
$SizeMB = (Get-Item $OutputPath).Length / 1MB
Write-Host ""
Write-Host "=" * 60
Write-Host "Package complete!"
Write-Host "  Output    : $OutputPath"
Write-Host ("  Size      : {0:N0} MB" -f $SizeMB)
Write-Host ("  Time      : {0:N0}s" -f $Elapsed.TotalSeconds)
Write-Host "=" * 60
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Open Google Colab: https://colab.research.google.com"
Write-Host "  2. Upload $OutputPath via Files panel (left sidebar -> Upload)"
Write-Host "  3. Open colab/ProtIntel_GPU_Training.ipynb from the package OR"
Write-Host "     directly upload colab/ProtIntel_GPU_Training.ipynb to Colab"
Write-Host "  4. Set Runtime -> Change runtime type -> T4 GPU"
Write-Host "  5. Run cells 1-10 in order"
Write-Host ""

