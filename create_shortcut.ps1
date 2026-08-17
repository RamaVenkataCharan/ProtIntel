# create_shortcut.ps1
# Creates a Desktop shortcut pointing to start_protintel.bat
# with the correct WorkingDirectory (project root) and a custom icon.

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Paths
$ScriptDir    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot  = $ScriptDir
$StartBat     = Join-Path $ProjectRoot 'start_protintel.bat'
$DesktopPath  = [Environment]::GetFolderPath('Desktop')
$ShortcutName = 'ProtIntel - Launch Demo.lnk'
$ShortcutPath = Join-Path $DesktopPath $ShortcutName
$IconIco      = Join-Path $ProjectRoot 'frontend\public\protintel.ico'

# Generate a custom ICO using System.Drawing (lightning bolt in brand purple)
function New-ProtIntelIcon {
    param([string]$IcoPath)

    Add-Type -AssemblyName System.Drawing

    $size   = 256
    $bmp    = New-Object System.Drawing.Bitmap($size, $size)
    $g      = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias

    # Dark background circle
    $bg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 12, 8, 28))
    $g.FillEllipse($bg, 0, 0, ($size - 1), ($size - 1))

    # Purple lightning bolt
    $fg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 123, 63, 255))
    $bolt = @(
        [System.Drawing.PointF]::new(148, 28),
        [System.Drawing.PointF]::new(88,  118),
        [System.Drawing.PointF]::new(128, 118),
        [System.Drawing.PointF]::new(104, 228),
        [System.Drawing.PointF]::new(168, 132),
        [System.Drawing.PointF]::new(128, 132),
        [System.Drawing.PointF]::new(168, 28)
    )
    $g.FillPolygon($fg, $bolt)

    # Blue highlight on top of bolt
    $hl = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(160, 71, 191, 255))
    $glint = @(
        [System.Drawing.PointF]::new(148, 28),
        [System.Drawing.PointF]::new(168, 28),
        [System.Drawing.PointF]::new(143, 90),
        [System.Drawing.PointF]::new(128, 90)
    )
    $g.FillPolygon($hl, $glint)
    $g.Dispose()

    # Write PNG bytes into a valid ICO file manually
    $ms = New-Object System.IO.MemoryStream
    $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
    $png = $ms.ToArray()
    $ms.Dispose()
    $bmp.Dispose()

    $ico    = New-Object System.IO.MemoryStream
    $writer = New-Object System.IO.BinaryWriter($ico)
    # ICONDIR
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]1)
    # ICONDIRENTRY
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([byte]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]32)
    $writer.Write([uint32]$png.Length)
    $writer.Write([uint32]22)
    # PNG data
    $writer.Write($png)
    $writer.Flush()

    [System.IO.File]::WriteAllBytes($IcoPath, $ico.ToArray())
    $ico.Dispose()
    Write-Host "  Icon written: $IcoPath"
}

# Choose icon path
$IconLocation = ''
try {
    New-ProtIntelIcon -IcoPath $IconIco
    $IconLocation = "${IconIco},0"
} catch {
    Write-Warning "Custom icon failed: $_"
    # Fallback: shell32.dll index 23 (lightning/gear), 45 (arrow), 277 (rocket)
    $IconLocation = 'C:\Windows\System32\shell32.dll,277'
    Write-Host "  Using fallback icon: $IconLocation"
}

# Create the .lnk shortcut via WScript.Shell
$ws        = New-Object -ComObject WScript.Shell
$lnk       = $ws.CreateShortcut($ShortcutPath)
$lnk.TargetPath       = $StartBat
$lnk.WorkingDirectory = $ProjectRoot
$lnk.Description      = 'Launch ProtIntel demo - backend and frontend'
$lnk.WindowStyle      = 1
$lnk.IconLocation     = $IconLocation
$lnk.Save()

Write-Host ''
Write-Host '  ========================================='
Write-Host '   ProtIntel Desktop Shortcut Created!'
Write-Host '  ========================================='
Write-Host "  Shortcut    : $ShortcutPath"
Write-Host "  Target      : $StartBat"
Write-Host "  Working Dir : $ProjectRoot"
Write-Host "  Icon        : $IconLocation"
Write-Host ''
