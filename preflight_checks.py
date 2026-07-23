"""ProtIntel Pre-Flight Diagnostics.

Runs 6 sanity checks before launching a real training run:
  1. Data loading & shape verification
  2. Label distribution analysis
  3. Embedding cache sanity
  4. Single-batch overfit test
  5. Dataset file integrity
  6. Checkpoint inspection

Usage:
    python preflight_checks.py
"""

from __future__ import annotations

import gzip
import io
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────

_WIDTH = 70

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[96m[INFO]\033[0m"


def header(title: str) -> None:
    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")


def subheader(title: str) -> None:
    print(f"\n--- {title} ---")


def result_line(label: str, status: str, detail: str = "") -> None:
    pad = " " * max(0, 40 - len(label))
    detail_str = f"  ({detail})" if detail else ""
    print(f"  {label}{pad}{status}{detail_str}")


# ──────────────────────────────────────────────────────────────────────
# Check 1: Data Loading & Shape Verification
# ──────────────────────────────────────────────────────────────────────

def check_data_loading() -> str:
    """Pull one batch and verify all tensor shapes."""
    header("CHECK 1: Data Loading & Shape Verification")
    try:
        from src.data.data_module import DataModule
        from src.utils.config_loader import load_config

        config = load_config()

        # Use small batch for diagnostics, and 0 workers to avoid
        # multiprocessing issues during debugging
        dm = DataModule.from_config(config=config.data, batch_size=4)
        dm.num_workers = 0
        dm.prefetch_factor = 2
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        print(f"  Train dataset size: {len(dm.train_dataset)} samples")
        print(f"  Train loader batches: {len(train_loader)}")

        val_loader = dm.val_dataloader()
        print(f"  Val dataset size: {len(dm.val_dataset)} samples")
        print(f"  Val loader batches: {len(val_loader)}")

        # Pull one batch
        batch = next(iter(train_loader))

        subheader("Batch tensor shapes")
        for key in ["input_ids", "attention_mask", "q3_labels", "q8_labels",
                     "lengths", "seq_length"]:
            if key in batch:
                val = batch[key]
                if isinstance(val, torch.Tensor):
                    print(f"    {key}: {val.shape}  dtype={val.dtype}")
                else:
                    print(f"    {key}: type={type(val).__name__}")

        if "sequences" in batch:
            print(f"    sequences: list of {len(batch['sequences'])} strings")
            print(f"    first seq (truncated): {batch['sequences'][0][:60]}...")

        if "cached_embeddings" in batch or "embeddings" in batch:
            emb_key = "cached_embeddings" if "cached_embeddings" in batch else "embeddings"
            emb = batch[emb_key]
            print(f"    {emb_key}: {emb.shape}  dtype={emb.dtype}")
            result_line("Embeddings present in batch", PASS)
        else:
            result_line("Embeddings present in batch", WARN,
                        "NOT cached — will need live ESM-2 inference")

        # Validate label ranges
        q3 = batch["q3_labels"]
        q8 = batch["q8_labels"]
        q3_valid = q3[q3 != -100]
        q8_valid = q8[q8 != -100]

        q3_range_ok = q3_valid.min() >= 0 and q3_valid.max() <= 2
        q8_range_ok = q8_valid.min() >= 0 and q8_valid.max() <= 7

        result_line("Q3 label range [0-2]",
                    PASS if q3_range_ok else FAIL,
                    f"actual: {q3_valid.min().item()}-{q3_valid.max().item()}")
        result_line("Q8 label range [0-7]",
                    PASS if q8_range_ok else FAIL,
                    f"actual: {q8_valid.min().item()}-{q8_valid.max().item()}")

        # Check input_ids vs labels length alignment
        # ESM-2 adds BOS/EOS so input_ids length = seq_length + 2
        input_len = batch["input_ids"].shape[1]
        label_len = batch["q3_labels"].shape[1]
        lengths = batch["lengths"] if "lengths" in batch else batch["seq_length"]
        max_seq_len = lengths.max().item()

        print(f"\n  input_ids padded length: {input_len}")
        print(f"  labels padded length:    {label_len}")
        print(f"  max actual seq length:   {max_seq_len}")
        print(f"  input_ids - labels diff: {input_len - label_len} (expect ~2 for BOS/EOS)")

        result_line("Data loading", PASS)
        return "PASS"

    except FileNotFoundError as e:
        print(f"\n  {FAIL} Dataset files not found!")
        print(f"  Error: {e}")
        print(f"  Action: Run 'python scripts/download_data.py' first")
        return "FAIL"
    except Exception as e:
        print(f"\n  {FAIL} Data loading failed!")
        traceback.print_exc()
        return "FAIL"


# ──────────────────────────────────────────────────────────────────────
# Check 2: Label Distribution Analysis
# ──────────────────────────────────────────────────────────────────────

def check_label_distribution() -> str:
    """Check class distributions for degeneracy."""
    header("CHECK 2: Label Distribution Analysis")
    try:
        from src.data.data_module import DataModule
        from src.utils.config_loader import load_config

        config = load_config()
        dm = DataModule.from_config(config=config.data, batch_size=4)
        dm.num_workers = 0
        dm.setup(stage="fit")

        overall_status = "PASS"

        for split_name, dataset in [("Train", dm.train_dataset),
                                     ("Val", dm.val_dataset)]:
            if dataset is None:
                result_line(f"{split_name} dataset", WARN, "not loaded")
                continue

            subheader(f"{split_name} Set ({len(dataset)} samples)")

            # Collect Q3 and Q8 distributions
            q3_counter: Counter = Counter()
            q8_counter: Counter = Counter()
            seq_lengths: list[int] = []

            for i in range(len(dataset)):
                q3_str = dataset.q3_labels[i]
                q8_str = dataset.q8_labels[i]
                seq_lengths.append(len(dataset.sequences[i]))
                for ch in q3_str:
                    q3_counter[ch] += 1
                for ch in q8_str:
                    q8_counter[ch] += 1

            # Q3 distribution
            q3_total = sum(q3_counter.values())
            print(f"\n  Q3 label distribution ({q3_total} total residues):")
            q3_classes = ["H", "E", "C"]
            max_pct = 0
            for cls in q3_classes:
                count = q3_counter.get(cls, 0)
                pct = 100 * count / q3_total if q3_total > 0 else 0
                max_pct = max(max_pct, pct)
                bar = "█" * int(pct / 2)
                print(f"    {cls}: {count:>8} ({pct:5.1f}%)  {bar}")

            if max_pct > 80:
                result_line(f"{split_name} Q3 balance", WARN,
                            f"dominant class at {max_pct:.1f}%")
                overall_status = "WARN"
            elif any(q3_counter.get(c, 0) == 0 for c in q3_classes):
                result_line(f"{split_name} Q3 balance", FAIL, "missing class!")
                overall_status = "FAIL"
            else:
                result_line(f"{split_name} Q3 balance", PASS)

            # Q8 distribution
            q8_total = sum(q8_counter.values())
            print(f"\n  Q8 label distribution ({q8_total} total residues):")
            q8_classes = ["H", "E", "G", "I", "B", "T", "S", "C"]
            missing_q8 = []
            for cls in q8_classes:
                count = q8_counter.get(cls, 0)
                pct = 100 * count / q8_total if q8_total > 0 else 0
                bar = "█" * int(pct / 2)
                print(f"    {cls}: {count:>8} ({pct:5.1f}%)  {bar}")
                if count == 0:
                    missing_q8.append(cls)

            if missing_q8:
                result_line(f"{split_name} Q8 balance", WARN,
                            f"missing classes: {missing_q8}")
                overall_status = "WARN"
            else:
                result_line(f"{split_name} Q8 balance", PASS)

            # Sequence length stats
            if seq_lengths:
                print(f"\n  Sequence lengths: min={min(seq_lengths)}, "
                      f"max={max(seq_lengths)}, "
                      f"mean={sum(seq_lengths)/len(seq_lengths):.0f}")

        return overall_status

    except Exception as e:
        print(f"\n  {FAIL} Label distribution check failed!")
        traceback.print_exc()
        return "FAIL"


# ──────────────────────────────────────────────────────────────────────
# Check 3: Embedding Cache Sanity
# ──────────────────────────────────────────────────────────────────────

def check_embedding_sanity() -> str:
    """Check embedding cache coverage and quality."""
    header("CHECK 3: Embedding Cache Sanity")
    try:
        from src.utils.io_utils import compute_sequence_hash

        embeddings_dir = PROJECT_ROOT / "datasets" / "processed" / "embeddings"

        if not embeddings_dir.exists():
            result_line("Embeddings directory", FAIL, "does not exist")
            return "FAIL"

        cached_files = list(embeddings_dir.glob("*.pt"))
        print(f"  Cached embedding files: {len(cached_files)}")

        # Count total sequences needed
        from src.data.data_module import DataModule
        from src.utils.config_loader import load_config

        config = load_config()
        dm = DataModule.from_config(config=config.data, batch_size=4)
        dm.num_workers = 0
        dm.setup(stage="fit")

        # Check coverage
        total_seqs = len(dm.train_dataset) + (len(dm.val_dataset) if dm.val_dataset else 0)
        cached_hashes = {f.stem for f in cached_files}

        train_cached = 0
        train_missing = 0
        for seq in dm.train_dataset.sequences:
            h = compute_sequence_hash(seq)
            if h in cached_hashes:
                train_cached += 1
            else:
                train_missing += 1

        val_cached = 0
        val_missing = 0
        if dm.val_dataset:
            for seq in dm.val_dataset.sequences:
                h = compute_sequence_hash(seq)
                if h in cached_hashes:
                    val_cached += 1
                else:
                    val_missing += 1

        print(f"\n  Train set: {train_cached}/{train_cached + train_missing} "
              f"sequences have cached embeddings "
              f"({100*train_cached/(train_cached+train_missing):.1f}%)")
        print(f"  Val set:   {val_cached}/{val_cached + val_missing} "
              f"sequences have cached embeddings "
              f"({100*val_cached/(val_cached+val_missing):.1f}%)" if (val_cached + val_missing) > 0 else "")

        coverage = (train_cached + val_cached) / total_seqs if total_seqs > 0 else 0

        if coverage < 0.1:
            result_line("Embedding cache coverage", FAIL,
                        f"{coverage*100:.1f}% — most seqs will need live ESM-2")
        elif coverage < 0.9:
            result_line("Embedding cache coverage", WARN,
                        f"{coverage*100:.1f}%")
        else:
            result_line("Embedding cache coverage", PASS,
                        f"{coverage*100:.1f}%")

        # Inspect a random cached embedding
        if cached_files:
            subheader("Sample embedding inspection")
            sample_file = cached_files[0]
            try:
                emb = torch.load(str(sample_file), map_location="cpu",
                                 weights_only=True)
                print(f"    File: {sample_file.name}")
                print(f"    Shape: {emb.shape}")
                print(f"    Dtype: {emb.dtype}")
                print(f"    Min:   {emb.min().item():.4f}")
                print(f"    Max:   {emb.max().item():.4f}")
                print(f"    Mean:  {emb.mean().item():.4f}")
                print(f"    Std:   {emb.std().item():.4f}")

                # Check for all zeros
                if emb.abs().max().item() < 1e-6:
                    result_line("Embedding values", FAIL, "ALL ZEROS!")
                    return "FAIL"

                # Check dimensions
                expected_dim = config.model.esm2.embedding_dim
                if emb.dim() == 2 and emb.shape[1] == expected_dim:
                    result_line("Embedding dimensions", PASS,
                                f"(L={emb.shape[0]}, D={expected_dim})")
                else:
                    result_line("Embedding dimensions", FAIL,
                                f"expected (L, {expected_dim}), got {emb.shape}")
                    return "FAIL"

                # Check for NaN/Inf
                if torch.isnan(emb).any() or torch.isinf(emb).any():
                    result_line("Embedding NaN/Inf", FAIL, "contains NaN or Inf!")
                    return "FAIL"
                else:
                    result_line("Embedding NaN/Inf", PASS, "clean")

            except Exception as e:
                result_line("Embedding file loading", FAIL, str(e))
                return "FAIL"

        return "PASS" if coverage >= 0.9 else ("WARN" if coverage >= 0.1 else "FAIL")

    except Exception as e:
        print(f"\n  {FAIL} Embedding check failed!")
        traceback.print_exc()
        return "FAIL"


# ──────────────────────────────────────────────────────────────────────
# Check 4: Single-Batch Overfit Test
# ──────────────────────────────────────────────────────────────────────

def check_single_batch_overfit() -> str:
    """Attempt to overfit a single batch for 100 steps."""
    header("CHECK 4: Single-Batch Overfit Test")
    try:
        from src.data.data_module import DataModule
        from src.utils.config_loader import load_config

        config = load_config()
        dm = DataModule.from_config(config=config.data, batch_size=4)
        dm.num_workers = 0
        dm.setup(stage="fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Check if embeddings are available, if not we'll use a simpler model
        has_embeddings = "embeddings" in batch or "cached_embeddings" in batch

        if not has_embeddings:
            print(f"  {WARN} No cached embeddings in batch — will attempt test")
            print(f"         with live ESM-2. This may be slow on CPU.")
            print(f"         Consider running 'python scripts/generate_embeddings.py' first.\n")

        # Build model
        from src.models.protintel_model import ProtIntelModel

        model = ProtIntelModel(config=config.model, device="cpu")
        model.train()

        # Disable all dropout for overfitting test
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.p = 0.0

        # Loss functions (no label smoothing, no class weights for clean test)
        from src.training.losses import create_loss_function

        q3_loss_fn = create_loss_function(
            loss_type="cross_entropy", label_smoothing=0.0
        )
        q8_loss_fn = create_loss_function(
            loss_type="cross_entropy", label_smoothing=0.0
        )

        # Optimizer with higher LR for overfitting
        params = model.get_downstream_parameters()
        optimizer = torch.optim.Adam(params, lr=1e-3)

        # Prepare batch tensors
        embeddings = batch.get("embeddings", batch.get("cached_embeddings"))
        sequences = batch.get("sequence", batch.get("sequences"))
        attention_mask = batch["attention_mask"]
        seq_lengths = batch.get("seq_length", batch.get("lengths"))
        q3_labels = batch["q3_labels"]
        q8_labels = batch["q8_labels"]

        print(f"  Running 100 overfit steps on batch of {q3_labels.shape[0]} samples...")
        print(f"  {'Step':>6}  {'Total Loss':>12}  {'Q3 Loss':>10}  {'Q8 Loss':>10}")
        print(f"  {'─'*44}")

        losses_track = []
        steps_to_print = {0, 9, 24, 49, 74, 99}
        start_time = time.time()

        for step in range(100):
            optimizer.zero_grad()
            outputs = model(
                sequences=sequences if embeddings is None else None,
                embeddings=embeddings,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )

            # Reshape logits for loss: (B, L, C) and labels: (B, L)
            q3_logits = outputs["q3_logits"]
            q8_logits = outputs["q8_logits"]

            q3_loss = q3_loss_fn(q3_logits, q3_labels)
            q8_loss = q8_loss_fn(q8_logits, q8_labels)
            total_loss = q3_loss + 0.5 * q8_loss

            total_loss.backward()
            optimizer.step()

            loss_val = total_loss.item()
            losses_track.append(loss_val)

            if step in steps_to_print:
                print(f"  {step+1:>6}  {loss_val:>12.4f}  "
                      f"{q3_loss.item():>10.4f}  {q8_loss.item():>10.4f}")

            # Early exit if NaN
            if torch.isnan(total_loss):
                print(f"\n  {FAIL} Loss became NaN at step {step+1}!")
                return "FAIL"

        elapsed = time.time() - start_time
        print(f"\n  Completed in {elapsed:.1f}s")

        # Analyze trajectory
        initial_loss = losses_track[0]
        final_loss = losses_track[-1]
        min_loss = min(losses_track)
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100

        print(f"\n  Initial loss:  {initial_loss:.4f}")
        print(f"  Final loss:    {final_loss:.4f}")
        print(f"  Min loss:      {min_loss:.4f}")
        print(f"  Reduction:     {loss_reduction:.1f}%")

        # Random-guess loss baselines
        # CE loss for uniform random guessing: -log(1/C) = log(C)
        import math
        q3_random = math.log(3)  # ~1.099
        q8_random = math.log(8)  # ~2.079
        combined_random = q3_random + 0.5 * q8_random  # ~2.139

        print(f"\n  Random-guess baseline loss: ~{combined_random:.3f} "
              f"(Q3: {q3_random:.3f}, Q8: {q8_random:.3f})")

        if final_loss < combined_random * 0.5:
            result_line("Single-batch overfit", PASS,
                        f"loss dropped to {final_loss:.4f}, well below random")
        elif final_loss < combined_random * 0.9:
            result_line("Single-batch overfit", PASS,
                        f"loss dropping ({loss_reduction:.0f}% reduction)")
        elif loss_reduction > 20:
            result_line("Single-batch overfit", WARN,
                        f"loss reduced {loss_reduction:.0f}% but still above random")
        else:
            result_line("Single-batch overfit", FAIL,
                        f"only {loss_reduction:.0f}% reduction — model may not be learning")
            return "FAIL"

        return "PASS"

    except Exception as e:
        print(f"\n  {FAIL} Single-batch overfit test failed!")
        traceback.print_exc()
        return "FAIL"


# ──────────────────────────────────────────────────────────────────────
# Check 5: Dataset File Integrity
# ──────────────────────────────────────────────────────────────────────

def check_dataset_integrity() -> str:
    """Verify raw dataset files are valid and reasonably sized."""
    header("CHECK 5: Dataset File Integrity")

    raw_dir = PROJECT_ROOT / "datasets" / "raw"

    # Expected files and approximate sizes (from the Princeton dataset)
    expected_files = {
        "cullpdb+profile_6133_filtered.npy.gz": {
            "min_size_mb": 100,  # The real file is ~600MB
            "expected_shape_0_min": 5000,
            "description": "CullPDB 6133 training set",
        },
        "cb513+profile_split1.npy.gz": {
            "min_size_mb": 5,    # The real file is ~35-40MB
            "expected_shape_0_min": 400,
            "description": "CB513 test set",
        },
    }

    overall_status = "PASS"

    for filename, expected in expected_files.items():
        filepath = raw_dir / filename
        subheader(f"{filename} — {expected['description']}")

        if not filepath.exists():
            result_line(f"  File exists", FAIL, "not found")
            overall_status = "FAIL"
            continue

        # Check file size
        size_bytes = filepath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_kb = size_bytes / 1024
        min_mb = expected["min_size_mb"]

        if size_mb < min_mb:
            print(f"    File size: {size_kb:.1f} KB")
            print(f"    Expected minimum: ~{min_mb} MB")
            result_line("File size", FAIL,
                        f"{size_kb:.0f} KB is WAY too small — likely corrupt/placeholder")
            overall_status = "FAIL"
        else:
            print(f"    File size: {size_mb:.1f} MB")
            result_line("File size", PASS)

        # Try to load and verify shape
        try:
            try:
                with gzip.open(str(filepath), "rb") as f:
                    raw_bytes = f.read()
                data = np.load(io.BytesIO(raw_bytes))
            except gzip.BadGzipFile:
                data = np.load(str(filepath))

            print(f"    Raw array shape: {data.shape}")
            print(f"    Dtype: {data.dtype}")

            # Try to reshape to (N, 700, 57)
            if data.ndim == 2:
                if data.shape[1] == 700 * 57:
                    n_samples = data.shape[0]
                    data = data.reshape(n_samples, 700, 57)
                    print(f"    Reshaped to: {data.shape}")
                elif data.shape[0] % 700 == 0:
                    n_samples = data.shape[0] // 700
                    try:
                        data = data.reshape(n_samples, 700, 57)
                        print(f"    Reshaped to: {data.shape}")
                    except ValueError:
                        print(f"    {WARN} Could not reshape to (N, 700, 57)")
                else:
                    print(f"    {WARN} Unexpected shape, cannot reshape")

            if data.ndim == 3:
                n_samples = data.shape[0]
                min_samples = expected["expected_shape_0_min"]

                if n_samples >= min_samples:
                    result_line("Sample count", PASS,
                                f"{n_samples} samples (expected >= {min_samples})")
                else:
                    result_line("Sample count", FAIL,
                                f"{n_samples} samples (expected >= {min_samples})")
                    overall_status = "FAIL"

                # Verify feature dimensions
                if data.shape[1] == 700 and data.shape[2] == 57:
                    result_line("Feature dimensions (700 x 57)", PASS)
                else:
                    result_line("Feature dimensions", FAIL,
                                f"expected (N, 700, 57), got {data.shape}")
                    overall_status = "FAIL"

                # Spot-check: are the amino acid one-hot columns reasonable?
                sample = data[0]
                aa_onehot = sample[:, :21]
                # Each row should have exactly one 1 in positions 0-20
                # (for valid residues) or all zeros (for padding)
                noseq = sample[:, 43]
                valid_residues = (noseq == 0).sum()
                print(f"    First protein: {valid_residues} valid residues out of 700")

                if valid_residues < 10:
                    result_line("First protein valid residues", WARN,
                                "very few valid residues")
                else:
                    result_line("First protein valid residues", PASS)

            else:
                result_line("Array dimensions", WARN,
                            f"unexpected ndim={data.ndim}")

        except Exception as e:
            result_line("File loading", FAIL, str(e))
            overall_status = "FAIL"

    return overall_status


# ──────────────────────────────────────────────────────────────────────
# Check 6: Checkpoint Inspection
# ──────────────────────────────────────────────────────────────────────

def check_checkpoint() -> str:
    """Inspect existing checkpoints for validity."""
    header("CHECK 6: Checkpoint Inspection")

    models_dir = PROJECT_ROOT / "models"
    checkpoints = sorted(models_dir.glob("*.pt"))

    if not checkpoints:
        result_line("Checkpoint files", INFO, "none found — no prior training")
        return "INFO"

    print(f"  Found {len(checkpoints)} checkpoint files:")
    for cp in checkpoints:
        size_mb = cp.stat().st_size / (1024 * 1024)
        print(f"    {cp.name}  ({size_mb:.1f} MB)")

    # Inspect best checkpoint
    best_path = models_dir / "best_checkpoint.pt"
    if best_path.exists():
        subheader("Best checkpoint analysis")
        try:
            checkpoint = torch.load(str(best_path), map_location="cpu",
                                    weights_only=False)

            if "epoch" in checkpoint:
                print(f"    Epoch: {checkpoint['epoch']}")
            if "metrics" in checkpoint:
                print(f"    Stored metrics:")
                for k, v in checkpoint["metrics"].items():
                    if isinstance(v, float):
                        print(f"      {k}: {v:.4f}")

            if "model_state_dict" in checkpoint:
                sd = checkpoint["model_state_dict"]
                n_keys = len(sd)
                total_params = sum(v.numel() for v in sd.values())
                print(f"    State dict keys: {n_keys}")
                print(f"    Total parameters: {total_params / 1e6:.1f}M")

                # Check for any NaN/Inf in weights
                nan_keys = []
                for k, v in sd.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        nan_keys.append(k)

                if nan_keys:
                    result_line("Weight NaN/Inf", FAIL,
                                f"{len(nan_keys)} layers have NaN/Inf")
                    for k in nan_keys[:5]:
                        print(f"      {k}")
                else:
                    result_line("Weight NaN/Inf", PASS, "all clean")

                # Verify against current model architecture
                try:
                    from src.models.protintel_model import ProtIntelModel
                    from src.utils.config_loader import load_config

                    config = load_config()
                    model = ProtIntelModel(config=config.model, device="cpu")
                    model_keys = set(model.state_dict().keys())
                    ckpt_keys = set(sd.keys())

                    missing = model_keys - ckpt_keys
                    extra = ckpt_keys - model_keys

                    if missing:
                        result_line("Architecture match", WARN,
                                    f"{len(missing)} missing keys")
                        for k in list(missing)[:3]:
                            print(f"      missing: {k}")
                    elif extra:
                        result_line("Architecture match", WARN,
                                    f"{len(extra)} extra keys")
                    else:
                        result_line("Architecture match", PASS)
                except Exception:
                    result_line("Architecture match", INFO, "could not verify")

            if "optimizer_state_dict" in checkpoint:
                result_line("Optimizer state", PASS, "saved")
            else:
                result_line("Optimizer state", WARN, "not saved — cannot resume")

            # Analyze stored metrics for training quality
            if "metrics" in checkpoint:
                metrics = checkpoint["metrics"]
                q3_acc = metrics.get("val_q3_accuracy", 0)
                q8_acc = metrics.get("val_q8_accuracy", 0)
                val_loss = metrics.get("val_loss", 0)

                print(f"\n    Training quality assessment:")
                if q3_acc < 0.40:
                    print(f"      Q3 accuracy {q3_acc:.4f} is near random (0.33)")
                    result_line("Training quality", FAIL,
                                "model barely learned — check data pipeline")
                elif q3_acc < 0.55:
                    print(f"      Q3 accuracy {q3_acc:.4f} is below expected (>0.60)")
                    result_line("Training quality", WARN,
                                "below expected performance")
                else:
                    result_line("Training quality", PASS,
                                f"Q3={q3_acc:.4f}")

        except Exception as e:
            result_line("Checkpoint loading", FAIL, str(e))
            return "FAIL"
    else:
        result_line("Best checkpoint", WARN, "not found")

    # Check TensorBoard logs
    subheader("TensorBoard log analysis")
    logs_dir = PROJECT_ROOT / "logs"
    event_files = list(logs_dir.glob("events.out.tfevents.*"))

    if event_files:
        print(f"  Found {len(event_files)} TensorBoard event files:")
        for ef in event_files:
            size = ef.stat().st_size
            if size < 100:
                print(f"    {ef.name}: {size} bytes  ← EMPTY (crashed immediately)")
            elif size < 1000:
                print(f"    {ef.name}: {size} bytes  ← minimal (1-2 epochs?)")
            else:
                print(f"    {ef.name}: {size} bytes  ← has data")

        non_empty = sum(1 for ef in event_files if ef.stat().st_size > 100)
        if non_empty == 0:
            result_line("TensorBoard logs", FAIL, "all empty — training never progressed")
        elif non_empty < len(event_files) // 2:
            result_line("TensorBoard logs", WARN,
                        f"only {non_empty}/{len(event_files)} have data")
        else:
            result_line("TensorBoard logs", PASS)
    else:
        result_line("TensorBoard logs", INFO, "none found")

    return "PASS"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run all pre-flight checks."""
    print("\n" + "╔" + "═" * (_WIDTH - 2) + "╗")
    print("║" + "  ProtIntel Pre-Flight Diagnostics  ".center(_WIDTH - 2) + "║")
    print("╚" + "═" * (_WIDTH - 2) + "╝")

    # System info
    print(f"\n  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  Device:  {os.environ.get('DEVICE', 'not set in env')}")

    results: dict[str, str] = {}

    # Run checks in order — check 5 (file integrity) first since others depend on it
    results["Dataset Integrity"] = check_dataset_integrity()
    results["Data Loading"] = check_data_loading()
    results["Label Distribution"] = check_label_distribution()
    results["Embedding Sanity"] = check_embedding_sanity()
    results["Checkpoint Status"] = check_checkpoint()

    # Only run overfit test if data loading passed and dataset isn't corrupt
    if results["Data Loading"] == "PASS" and results["Dataset Integrity"] != "FAIL":
        results["Single-Batch Overfit"] = check_single_batch_overfit()
    else:
        results["Single-Batch Overfit"] = "SKIP"
        print(f"\n  Skipping single-batch overfit test — data issues detected")

    # ── Summary ──
    header("SUMMARY")

    status_order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "INFO": 3, "PASS": 4}
    for check_name, status in sorted(results.items(),
                                      key=lambda x: status_order.get(x[1], 5)):
        status_str = {
            "PASS": PASS, "FAIL": FAIL, "WARN": WARN,
            "INFO": INFO, "SKIP": "\033[90m[SKIP]\033[0m",
        }.get(status, status)
        pad = " " * max(0, 35 - len(check_name))
        print(f"  {check_name}{pad}{status_str}")

    # Recommendations
    fails = [k for k, v in results.items() if v == "FAIL"]
    warns = [k for k, v in results.items() if v == "WARN"]

    if fails:
        print(f"\n{'=' * _WIDTH}")
        print(f"  \033[91mBLOCKERS — Fix these before training:\033[0m")
        print(f"{'=' * _WIDTH}")

        if "Dataset Integrity" in fails:
            print("""
  1. DATASET FILES ARE CORRUPT/TOO SMALL
     The .npy.gz files in datasets/raw/ are far too small to contain
     the real protein data. The CullPDB file should be ~600 MB but
     is only ~86 KB.

     FIX: Re-download the datasets:
       python scripts/download_data.py

     If the download script fails, manually download from:
       https://www.princeton.edu/~jzthree/datasets/ICML2014/
""")

        if "Single-Batch Overfit" in fails:
            print("""
  2. MODEL CANNOT OVERFIT A SINGLE BATCH
     This indicates a structural issue — wrong loss function,
     frozen weights, or broken forward pass.

     DEBUG: Check that get_downstream_parameters() returns
     trainable params, and that loss.backward() produces gradients.
""")

    if warns:
        print(f"\n{'=' * _WIDTH}")
        print(f"  \033[93mWARNINGS — Address before a real run:\033[0m")
        print(f"{'=' * _WIDTH}")

        if "Embedding Sanity" in warns or "Embedding Sanity" in fails:
            print("""
  - EMBEDDING CACHE IS INCOMPLETE
    Most training sequences don't have pre-computed embeddings.
    Without cached embeddings, training runs live ESM-2 (650M params)
    inference per batch ON CPU — this is infeasible.

    FIX: Pre-compute ALL embeddings first:
      python scripts/generate_embeddings.py --device cpu

    This is slow (~hours on CPU) but only needs to be done once.
    After this, training only uses the lightweight CNN-BiLSTM-Attention
    model and will be fast even on CPU.
""")

    if not fails:
        print(f"\n  \033[92m✓ No blockers found! You can proceed to training.\033[0m")
        print(f"    python train.py --device cpu --epochs 50 --batch-size 8")


if __name__ == "__main__":
    main()
