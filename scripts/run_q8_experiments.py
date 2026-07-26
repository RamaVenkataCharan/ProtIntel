"""
Q8 Loss Experiments: Class-Weighted CE and Focal Loss
=====================================================
Pre-loads all ESM-2 embeddings from the SQLite cache into RAM,
then trains the classification heads only (no ESM-2 forward passes).

Experiments:
  1. Q8 class-weighted cross-entropy (inverse frequency weights)
  2. Q8 focal loss (gamma=2.0)

Outputs:
  - models/experiments/q8_weighted_checkpoint.pt
  - models/experiments/q8_focal_checkpoint.pt
  - models/experiments/cb513_q8_weighted.json
  - models/experiments/cb513_q8_focal.json
"""

import os
import sys
import time
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
sys.modules["tensorflow"] = None

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from src.data.data_module import DataModule
from src.data.protein_dataset import collate_fn
from src.models.protintel_model import ProtIntelModel
from src.models.embedding_cache import SQLiteEmbeddingCache
from src.training.losses import create_loss_function
from src.training.metrics import ProteinMetrics
from src.utils.config_loader import load_config
from src.utils.io_utils import compute_sequence_hash, save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Embedding pre-loader ───────────────────────────────────────────
def preload_embeddings(dataset, db_path, embedding_dim=1280):
    """Load all embeddings from SQLite cache into a dict keyed by sequence hash."""
    cache = SQLiteEmbeddingCache(db_path)
    emb_dict = {}
    hits = 0
    misses = 0
    for i in range(len(dataset)):
        seq = dataset.sequences[i]
        seq_hash = compute_sequence_hash(seq)
        if seq_hash not in emb_dict:
            emb = cache.get(seq, embedding_dim)
            if emb is not None:
                emb_dict[seq_hash] = emb
                hits += 1
            else:
                misses += 1
    logger.info(f"Preloaded {hits} embeddings from cache, {misses} cache misses")
    if misses > 0:
        logger.warning(f"{misses} sequences have no cached embeddings - they will use ESM-2 forward pass")
    return emb_dict


class CachedEmbeddingDataset(Dataset):
    """Wraps a ProteinDataset and injects pre-loaded embeddings."""

    def __init__(self, base_dataset, emb_dict):
        self.base_dataset = base_dataset
        self.emb_dict = emb_dict

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        seq = self.base_dataset.sequences[idx]
        seq_hash = compute_sequence_hash(seq)
        if seq_hash in self.emb_dict:
            emb = self.emb_dict[seq_hash]
            # Trim or pad embedding to match sequence length
            seq_len = sample["length"]
            if emb.size(0) >= seq_len:
                sample["cached_embedding"] = emb[:seq_len]
            else:
                # Pad if embedding is shorter (shouldn't happen normally)
                padded = torch.zeros(seq_len, emb.size(1))
                padded[:emb.size(0)] = emb
                sample["cached_embedding"] = padded
        return sample


# ─── Training loop ──────────────────────────────────────────────────
def train_experiment(
    model,
    train_loader,
    val_loader,
    q3_loss_fn,
    q8_loss_fn,
    config,
    device,
    num_epochs,
    experiment_name,
    save_dir,
    q3_weight=0.5,
    q8_weight=0.5,
):
    """Run a training experiment and return the best checkpoint path."""
    model.to(device)
    model.train()

    # Get downstream parameters only
    if hasattr(model, "get_downstream_parameters"):
        params = model.get_downstream_parameters()
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    best_val_q3 = 0.0
    best_checkpoint_path = None

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_metrics = ProteinMetrics(device=str(device))
        train_losses = []

        for batch_idx, batch in enumerate(train_loader):
            embeddings = batch.get("embeddings")
            if embeddings is not None:
                embeddings = embeddings.to(device)

            attention_mask = batch["attention_mask"].to(device)
            seq_lengths = batch["seq_length"].to(device)

            outputs = model(
                sequences=batch.get("sequence") if embeddings is None else None,
                embeddings=embeddings,
                attention_mask=attention_mask,
                seq_lengths=seq_lengths,
            )

            q3_loss = q3_loss_fn(outputs["q3_logits"], batch["q3_labels"].to(device))
            q8_loss = q8_loss_fn(outputs["q8_logits"], batch["q8_labels"].to(device))
            total_loss = q3_weight * q3_loss + q8_weight * q8_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

            with torch.no_grad():
                train_metrics.update(
                    q3_preds=outputs["q3_preds"].cpu(),
                    q3_targets=batch["q3_labels"],
                    q8_preds=outputs["q8_preds"].cpu(),
                    q8_targets=batch["q8_labels"],
                )

            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"  [{experiment_name}] Epoch {epoch} batch {batch_idx+1}/{len(train_loader)} "
                    f"loss={total_loss.item():.4f}"
                )

        train_results = train_metrics.compute()
        avg_train_loss = sum(train_losses) / len(train_losses)
        epoch_time = time.time() - epoch_start

        logger.info(
            f"[{experiment_name}] Epoch {epoch}/{num_epochs} "
            f"train_loss={avg_train_loss:.4f} "
            f"train_Q3={train_results['q3_accuracy']:.4f} "
            f"train_Q8={train_results['q8_accuracy']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        # Validation
        model.eval()
        val_metrics = ProteinMetrics(device=str(device))
        with torch.no_grad():
            for batch in val_loader:
                embeddings = batch.get("embeddings")
                if embeddings is not None:
                    embeddings = embeddings.to(device)

                attention_mask = batch["attention_mask"].to(device)
                seq_lengths = batch["seq_length"].to(device)

                outputs = model(
                    sequences=batch.get("sequence") if embeddings is None else None,
                    embeddings=embeddings,
                    attention_mask=attention_mask,
                    seq_lengths=seq_lengths,
                )

                val_metrics.update(
                    q3_preds=outputs["q3_preds"].cpu(),
                    q3_targets=batch["q3_labels"],
                    q8_preds=outputs["q8_preds"].cpu(),
                    q8_targets=batch["q8_labels"],
                )

        val_results = val_metrics.compute()
        val_q3 = val_results["q3_accuracy"]
        val_q8 = val_results["q8_accuracy"]

        logger.info(
            f"[{experiment_name}] Epoch {epoch} val_Q3={val_q3:.4f} val_Q8={val_q8:.4f} "
            f"val_MCC={val_results.get('q3_mcc', 0):.4f}"
        )

        scheduler.step(val_q3)

        # Save best checkpoint
        if val_q3 > best_val_q3:
            best_val_q3 = val_q3
            best_checkpoint_path = os.path.join(save_dir, f"{experiment_name}_checkpoint.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": {
                    "val_q3_accuracy": val_q3,
                    "val_q8_accuracy": val_q8,
                    "val_q3_mcc": val_results.get("q3_mcc", 0),
                    "train_loss": avg_train_loss,
                },
                "experiment": experiment_name,
            }, best_checkpoint_path)
            logger.info(f"  Saved best checkpoint → {best_checkpoint_path} (val_Q3={val_q3:.4f})")

    return best_checkpoint_path, best_val_q3


# ─── Evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, model_config, test_loader, device, output_path, dataset_name="CB513"):
    """Load checkpoint and evaluate on test set."""
    model = ProtIntelModel(config=model_config, device=device)
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    metrics = ProteinMetrics(device=str(device))

    for batch in test_loader:
        embeddings = batch.get("embeddings")
        if embeddings is not None:
            embeddings = embeddings.to(torch.device(device))

        attention_mask = batch["attention_mask"].to(device)
        seq_lengths = batch["seq_length"].to(device)

        outputs = model(
            sequences=batch.get("sequence") if embeddings is None else None,
            embeddings=embeddings,
            attention_mask=attention_mask,
            seq_lengths=seq_lengths,
        )

        metrics.update(
            q3_preds=outputs["q3_preds"].cpu(),
            q3_targets=batch["q3_labels"],
            q8_preds=outputs["q8_preds"].cpu(),
            q8_targets=batch["q8_labels"],
        )

    results = metrics.compute()
    results["dataset"] = dataset_name
    results["checkpoint"] = str(checkpoint_path)

    # Save JSON (filter serializable values)
    serializable = {k: v for k, v in results.items() if isinstance(v, (float, int, str, list))}
    save_json(serializable, output_path)
    logger.info(f"Evaluation results saved to {output_path}")

    return results


# ─── Main ───────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ProtIntel Q8 Loss Experiments")
    print("=" * 70)

    save_dir = str(PROJECT_ROOT / "models" / "experiments")
    os.makedirs(save_dir, exist_ok=True)

    config = load_config()
    device = "cpu"
    num_epochs = 5
    batch_size = 16

    # ── 1. Load datasets ───────────────────────────────────────────
    print("\n[1/6] Loading datasets...")
    dm = DataModule.from_config(config.data, batch_size=batch_size)
    dm.setup("fit")
    dm.setup("test")
    print(f"  Train: {len(dm.train_dataset)} samples")
    print(f"  Val:   {len(dm.val_dataset)} samples")
    print(f"  Test:  {len(dm.test_dataset)} samples")

    # ── 2. Pre-load embeddings from SQLite cache ───────────────────
    print("\n[2/6] Pre-loading embeddings from SQLite cache...")
    db_path = PROJECT_ROOT / "datasets" / "processed" / "embeddings_650m" / "embedding_cache.db"
    if not db_path.exists():
        # Fallback to embeddings dir
        db_path = PROJECT_ROOT / "datasets" / "processed" / "embeddings" / "embedding_cache.db"

    emb_dim = config.model.esm2.embedding_dim  # 1280

    train_emb_dict = preload_embeddings(dm.train_dataset, db_path, emb_dim)
    val_emb_dict = preload_embeddings(dm.val_dataset, db_path, emb_dim)
    test_emb_dict = preload_embeddings(dm.test_dataset, db_path, emb_dim)

    # Merge all into one dict (sequences might overlap)
    all_emb_dict = {**train_emb_dict, **val_emb_dict, **test_emb_dict}

    # Wrap datasets with cached embeddings
    cached_train = CachedEmbeddingDataset(dm.train_dataset, all_emb_dict)
    cached_val = CachedEmbeddingDataset(dm.val_dataset, all_emb_dict)
    cached_test = CachedEmbeddingDataset(dm.test_dataset, all_emb_dict)

    train_loader = DataLoader(
        cached_train, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )
    val_loader = DataLoader(
        cached_val, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )
    test_loader = DataLoader(
        cached_test, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_fn, drop_last=False,
    )

    # Verify embeddings are in batches
    sample_batch = next(iter(train_loader))
    has_embs = "embeddings" in sample_batch
    print(f"  Embeddings in batch: {has_embs}")
    if has_embs:
        print(f"  Embedding shape: {sample_batch['embeddings'].shape}")
    else:
        print("  WARNING: Embeddings not found in batch - will use ESM-2 forward pass (slow!)")

    # ── 3. Compute Q8 class weights ────────────────────────────────
    print("\n[3/6] Computing Q8 class weights...")
    q8_weights = dm.get_class_weights("q8")
    print(f"  Q8 inverse-frequency weights: {q8_weights.tolist()}")

    # ── 4. Experiment 1: Class-Weighted CE ─────────────────────────
    print("\n[4/6] Experiment 1: Q8 Class-Weighted Cross-Entropy")
    print("-" * 50)

    model_wce = ProtIntelModel(config=config.model, device=device)
    q3_loss_wce = create_loss_function("cross_entropy", label_smoothing=0.1)
    q8_loss_wce = create_loss_function(
        "cross_entropy", label_smoothing=0.1, class_weights=q8_weights
    )

    wce_ckpt, wce_val_q3 = train_experiment(
        model=model_wce,
        train_loader=train_loader,
        val_loader=val_loader,
        q3_loss_fn=q3_loss_wce,
        q8_loss_fn=q8_loss_wce,
        config=config.training,
        device=device,
        num_epochs=num_epochs,
        experiment_name="q8_weighted",
        save_dir=save_dir,
    )

    # Free model memory
    del model_wce
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 5. Experiment 2: Focal Loss ────────────────────────────────
    print(f"\n[5/6] Experiment 2: Q8 Focal Loss (gamma=2.0)")
    print("-" * 50)

    model_focal = ProtIntelModel(config=config.model, device=device)
    q3_loss_focal = create_loss_function("cross_entropy", label_smoothing=0.1)
    q8_loss_focal = create_loss_function("focal", focal_gamma=2.0)

    focal_ckpt, focal_val_q3 = train_experiment(
        model=model_focal,
        train_loader=train_loader,
        val_loader=val_loader,
        q3_loss_fn=q3_loss_focal,
        q8_loss_fn=q8_loss_focal,
        config=config.training,
        device=device,
        num_epochs=num_epochs,
        experiment_name="q8_focal",
        save_dir=save_dir,
    )

    del model_focal
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 6. Evaluate both on CB513 ──────────────────────────────────
    print(f"\n[6/6] Evaluating checkpoints on CB513...")
    print("-" * 50)

    results = {}

    if wce_ckpt:
        print(f"  Evaluating class-weighted CE checkpoint...")
        wce_results = evaluate_checkpoint(
            wce_ckpt, config.model, test_loader, device,
            os.path.join(save_dir, "cb513_q8_weighted.json"),
        )
        results["weighted_ce"] = wce_results
        print(f"  → Q3={wce_results['q3_accuracy']:.4f} Q8={wce_results['q8_accuracy']:.4f} MCC={wce_results.get('q3_mcc', 0):.4f}")

    if focal_ckpt:
        print(f"  Evaluating focal loss checkpoint...")
        focal_results = evaluate_checkpoint(
            focal_ckpt, config.model, test_loader, device,
            os.path.join(save_dir, "cb513_q8_focal.json"),
        )
        results["focal"] = focal_results
        print(f"  → Q3={focal_results['q3_accuracy']:.4f} Q8={focal_results['q8_accuracy']:.4f} MCC={focal_results.get('q3_mcc', 0):.4f}")

    # ── Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<25} {'Q3 Acc':>10} {'Q8 Acc':>10} {'MCC':>10}")
    print("-" * 55)
    print(f"{'Baseline (existing)':25} {'69.42%':>10} {'34.03%':>10} {'0.527':>10}")
    if "weighted_ce" in results:
        r = results["weighted_ce"]
        print(f"{'Q8 Weighted CE':<25} {r['q3_accuracy']*100:>9.2f}% {r['q8_accuracy']*100:>9.2f}% {r.get('q3_mcc', 0):>10.3f}")
    if "focal" in results:
        r = results["focal"]
        print(f"{'Q8 Focal (γ=2.0)':<25} {r['q3_accuracy']*100:>9.2f}% {r['q8_accuracy']*100:>9.2f}% {r.get('q3_mcc', 0):>10.3f}")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
