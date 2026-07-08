"""CLI single-sequence inference for ProtIntel.

Usage:
    python infer.py MKFLILLFNILCLFPVLAADNHGVSMNAS
    python infer.py --file input.fasta --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fasta_parser import parse_fasta_string, read_fasta
from src.utils.config_loader import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import get_device

logger = get_logger(__name__)


def main() -> None:
    """Main CLI inference entry point."""
    parser = argparse.ArgumentParser(description="ProtIntel single-sequence inference.")
    parser.add_argument("sequence", nargs="?", type=str, help="Amino acid sequence")
    parser.add_argument("--file", type=str, help="FASTA file path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--device", type=str, default="auto", help="Device")
    parser.add_argument("--xai", action="store_true", help="Compute XAI attributions")
    parser.add_argument("--xai-method", type=str, default="ig", help="XAI method")
    args = parser.parse_args()

    if not args.sequence and not args.file:
        parser.print_help()
        print("\nExample: python infer.py MKFLILLFNILCLFPVLAADNHGVSMNAS")
        sys.exit(1)

    config = load_config()
    device = get_device(args.device)

    from backend.services.inference_service import InferenceService

    checkpoint = args.checkpoint or config.inference.checkpoint_path
    service = InferenceService(
        checkpoint_path=checkpoint,
        device=str(device),
        model_config=config.model,
    )
    service.load_model()

    # Get sequences
    sequences: list[tuple[str, str]] = []
    if args.file:
        records = read_fasta(args.file)
        sequences = [(r.protein_id, r.sequence) for r in records]
    elif args.sequence:
        # Check if it's FASTA format
        if args.sequence.startswith(">"):
            records = parse_fasta_string(args.sequence)
            sequences = [(r.protein_id, r.sequence) for r in records]
        else:
            sequences = [("query", args.sequence)]

    for pid, seq in sequences:
        print(f"\n{'='*60}")
        print(f"Protein: {pid}")
        print(f"Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print(f"Length: {len(seq)} residues")
        print(f"{'='*60}")

        result = service.predict(
            sequence=seq,
            return_xai=args.xai,
            xai_method=args.xai_method,
        )

        # Display Q3 prediction
        print(f"\nQ3 Prediction:")
        print(f"  Seq:  {''.join(list(result['sequence']))}")
        print(f"  Q3:   {''.join(result['q3_prediction'])}")
        print(f"  Q8:   {''.join(result['q8_prediction'])}")

        # Summary statistics
        q3_preds = result["q3_prediction"]
        h_count = q3_preds.count("H")
        e_count = q3_preds.count("E")
        c_count = q3_preds.count("C")
        total = len(q3_preds)

        print(f"\nComposition:")
        print(f"  Helix (H): {h_count:>4d} ({100*h_count/total:.1f}%)")
        print(f"  Sheet (E): {e_count:>4d} ({100*e_count/total:.1f}%)")
        print(f"  Coil  (C): {c_count:>4d} ({100*c_count/total:.1f}%)")

        avg_conf = sum(result["confidence"]) / len(result["confidence"])
        print(f"\nAverage confidence: {avg_conf:.4f}")
        print(f"Processing time: {result['processing_time_ms']:.1f} ms")

        if "residue_importance" in result:
            importance = result["residue_importance"]
            top_k = sorted(
                enumerate(importance), key=lambda x: x[1], reverse=True
            )[:5]
            print(f"\nTop-5 important residues ({args.xai_method}):")
            for pos, score in top_k:
                print(f"  Position {pos+1}: {result['sequence'][pos]} "
                      f"(importance={score:.4f})")


if __name__ == "__main__":
    main()
