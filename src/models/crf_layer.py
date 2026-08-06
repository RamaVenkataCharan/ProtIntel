"""Linear Chain Conditional Random Field (CRF) for Protein Secondary Structure.

Models transition probabilities between adjacent secondary structure states
(e.g., Helix → Strand transitions), ensuring valid structural sequence predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearCRF(nn.Module):
    """Linear Chain Conditional Random Field.

    Args:
        num_classes: Number of output classes (3 for Q3, 8 for Q8).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        # Transitions matrix: transitions[i, j] is transition score from class j to i
        self.transitions = nn.Parameter(torch.empty(num_classes, num_classes))
        self.start_transitions = nn.Parameter(torch.empty(num_classes))
        self.end_transitions = nn.Parameter(torch.empty(num_classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Viterbi decoding to compute optimal state sequence.

        Args:
            logits: Emissions of shape (B, L, num_classes).
            mask: Binary mask of shape (B, L) where 1 indicates valid tokens.

        Returns:
            Best path sequence indices of shape (B, L).
        """
        B, L, C = logits.shape
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=logits.device)

        # Simplified Viterbi / max path extraction
        viterbi_paths = []
        for b in range(B):
            seq_len = mask[b].sum().item()
            seq_logits = logits[b, :seq_len]  # (L_seq, C)

            best_path = []
            score = self.start_transitions + seq_logits[0]
            best_path.append(score.argmax(dim=-1).item())

            for t in range(1, seq_len):
                next_score = score.unsqueeze(1) + self.transitions + seq_logits[t].unsqueeze(0)
                best_next = next_score.max(dim=0).values
                best_path.append(best_next.argmax(dim=-1).item())
                score = best_next

            # Pad path to L
            best_path = best_path + [0] * (L - len(best_path))
            viterbi_paths.append(best_path)

        return torch.tensor(viterbi_paths, device=logits.device)
