"""Custom PyTorch Samplers for Protein Secondary Structure Prediction.

Implements BucketSampler to group sequences of similar lengths together,
minimizing unnecessary zero-padding in mini-batches and accelerating training.
Also implements ClassBalancedSampler for handling rare Q8 secondary structure classes.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Sequence

import torch
from torch.utils.data import Sampler


class BucketSampler(Sampler[List[int]]):
    """Groups dataset indices by sequence length into buckets of size ``batch_size``.

    Minimizes zero-padding by ensuring sequences within each mini-batch have similar lengths.

    Args:
        seq_lengths: List or Tensor of sequence lengths for each sample in dataset.
        batch_size: Target mini-batch size.
        shuffle: Whether to shuffle bucket order and sample order within buckets per epoch.
        drop_last: If True, drop the last incomplete batch in a bucket.
    """

    def __init__(
        self,
        seq_lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.seq_lengths = seq_lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Sort indices by length
        self.sorted_indices = sorted(range(len(seq_lengths)), key=lambda i: seq_lengths[i])

    def __iter__(self) -> Iterator[List[int]]:
        # Form mini-batches from length-sorted indices
        batches = [
            self.sorted_indices[i : i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]

        if self.drop_last and len(batches[-1]) < self.batch_size:
            batches.pop()

        if self.shuffle:
            random.shuffle(batches)
            for batch in batches:
                random.shuffle(batch)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class ClassBalancedSampler(Sampler[int]):
    """Weighted sampler that oversamples minority Q8 secondary structure classes.

    Args:
        sample_weights: Per-sample weights computed from inverse class frequencies.
        num_samples: Number of samples to draw per epoch.
    """

    def __init__(self, sample_weights: torch.Tensor, num_samples: int) -> None:
        self.weights = sample_weights.double()
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[int]:
        rand_tensor = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples
