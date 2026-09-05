from __future__ import annotations

import math
from typing import Iterator, Sized


class ZeroPaddedDistributedSampler:
    """Equal-length distributed shards without giving padded duplicates weight.

    Yields `(dataset_index, multiplier)`. Real examples use 1.0; indices added
    only to equalize FSDP step counts use 0.0. The dataset must multiply its
    scalar sample weight by this multiplier.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        num_replicas: int,
        rank: int,
        shuffle: bool,
        seed: int = 0,
    ) -> None:
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("invalid distributed rank specification")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self) -> Iterator[tuple[int, float]]:
        if len(self.dataset) == 0:
            return iter(())
        if self.shuffle:
            import torch

            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        else:
            indices = list(range(len(self.dataset)))
        weighted = [(index, 1.0) for index in indices]
        padding = self.total_size - len(weighted)
        weighted.extend((indices[offset % len(indices)], 0.0) for offset in range(padding))
        return iter(weighted[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
