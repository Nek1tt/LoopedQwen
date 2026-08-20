from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    """Memory-mapped uint16 token stream produced by prepare_data.py."""

    def __init__(self, path: str | Path, sequence_length: int) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing token file: {self.path}")
        self.tokens = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.sequence_length = sequence_length
        if len(self.tokens) <= sequence_length + 1:
            raise ValueError(f"{self.path} is too small for sequence length {sequence_length}")

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.tokens) - self.sequence_length - 1
        starts = torch.randint(0, max_start, (batch_size,), generator=generator).tolist()
        chunks = [
            np.asarray(self.tokens[i : i + self.sequence_length + 1], dtype=np.int64)
            for i in starts
        ]
        batch = torch.from_numpy(np.stack(chunks))
        x = batch[:, :-1].to(device, non_blocking=True)
        y = batch[:, 1:].to(device, non_blocking=True)
        return x, y

    def get_sequential_batch(
        self,
        batch_size: int,
        batch_index: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return non-overlapping contexts; batch_index makes resume exact."""
        tokens_per_batch = batch_size * self.sequence_length
        start = batch_index * tokens_per_batch
        end = start + tokens_per_batch
        if end > len(self.tokens):
            raise IndexError(
                f"Sequential batch {batch_index} ends at token {end:,}, "
                f"but {self.path} contains only {len(self.tokens):,} tokens"
            )
        array = np.asarray(self.tokens[start:end], dtype=np.int64).reshape(
            batch_size, self.sequence_length
        )
        return torch.from_numpy(array).to(device, non_blocking=True)
