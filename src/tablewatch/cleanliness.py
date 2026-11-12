"""Is an empty table clean? Compare it with what the same table looked like when clean.

No "dirty table" training data is needed. With a fixed camera, a table's clean appearance
is a known reference image; used plates, cups and rubbish show up as local structure that
was not there before. This also handles restaurants where a clean table is not a bare
table (pre-set bowls, condiment racks): the place setting is part of the reference.

The comparison is done per cell of a grid with normalised cross-correlation, which ignores
overall brightness changes (clouds, lights switched on) and reacts to new objects.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def prepare(crop: np.ndarray, size: int = 96) -> np.ndarray:
    """uint8 image crop (grey or 3-channel) -> size x size float32 grey image."""
    im = Image.fromarray(np.ascontiguousarray(crop)).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)


def changed_cells(ref: np.ndarray, cur: np.ndarray, grid: int = 8, flat_std: float = 5.0,
                  ncc_threshold: float = 0.5, flat_shift: float = 25.0) -> np.ndarray:
    """grid x grid boolean map: which parts of the table look different from the reference."""
    cur = cur * ((np.median(ref) + 1.0) / (np.median(cur) + 1.0))  # undo global brightness change
    c = ref.shape[0] // grid

    def cells(a):
        return a[:grid * c, :grid * c].reshape(grid, c, grid, c).transpose(0, 2, 1, 3).reshape(grid, grid, c * c)

    r, k = cells(ref), cells(cur)
    r_mean, k_mean, r_std, k_std = r.mean(-1), k.mean(-1), r.std(-1), k.std(-1)
    ncc = ((r - r_mean[..., None]) * (k - k_mean[..., None])).mean(-1) / (r_std * k_std + 1e-6)

    r_flat, k_flat = r_std < flat_std, k_std < flat_std
    both_flat, one_flat = r_flat & k_flat, r_flat ^ k_flat
    textured = ~(r_flat | k_flat)
    return ((both_flat & (np.abs(r_mean - k_mean) > flat_shift))      # plain surface covered by something plain
            | (one_flat & (np.abs(r_std - k_std) > flat_std))         # object appeared on / vanished from plain surface
            | (textured & (ncc < ncc_threshold)))                     # pattern replaced by a different pattern


class ReferenceScorer:
    """Keeps one clean reference per table and scores new crops against it (0 = identical)."""

    def __init__(self, size: int = 96, grid: int = 8, warmup_frames: int = 5, adapt_rate: float = 0.02):
        self.size, self.grid, self.warmup_frames, self.adapt_rate = size, grid, warmup_frames, adapt_rate
        self.refs: dict[str, np.ndarray] = {}
        self._warmup: dict[str, list[np.ndarray]] = {}

    def ready(self, table_id: str) -> bool:
        return table_id in self.refs

    def set_reference(self, table_id: str, crop: np.ndarray) -> None:
        self.refs[table_id] = prepare(crop, self.size)

    def warm_up(self, table_id: str, crop: np.ndarray) -> None:
        """Collect the first few unoccupied views of a table; their mean becomes the reference."""
        buf = self._warmup.setdefault(table_id, [])
        buf.append(prepare(crop, self.size))
        if len(buf) >= self.warmup_frames:
            self.refs[table_id] = np.mean(buf, axis=0)
            del self._warmup[table_id]

    def cells(self, table_id: str, crop: np.ndarray) -> np.ndarray:
        return changed_cells(self.refs[table_id], prepare(crop, self.size), self.grid)

    def score(self, table_id: str, crop: np.ndarray) -> float:
        """Share of the table's area that differs from its clean reference."""
        return float(self.cells(table_id, crop).mean())

    def adapt(self, table_id: str, crop: np.ndarray) -> None:
        """Slowly follow lighting drift. Call only while the table is known to be clean."""
        a = self.adapt_rate
        self.refs[table_id] = (1 - a) * self.refs[table_id] + a * prepare(crop, self.size)
