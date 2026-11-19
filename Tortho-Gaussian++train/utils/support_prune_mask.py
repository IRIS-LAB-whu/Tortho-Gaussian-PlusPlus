"""CPU-only orthographic center-grid low-support ratios; no renderer imports."""
from pathlib import Path
import re

import numpy as np


MASK_GRID_SIZE = 256
MASK_RATIO = 0.4
MASK_MARGIN = 1.03
PROJECTION_CHUNK = 65536


def load_render_orientation(model_path):
    """Read the existing render_dummy.py cache, without repairing its matrix."""
    root = Path(model_path)
    path = root / "render_direction.txt"
    # The standalone command also accepts an iteration directory (or its export).
    if not path.is_file():
        for directory in (root,) + tuple(root.parents):
            if re.fullmatch(r"iteration_\d+", directory.name) and directory.parent.name == "point_cloud":
                path = directory.parent.parent / "render_direction.txt"
                break
    if not path.is_file():
        raise ValueError("Mask pruning requires an existing render_direction.txt: {}".format(path))
    values = np.loadtxt(str(path), dtype=np.float64)
    if values.shape != (4, 3) or not np.isfinite(values).all():
        raise ValueError("Invalid render direction cache {}; expected 4 finite rows of 3 values.".format(path))
    norm = np.linalg.norm(values[0])
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid render direction in {}".format(path))
    rotation = values[1:]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
            or not np.allclose(rotation[:, 2], values[0] / norm, atol=1e-5)):
        raise ValueError("Invalid orientation in render direction cache: {}".format(path))
    return rotation


def validate_grid_options(ratio, grid_size):
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("--support_prune_mask_ratio must be finite and in (0, 1].")
    if (isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer))
            or not 1 <= grid_size <= np.iinfo(np.int32).max):
        raise ValueError("--support_prune_mask_grid_size must be a positive 32-bit integer.")


def _project_centers(records, rotation):
    """Project xyz only; Gaussian scale, rotation and opacity do not affect votes."""
    fields = ("x", "y", "z")
    if any(name not in (records.dtype.names or ()) for name in fields):
        raise ValueError("Mask pruning requires xyz in the Gaussian PLY.")
    centers = np.empty((len(records), 2), dtype=np.float64)
    origin = np.array([records[name][0] for name in fields], dtype=np.float64) if len(records) else np.zeros(3)
    for start in range(0, len(records), PROJECTION_CHUNK):
        end = min(start + PROJECTION_CHUNK, len(records))
        values = np.column_stack([records[name][start:end] for name in fields]).astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite Gaussian geometry; refusing mask pruning.")
        centers[start:end] = (values - origin) @ rotation[:, :2]
    if not np.isfinite(centers).all():
        raise ValueError("Projected Gaussian centers overflowed.")
    return centers


def _grid_ids(centers, grid_size):
    """Each center belongs to exactly one cell in the unfiltered model bounds."""
    if not len(centers):
        return np.empty(0, dtype=np.int64), 1, 1
    low, high = centers.min(axis=0), centers.max(axis=0)
    extent = high - low
    # Degenerate synthetic/empty-plane models still need a usable mask.
    extent = np.maximum(extent, 1e-6)
    size = np.maximum(1, np.rint(grid_size * (extent / extent.max()))).astype(np.int64)
    lower = (low + high) * 0.5 - extent * MASK_MARGIN * 0.5
    coordinates = (centers - lower) / (extent * MASK_MARGIN) * size
    if not np.isfinite(coordinates).all():
        raise ValueError("Projected grid coordinates overflowed.")
    cells = np.floor(np.clip(coordinates, 0, size - 1)).astype(np.int64)
    return cells[:, 1] * size[0] + cells[:, 0], int(size[0]), int(size[1])


def orthographic_mask_keep(records, low_score, rotation, valid,
                           ratio=MASK_RATIO, grid_size=MASK_GRID_SIZE):
    """Delete whole cells whose low-score / valid-score count reaches the ratio.

    Counts use centers only, once per Gaussian. Unknown scores do not vote, but
    are removed if their center falls in a selected cell. Low scores in other
    cells survive. There is no depth test, footprint expansion or propagation.
    """
    validate_grid_options(ratio, grid_size)
    low_score, valid = np.asarray(low_score, dtype=bool), np.asarray(valid, dtype=bool)
    if low_score.shape != (len(records),) or valid.shape != (len(records),):
        raise ValueError("Support row counts must match the Gaussian PLY.")
    if np.any(low_score & ~valid):
        raise ValueError("Low-score votes must have valid support.")
    print("[SupportMask] Projecting {} centers; grid long edge {}; low-score ratio >= {:.2%}.".format(
        len(records), grid_size, ratio), flush=True)
    cell_ids, width, height = _grid_ids(_project_centers(records, rotation), grid_size)
    # Only occupied cells are allocated, even for a user-supplied large grid.
    occupied, inverse = np.unique(cell_ids, return_inverse=True)
    valid_counts = np.bincount(inverse[valid], minlength=len(occupied))
    low_counts = np.bincount(inverse[low_score], minlength=len(occupied))
    fractions = np.divide(low_counts, valid_counts, out=np.zeros(len(occupied), dtype=np.float64),
                          where=valid_counts > 0)
    selected = (valid_counts > 0) & (fractions >= ratio)
    removed = selected[inverse]
    stats = dict(low_score_count=int(low_score.sum()), low_score_removed=int((removed & low_score).sum()),
                 low_score_kept=int((~removed & low_score).sum()),
                 extra_removed=int((removed & ~low_score).sum()),
                 occupied_cells=len(occupied), valid_cells=int((valid_counts > 0).sum()),
                 selected_cells=int(selected.sum()), grid_width=width, grid_height=height,
                 ratio=float(ratio), grid_size=int(grid_size))
    print("[SupportMask] Grid {}x{}; cells with valid scores {}; selected cells {}; low-score points {}.".format(
        width, height, stats["valid_cells"], stats["selected_cells"], stats["low_score_count"]), flush=True)
    return ~removed, stats
