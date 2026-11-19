"""Final-output support records and CPU-only, paired PLY/record filtering."""

import argparse
import contextlib
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback

import numpy as np
from plyfile import PlyData
from utils.support_prune_mask import MASK_GRID_SIZE, MASK_RATIO, validate_grid_options


SUPPORT_FILENAME = "weak_support.npz"
ROW_FIELDS = ("support_score", "weak_dir", "valid", "obs_count")
POSTPROCESS_MASK_OPTION = "--postprocess_support_prune_mask_thresh"


def support_path(ply_path):
    return Path(ply_path).with_name(SUPPORT_FILENAME)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_threshold(value):
    # Values above the score range are allowed for mask selection.
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError("Support pruning threshold must be finite and non-negative.")
    return value


def validate_support_output_options(args, model_path=None, check_render_direction=True):
    mask_threshold = getattr(args, "final_support_prune_mask_thresh", None)
    validate_threshold(mask_threshold)
    if mask_threshold is not None:
        validate_grid_options(getattr(args, "support_prune_mask_ratio", MASK_RATIO),
                              getattr(args, "support_prune_mask_grid_size", MASK_GRID_SIZE))
        if not args.enable_weak_support:
            raise ValueError("--final_support_prune_mask_thresh requires --enable_weak_support")
        root = model_path if model_path is not None else getattr(args, "model_path", None)
        if root is not None and check_render_direction:
            from utils.support_prune_mask import load_render_orientation
            load_render_orientation(root)
    if getattr(args, POSTPROCESS_MASK_OPTION[2:], None) is not None:
        raise ValueError("Use a standalone mask postprocess command with -m and mask options")


def add_postprocess_argument(parser):
    parser.add_argument(POSTPROCESS_MASK_OPTION, type=float, default=None,
                        help="CPU-only center-grid ratio mask export; requires render_direction.txt; accepts -m and support_prune_mask_ratio/grid_size.")


def _validate_rows(data, count):
    for key in ROW_FIELDS:
        shape = (count, 3) if key == "weak_dir" else (count,)
        if key not in data or data[key].shape != shape:
            raise ValueError("Invalid support record shape for {} (expected {}).".format(key, shape))
    if data["valid"].dtype != np.dtype(bool):
        raise ValueError("Support valid field must be boolean.")
    valid = data["valid"]
    if (not np.all(np.isfinite(data["support_score"][valid]))
            or not np.all(np.isfinite(data["weak_dir"][valid]))
            or not np.all(np.isfinite(data["obs_count"]))
            or np.any(data["obs_count"] < 0)):
        raise ValueError("Support records contain invalid numeric values.")


def write_support_data(ply_path, data, iteration, mode="training", destination=None):
    """Write rows in exactly the PLY's order; bind them to its complete contents."""
    count = len(PlyData.read(str(ply_path))["vertex"].data)
    _validate_rows(data, count)
    destination = Path(destination) if destination is not None else support_path(ply_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".weak_support-", suffix=".npz", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **{key: data[key] for key in ROW_FIELDS},
                     schema_version=np.int32(1), iteration=np.int64(iteration),
                     mode=np.asarray(mode), ply_sha256=np.asarray(_sha256(ply_path)))
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(destination)


def load_support_data(ply_path):
    record_path = support_path(ply_path)
    if not record_path.is_file():
        raise FileNotFoundError("Missing support records: {}. Old PLY files alone cannot recover training support.".format(record_path))
    with np.load(str(record_path), allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if int(data.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported support record schema: {}".format(record_path))
    if str(data.get("ply_sha256", "")) != _sha256(ply_path):
        raise ValueError("PLY/support checksum mismatch; refusing to filter misaligned records: {}".format(ply_path))
    _validate_rows(data, len(PlyData.read(str(ply_path))["vertex"].data))
    if str(data.get("mode", "")) not in ("training", "score_only") or "iteration" not in data:
        raise ValueError("Invalid support record metadata.")
    return data


def export_support_colored_ply(ply_path, score_to_rgb):
    """Create a diagnostic SH-colored copy of the saved final model."""
    from utils.sh_utils import RGB2SH

    ply_path = Path(ply_path)
    output_path = ply_path.with_name("point_cloud_weak_support.ply")
    if output_path.resolve() == ply_path.resolve():
        raise ValueError("The diagnostic PLY must not overwrite its source.")
    support = load_support_data(ply_path)
    ply = PlyData.read(str(ply_path), mmap="r")
    if any(element.name != "vertex" for element in ply.elements):
        raise ValueError("Expected a vertex-only Gaussian PLY.")
    names = ply["vertex"].data.dtype.names
    dc_names = ("f_dc_0", "f_dc_1", "f_dc_2")
    if any(name not in names for name in dc_names):
        raise ValueError("Gaussian PLY is missing its three DC SH color coefficients.")

    # Copy before touching SH fields: neither the source PLY nor NPZ is changed.
    records = ply["vertex"].data.copy()
    valid = support["valid"]
    colors = np.full((len(records), 3), 0.5, dtype=np.float32)
    if np.any(valid):
        colors[valid] = score_to_rgb(support["support_score"][valid])
    dc = RGB2SH(colors)
    for channel, name in enumerate(dc_names):
        records[name] = dc[:, channel]
    # Higher-order SH would otherwise tint the diagnostic colors by view angle.
    for name in names:
        if name.startswith("f_rest_"):
            records[name] = 0.0
    ply["vertex"].data = records

    fd, temporary = tempfile.mkstemp(prefix=".weak-support-color-", suffix=".ply",
                                     dir=str(ply_path.parent))
    os.close(fd)
    try:
        ply.write(temporary)
        os.replace(temporary, str(output_path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return str(output_path)


def export_gaussian_support(gaussians, ply_path, iteration, mode="training", chunk_size=65536):
    """Recompute a final snapshot without changing any training buffers or parameters."""
    import torch

    if not gaussians.view_support_enabled:
        raise ValueError("Support tracking is not enabled.")
    count = int(gaussians.get_xyz.shape[0])
    scores = np.full(count, np.nan, dtype=np.float32)
    directions = np.zeros((count, 3), dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    obs_count = gaussians.obs_count.detach().cpu().numpy().reshape(-1).copy()
    with torch.no_grad():
        for start in range(0, count, chunk_size):
            end = min(start + chunk_size, count)
            observed = gaussians.obs_count[start:end].reshape(-1) > 0
            indices = torch.nonzero(observed, as_tuple=False).reshape(-1)
            if not indices.numel():
                continue
            matrix = gaussians.obs_tensor[start:end][indices]
            matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
            eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
            trace = eigenvalues.sum(dim=-1).clamp_min(0.0)
            score = eigenvalues[:, 0].clamp_min(0.0) / (trace + 1e-8)
            direction = eigenvectors[:, :, 0]
            finite = torch.isfinite(score) & torch.isfinite(direction).all(dim=-1)
            rows = indices[finite].cpu().numpy() + start
            scores[rows] = score[finite].cpu().numpy()
            directions[rows] = direction[finite].cpu().numpy()
            valid[rows] = True
    return write_support_data(ply_path, dict(support_score=scores, weak_dir=directions,
                                            valid=valid, obs_count=obs_count), iteration, mode)



def export_mask_pruned_support_files(ply_path, threshold, model_path,
                                    ratio=MASK_RATIO, grid_size=MASK_GRID_SIZE):
    """Export a center-grid low-score-ratio selection; never alter source files."""
    from utils.support_prune_mask import load_render_orientation, orthographic_mask_keep

    validate_threshold(threshold)
    validate_grid_options(ratio, grid_size)
    if threshold is None:
        raise ValueError("A mask pruning threshold is required.")
    ply_path = Path(ply_path)
    output_dir = ply_path.parent / "support_prune_mask_{}".format(float(threshold))
    if os.path.lexists(str(output_dir)):
        raise FileExistsError("Refusing to overwrite existing support export: {}".format(output_dir))
    rotation = load_render_orientation(model_path)
    data = load_support_data(ply_path)
    ply = PlyData.read(str(ply_path), mmap="r")
    if any(element.name != "vertex" for element in ply.elements):
        raise ValueError("Only Gaussian vertex-only PLY files can be filtered safely.")
    low_score = data["valid"] & (data["support_score"] < threshold)
    # Keep the export summary while hiding the helper's per-stage diagnostics.
    with contextlib.redirect_stdout(None):
        keep, statistics = orthographic_mask_keep(ply["vertex"].data, low_score, rotation,
                                                 data["valid"], ratio=ratio, grid_size=grid_size)
    result = dict(before=len(keep), after=int(keep.sum()), removed=int((~keep).sum()),
                  output_dir=str(output_dir), **statistics)
    with tempfile.TemporaryDirectory(prefix=".support-mask-export-", dir=str(ply_path.parent)) as temporary:
        staged_ply = Path(temporary) / "point_cloud.ply"
        ply["vertex"].data = ply["vertex"].data[keep]
        ply.write(str(staged_ply))
        write_support_data(staged_ply, {key: data[key][keep] for key in ROW_FIELDS},
                           int(data["iteration"]), str(data["mode"]))
        if os.path.lexists(str(output_dir)):
            raise FileExistsError("Refusing to overwrite existing support export: {}".format(output_dir))
        os.rename(temporary, str(output_dir))
    print("[SupportMask] {} -> {} Gaussians; removed {} (low-score {}, other {}); low-score kept {}; output: {}".format(
        result["before"], result["after"], result["removed"], result["low_score_removed"],
        result["extra_removed"], result["low_score_kept"], output_dir))
    return result


def final_ply_path(model_path, iteration):
    return os.path.join(model_path, "point_cloud", "iteration_{}".format(int(iteration)), "point_cloud.ply")


def find_postprocess_ply(model_path):
    root = Path(model_path)
    if (root / "point_cloud.ply").is_file():
        return root / "point_cloud.ply"
    candidates = []
    for directory in (root / "point_cloud").glob("iteration_*"):
        match = re.fullmatch(r"iteration_(\d+)", directory.name)
        if match and (directory / "point_cloud.ply").is_file():
            candidates.append((int(match.group(1)), directory / "point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError("No point_cloud/iteration_*/point_cloud.ply under {}".format(root))
    return max(candidates, key=lambda pair: pair[0])[1]


def maybe_run_support_postprocess(argv=None):
    """Called before importing torch, Scene, or CUDA extensions in train.py."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Detect abbreviations too, so argparse's training parser cannot accidentally
    # accept an abbreviated standalone flag and start training instead.
    requested = any(token.split("=", 1)[0].startswith("--postprocess") for token in argv)
    if not requested:
        return False
    parser = argparse.ArgumentParser(description="Filter existing Gaussian support records (no training)", allow_abbrev=False)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument(POSTPROCESS_MASK_OPTION, type=float, required=True)
    parser.add_argument("--support_prune_mask_ratio", type=float, default=None,
                        help="Mask cell low-score fraction in (0, 1]; default {}.".format(MASK_RATIO))
    parser.add_argument("--support_prune_mask_grid_size", type=int, default=None,
                        help="Mask grid long edge in cells; default 256.")
    args = parser.parse_args(argv)
    try:
        validate_threshold(args.postprocess_support_prune_mask_thresh)
        args.support_prune_mask_ratio = MASK_RATIO if args.support_prune_mask_ratio is None else args.support_prune_mask_ratio
        args.support_prune_mask_grid_size = MASK_GRID_SIZE if args.support_prune_mask_grid_size is None else args.support_prune_mask_grid_size
        validate_grid_options(args.support_prune_mask_ratio, args.support_prune_mask_grid_size)
        root = Path(args.model_path)
        if not root.is_dir():
            raise ValueError("Model path must be an existing directory: {}".format(root))
    except ValueError as error:
        parser.error(str(error))
    try:
        ply_path = find_postprocess_ply(root)
        export_mask_pruned_support_files(ply_path, args.postprocess_support_prune_mask_thresh, root,
                                         ratio=args.support_prune_mask_ratio,
                                         grid_size=args.support_prune_mask_grid_size)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
    return True
