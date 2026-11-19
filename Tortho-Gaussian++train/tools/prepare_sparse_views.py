#!/usr/bin/env python3
"""Prepare known-pose sparse-view COLMAP models for 3DGS.

The script supports both the complete intermediate layout and the final
undistorted layout produced by this repository's ``convert.py``.  It keeps
all known camera poses for evaluation, selects a small set of training
images, clears the original 3D points, and triangulates a new sparse SfM
model.  Sparse mode stops after triangulation.  Dense mode additionally runs
PatchMatch stereo plus stereo fusion using only the selected images.  If the
original COLMAP database is no longer available, features and matches are
recomputed on the final images.

Each result keeps the layout expected by this repository's 3DGS loader::

    <scene>/<N>_views/
        images/                  # all registered images/poses for evaluation
        sparse/0/{cameras.bin,images.bin,points3D.bin}
        sparse/0/points3D.ply    # dense mode only: fused initialization cloud
        train_views.txt
        test_views.txt
        manifest.json

``points3D.bin`` is the selected-view sparse SfM initialization.  In dense
mode, the fused ``points3D.ply`` deliberately lives in the same ``sparse/0``
folder so the unmodified 3DGS loader uses it instead.  In sparse mode, the
loader converts ``points3D.bin`` to ``points3D.ply`` on first use.

The complete model is read only.  Existing output directories are never
overwritten.
"""

import argparse
import datetime as dt
import json
import os
import shlex
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, NamedTuple, Sequence, Set, TextIO, Tuple


COLMAP_MAX_IMAGE_ID = 2**31 - 1

CAMERA_MODEL_IDS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "FOV": 7,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
    "THIN_PRISM_FISHEYE": 10,
    "RAD_TAN_THIN_PRISM_FISHEYE": 11,
}


class ImageRecord(NamedTuple):
    metadata: str
    observations: str
    image_id: int
    camera_id: int


class TeeStream:
    """Forward Python output to both the terminal and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, data: str) -> int:
        written = self.terminal.write(data)
        self.log_file.write(data)
        return written

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    @property
    def encoding(self) -> str:
        return self.terminal.encoding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create sparse SfM or dense MVS COLMAP initialization point "
            "clouds from selected training views while retaining camera "
            "poses from the complete COLMAP reconstruction."
        )
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--sparse",
        dest="point_cloud_mode",
        action="store_const",
        const="sparse",
        help=(
            "Stop after fixed-pose SfM triangulation and initialize 3DGS "
            "from points3D.bin (default)."
        ),
    )
    mode_group.add_argument(
        "--dense",
        dest="point_cloud_mode",
        action="store_const",
        const="dense",
        help=(
            "Run PatchMatch Stereo and Stereo Fusion after SfM triangulation "
            "and initialize 3DGS from the fused points3D.ply."
        ),
    )
    parser.set_defaults(point_cloud_mode="sparse")
    parser.add_argument(
        "--source-path",
        "-s",
        required=True,
        type=Path,
        help="Scene directory previously processed by convert.py.",
    )
    parser.add_argument(
        "--n-views",
        nargs="+",
        type=int,
        default=[3, 6, 9],
        help="Sparse training-view counts to prepare (default: 3 6 9).",
    )
    parser.add_argument(
        "--hold-every",
        type=int,
        default=8,
        help=(
            "Use every N-th registered image as test data before sampling "
            "training views (default: 8, matching LLFF)."
        ),
    )
    parser.add_argument(
        "--test-list",
        type=Path,
        default=None,
        help=(
            "Optional newline-separated test image list. When provided, it "
            "takes precedence over --hold-every."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Parent directory for <N>_views outputs. Defaults to the scene "
            "directory."
        ),
    )
    parser.add_argument(
        "--pose-model-path",
        type=Path,
        default=None,
        help=(
            "Complete COLMAP model whose registered poses will be retained. "
            "Its default depends on --input-layout."
        ),
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help=(
            "COLMAP database corresponding to the distorted model and images. "
            "Used only for the raw layout."
        ),
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help=(
            "Images corresponding to the pose model. Its default depends on "
            "--input-layout."
        ),
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "raw", "converted"),
        default="auto",
        help=(
            "Input data available: raw reuses distorted/database.db and "
            "input/; converted uses sparse/0 and images/ and recomputes "
            "features. Auto prefers raw when all intermediate files exist "
            "and otherwise uses converted (default: auto)."
        ),
    )
    parser.add_argument(
        "--colmap-executable",
        default="colmap",
        help="COLMAP executable or absolute path (default: colmap).",
    )
    parser.add_argument(
        "--gpu-index",
        default="0",
        help=(
            "COLMAP GPU index used for SIFT extraction/matching in converted "
            "layout and for PatchMatch Stereo in dense mode (default: 0)."
        ),
    )
    parser.add_argument(
        "--use-gpu",
        type=int,
        choices=(0, 1),
        default=1,
        help=(
            "Whether COLMAP SIFT extraction/matching uses CUDA in converted "
            "layout (default: 1). Dense PatchMatch still requires a "
            "CUDA-enabled COLMAP build."
        ),
    )
    parser.add_argument(
        "--copy-policy",
        choices=("copy", "soft-link", "hard-link"),
        default="soft-link",
        help=(
            "How output images are materialized. All registered images are "
            "kept for evaluation (default: soft-link)."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Combined Python/COLMAP log path. Defaults to "
            "<output-root>/prepare_sparse_views.log."
        ),
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help=(
            "Keep copied databases and intermediate models after success, "
            "plus stereo depth maps in dense mode."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected train/test views without creating models.",
    )
    return parser.parse_args()


def enable_output_tee(log_path: Path) -> Tuple[TextIO, TextIO, TextIO]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = TeeStream(original_stderr, log_handle)  # type: ignore[assignment]
    return log_handle, original_stdout, original_stderr


def run_command(command: Sequence[str], command_log: List[List[str]]) -> None:
    command = [str(part) for part in command]
    command_log.append(list(command))
    print("$", shlex.join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def require_path(path: Path, description: str, kind: str) -> Path:
    path = path.expanduser().resolve()
    exists = path.is_file() if kind == "file" else path.is_dir()
    if not exists:
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def read_model_images(images_txt: Path) -> Dict[str, ImageRecord]:
    """Return image name -> complete two-line COLMAP image record."""

    records: Dict[str, ImageRecord] = {}
    with images_txt.open("r", encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if line == "":
                break
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError(f"Invalid COLMAP image record: {stripped}")
            image_name = fields[9]
            if image_name in records:
                raise ValueError(f"Duplicate image name in model: {image_name}")
            observations = handle.readline()
            if observations == "":
                raise ValueError(
                    f"Missing POINTS2D line after image record: {image_name}"
                )
            records[image_name] = ImageRecord(
                metadata=stripped,
                observations=observations.rstrip("\n"),
                image_id=int(fields[0]),
                camera_id=int(fields[8]),
            )
    if not records:
        raise ValueError(f"No registered images found in {images_txt}")
    return records


def read_name_list(path: Path) -> List[str]:
    names: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                names.append(stripped)
    if not names:
        raise ValueError(f"Image list is empty: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Image list contains duplicate names: {path}")
    return names


def evenly_spaced_indices(total: int, count: int) -> List[int]:
    if count <= 0:
        raise ValueError(f"View count must be positive, got {count}")
    if count > total:
        raise ValueError(
            f"Cannot select {count} training views from a pool of {total}"
        )
    if count == 1:
        return [0]
    indices = [round(i * (total - 1) / (count - 1)) for i in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError(
            f"Even sampling produced duplicate indices: total={total}, count={count}"
        )
    return indices


def select_view_names(
    all_names: Sequence[str],
    view_counts: Sequence[int],
    hold_every: int,
    test_list: Path,
) -> Tuple[List[str], Dict[int, List[str]]]:
    all_name_set = set(all_names)
    if test_list is not None:
        test_names = read_name_list(test_list)
        unknown = sorted(set(test_names) - all_name_set)
        if unknown:
            raise ValueError(
                "Test list contains images absent from the pose model: "
                + ", ".join(unknown)
            )
    else:
        if hold_every <= 0:
            raise ValueError("--hold-every must be positive without --test-list")
        test_names = list(all_names[::hold_every])

    test_name_set = set(test_names)
    train_pool = [name for name in all_names if name not in test_name_set]
    selections: Dict[int, List[str]] = {}
    for count in view_counts:
        indices = evenly_spaced_indices(len(train_pool), count)
        selections[count] = [train_pool[index] for index in indices]
    return test_names, selections


def write_name_list(path: Path, names: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for name in names:
            handle.write(f"{name}\n")


def write_patch_match_config(
    path: Path, selected_names: Sequence[str]
) -> None:
    """Write an explicit training-only PatchMatch source-view configuration."""

    if len(selected_names) < 2:
        raise ValueError(
            "Dense stereo requires at least two selected training views"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for reference_name in selected_names:
            if "," in reference_name:
                raise ValueError(
                    "COLMAP PatchMatch configuration cannot safely encode an "
                    f"image name containing a comma: {reference_name}"
                )
            source_names = [
                name for name in selected_names if name != reference_name
            ]
            handle.write(reference_name + "\n")
            handle.write(", ".join(source_names) + "\n")


def initialize_pinhole_dense_workspace(
    stereo_dir: Path, selected_names: Sequence[str]
) -> None:
    """Create the COLMAP MVS workspace skipped for undistorted PINHOLE input."""

    stereo_dir.mkdir(parents=True, exist_ok=True)
    for directory_name in (
        "depth_maps",
        "normal_maps",
        "consistency_graphs",
    ):
        (stereo_dir / directory_name).mkdir(exist_ok=True)
    write_name_list(stereo_dir / "fusion.cfg", selected_names)
    print(
        f"Initialized PINHOLE dense workspace for {len(selected_names)} "
        f"training views: {stereo_dir}"
    )


def read_dense_ply_vertex_count(path: Path) -> int:
    """Validate COLMAP's fused PLY schema and return its vertex count."""

    required_properties = {
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "red",
        "green",
        "blue",
    }
    vertex_count = None
    vertex_properties = set()
    current_element = None
    with path.open("rb") as handle:
        first_line = handle.readline().decode("ascii", errors="strict").strip()
        if first_line != "ply":
            raise ValueError(f"Dense fusion output is not a PLY file: {path}")
        while True:
            raw_line = handle.readline()
            if raw_line == b"":
                raise ValueError(f"PLY header has no end_header marker: {path}")
            line = raw_line.decode("ascii", errors="strict").strip()
            fields = line.split()
            if len(fields) == 3 and fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
                current_element = "vertex"
            elif len(fields) >= 2 and fields[0] == "element":
                current_element = fields[1]
            elif (
                len(fields) >= 3
                and fields[0] == "property"
                and current_element == "vertex"
            ):
                vertex_properties.add(fields[-1])
            elif line == "end_header":
                break

    if vertex_count is None:
        raise ValueError(f"PLY header has no vertex element: {path}")
    if vertex_count == 0:
        raise RuntimeError(f"Dense stereo fusion produced zero points: {path}")
    missing_properties = sorted(required_properties - vertex_properties)
    if missing_properties:
        raise ValueError(
            "Dense PLY is incompatible with the 3DGS loader; missing vertex "
            f"properties {missing_properties}: {path}"
        )
    return vertex_count


def write_seed_model(
    full_text_model: Path,
    seed_model: Path,
    image_records: Dict[str, ImageRecord],
    selected_names: Sequence[str],
) -> None:
    seed_model.mkdir(parents=True, exist_ok=False)
    for source_file in full_text_model.glob("*.txt"):
        if source_file.name not in {"images.txt", "points3D.txt"}:
            shutil.copy2(source_file, seed_model / source_file.name)

    with (seed_model / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write(
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(selected_names)}\n")
        for name in selected_names:
            handle.write(image_records[name].metadata + "\n\n")
    (seed_model / "points3D.txt").write_text("", encoding="utf-8")


def read_model_cameras(cameras_txt: Path) -> List[Tuple[int, int, int, int, bytes]]:
    cameras = []
    with cameras_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 5:
                raise ValueError(f"Invalid COLMAP camera record: {stripped}")
            camera_id = int(fields[0])
            model_name = fields[1]
            if model_name not in CAMERA_MODEL_IDS:
                raise ValueError(
                    f"Unsupported COLMAP camera model for database creation: "
                    f"{model_name}"
                )
            width = int(fields[2])
            height = int(fields[3])
            params = [float(value) for value in fields[4:]]
            params_blob = struct.pack("<" + "d" * len(params), *params)
            cameras.append(
                (camera_id, CAMERA_MODEL_IDS[model_name], width, height, params_blob)
            )
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_txt}")
    return cameras


def read_camera_model_names(cameras_txt: Path) -> Set[str]:
    model_names = set()
    with cameras_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) < 2:
                raise ValueError(f"Invalid COLMAP camera record: {stripped}")
            model_names.add(fields[1])
    if not model_names:
        raise ValueError(f"No cameras found in {cameras_txt}")
    return model_names


def initialize_feature_database(
    database_path: Path,
    full_text_model: Path,
    image_records: Dict[str, ImageRecord],
    selected_names: Sequence[str],
) -> None:
    """Insert exact model camera/image IDs before COLMAP extracts features."""

    cameras = read_model_cameras(full_text_model / "cameras.txt")
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executemany(
            "INSERT INTO cameras(camera_id, model, width, height, params, "
            "prior_focal_length) VALUES (?, ?, ?, ?, ?, 1)",
            cameras,
        )
        connection.executemany(
            "INSERT INTO images(image_id, name, camera_id) VALUES (?, ?, ?)",
            [
                (
                    image_records[name].image_id,
                    name,
                    image_records[name].camera_id,
                )
                for name in selected_names
            ],
        )
        connection.commit()
    finally:
        connection.close()


def database_counts(database_path: Path) -> dict:
    connection = sqlite3.connect(str(database_path))
    try:
        counts = {}
        for table_name in (
            "images",
            "keypoints",
            "descriptors",
            "matches",
            "two_view_geometries",
        ):
            if table_exists(connection, table_name):
                counts[table_name] = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table_name}"
                    ).fetchone()[0]
                )
                if table_name in {"matches", "two_view_geometries"}:
                    total_rows, nonempty_rows = connection.execute(
                        f"SELECT COALESCE(SUM(rows), 0), "
                        f"COALESCE(SUM(rows > 0), 0) FROM {table_name}"
                    ).fetchone()
                    counts[f"{table_name}_correspondences"] = int(total_rows)
                    counts[f"{table_name}_nonempty_pairs"] = int(nonempty_rows)
        return counts
    finally:
        connection.close()


def build_feature_database(
    database_path: Path,
    image_list_path: Path,
    image_path: Path,
    full_text_model: Path,
    image_records: Dict[str, ImageRecord],
    selected_names: Sequence[str],
    colmap_executable: str,
    gpu_index: str,
    use_gpu: int,
    command_log: List[List[str]],
) -> dict:
    write_name_list(image_list_path, selected_names)
    run_command(
        [
            colmap_executable,
            "database_creator",
            "--database_path",
            str(database_path),
        ],
        command_log,
    )
    initialize_feature_database(
        database_path, full_text_model, image_records, selected_names
    )
    run_command(
        [
            colmap_executable,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--image_list_path",
            str(image_list_path),
            "--SiftExtraction.gpu_index",
            gpu_index,
            "--SiftExtraction.use_gpu",
            str(use_gpu),
        ],
        command_log,
    )
    run_command(
        [
            colmap_executable,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
            "--SiftMatching.gpu_index",
            gpu_index,
            "--SiftMatching.use_gpu",
            str(use_gpu),
        ],
        command_log,
    )
    counts = database_counts(database_path)
    if counts.get("keypoints", 0) != len(selected_names):
        raise RuntimeError(
            "Feature extraction did not produce keypoints for every selected image: "
            f"{counts}"
        )
    if (
        len(selected_names) > 1
        and counts.get("two_view_geometries_correspondences", 0) == 0
    ):
        raise RuntimeError(
            "Feature matching produced zero geometrically verified "
            "correspondences. The selected sparse views likely have "
            "insufficient overlap; do not substitute the all-view point cloud "
            "if you need a leakage-free sparse-view experiment."
        )
    return counts


def table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def pair_image_ids(pair_id: int) -> Tuple[int, int]:
    image_id2 = pair_id % COLMAP_MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // COLMAP_MAX_IMAGE_ID
    return image_id1, image_id2


def filter_database(database_path: Path, selected_names: Sequence[str]) -> dict:
    """Remove all non-selected images and image pairs from a copied database."""

    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA foreign_keys=OFF")
        rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        name_to_id = {name: int(image_id) for image_id, name in rows}
        missing = sorted(set(selected_names) - set(name_to_id))
        if missing:
            raise ValueError(
                "Selected images are missing from the COLMAP database: "
                + ", ".join(missing)
            )

        selected_ids: Set[int] = {name_to_id[name] for name in selected_names}
        pair_counts = {}
        for table_name in ("matches", "two_view_geometries"):
            if not table_exists(connection, table_name):
                continue
            pair_rows = connection.execute(
                f"SELECT pair_id, rows FROM {table_name}"
            ).fetchall()
            remove_pairs = []
            keep_count = 0
            keep_correspondences = 0
            keep_nonempty_pairs = 0
            for pair_id_raw, rows_raw in pair_rows:
                pair_id = int(pair_id_raw)
                image_id1, image_id2 = pair_image_ids(pair_id)
                if image_id1 in selected_ids and image_id2 in selected_ids:
                    keep_count += 1
                    rows_count = int(rows_raw)
                    keep_correspondences += rows_count
                    keep_nonempty_pairs += int(rows_count > 0)
                else:
                    remove_pairs.append((pair_id,))
            connection.executemany(
                f"DELETE FROM {table_name} WHERE pair_id=?", remove_pairs
            )
            pair_counts[table_name] = {
                "records": keep_count,
                "nonempty_pairs": keep_nonempty_pairs,
                "correspondences": keep_correspondences,
            }

        placeholders = ",".join("?" for _ in selected_ids)
        selected_id_values = tuple(sorted(selected_ids))
        for table_name in ("keypoints", "descriptors", "pose_priors"):
            if table_exists(connection, table_name):
                connection.execute(
                    f"DELETE FROM {table_name} WHERE image_id NOT IN ({placeholders})",
                    selected_id_values,
                )
        connection.execute(
            f"DELETE FROM images WHERE image_id NOT IN ({placeholders})",
            selected_id_values,
        )
        connection.commit()
        remaining_images = int(
            connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        )
    finally:
        connection.close()

    return {
        "remaining_images": remaining_images,
        "remaining_pairs": pair_counts,
    }


def normalize_sparse_layout(output_dir: Path) -> Path:
    sparse_dir = output_dir / "sparse"
    sparse_zero_dir = sparse_dir / "0"
    if sparse_zero_dir.is_dir():
        return sparse_zero_dir

    model_files = [
        path
        for path in sparse_dir.iterdir()
        if path.is_file() and path.name.startswith(("cameras", "images", "points3D"))
    ]
    if not model_files:
        raise FileNotFoundError(
            f"image_undistorter did not create a COLMAP model in {sparse_dir}"
        )
    sparse_zero_dir.mkdir()
    for model_file in model_files:
        shutil.move(str(model_file), str(sparse_zero_dir / model_file.name))
    return sparse_zero_dir


def materialize_patch_match_model(sparse_model: Path) -> List[Path]:
    """Copy a sparse/0 model to sparse/ while COLMAP dense stereo runs."""

    workspace_sparse_dir = sparse_model.parent
    if sparse_model == workspace_sparse_dir:
        return []

    model_sources = None
    for suffix in (".bin", ".txt"):
        candidates = [
            sparse_model / f"{stem}{suffix}"
            for stem in ("cameras", "images", "points3D")
        ]
        if all(path.is_file() for path in candidates):
            model_sources = candidates
            break
    if model_sources is None:
        raise FileNotFoundError(
            "Cannot expose the sparse model for PatchMatch; expected a "
            f"complete BIN or TXT model in {sparse_model}"
        )

    workspace_model_files = []
    for source_file in model_sources:
        destination = workspace_sparse_dir / source_file.name
        if destination.exists():
            raise FileExistsError(
                f"PatchMatch workspace model file already exists: {destination}"
            )
        shutil.copy2(source_file, destination)
        workspace_model_files.append(destination)
    return workspace_model_files


def write_merged_model(
    full_text_model: Path,
    triangulated_text_model: Path,
    merged_model: Path,
    full_image_records: Dict[str, ImageRecord],
    selected_names: Sequence[str],
) -> None:
    """Keep all poses but only selected-view point observations and points."""

    triangulated_records = read_model_images(
        triangulated_text_model / "images.txt"
    )
    missing = sorted(set(selected_names) - set(triangulated_records))
    if missing:
        raise RuntimeError(
            "Triangulation output dropped selected registered images: "
            + ", ".join(missing)
        )

    merged_model.mkdir(parents=True, exist_ok=False)
    for source_file in full_text_model.glob("*.txt"):
        if source_file.name not in {"images.txt", "points3D.txt"}:
            shutil.copy2(source_file, merged_model / source_file.name)
    shutil.copy2(
        triangulated_text_model / "points3D.txt",
        merged_model / "points3D.txt",
    )

    selected_set = set(selected_names)
    records_in_id_order = sorted(
        full_image_records.items(), key=lambda item: item[1].image_id
    )
    with (merged_model / "images.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write(
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        handle.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        handle.write(f"# Number of images: {len(records_in_id_order)}\n")
        for name, full_record in records_in_id_order:
            observations = (
                triangulated_records[name].observations
                if name in selected_set
                else ""
            )
            # Always use the complete model metadata so poses remain exact.
            handle.write(full_record.metadata + "\n")
            handle.write(observations + "\n")


def validate_image_name(name: str) -> Path:
    relative_path = Path(name)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe image name in COLMAP model: {name}")
    return relative_path


def materialize_images(
    source_images: Path,
    output_images: Path,
    image_names: Sequence[str],
    copy_policy: str,
) -> None:
    output_images.mkdir(parents=True, exist_ok=False)
    for name in image_names:
        relative_path = validate_image_name(name)
        source_file = source_images / relative_path
        if not source_file.is_file():
            raise FileNotFoundError(f"Registered image not found: {source_file}")
        destination = output_images / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy_policy == "copy":
            shutil.copy2(source_file, destination)
        elif copy_policy == "hard-link":
            os.link(source_file, destination)
        else:
            destination.symlink_to(source_file.resolve())


def count_text_model(model_txt: Path) -> Tuple[int, int]:
    image_count = len(read_model_images(model_txt / "images.txt"))
    point_count = 0
    with (model_txt / "points3D.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                point_count += 1
    return image_count, point_count


def prepare_view_count(
    count: int,
    selected_names: Sequence[str],
    test_names: Sequence[str],
    all_names: Sequence[str],
    output_root: Path,
    full_text_model: Path,
    image_records: Dict[str, ImageRecord],
    database_path: Path,
    image_path: Path,
    pose_model_path: Path,
    input_layout: str,
    point_cloud_mode: str,
    undistort_output: bool,
    features_recomputed: bool,
    shared_command_log: Sequence[Sequence[str]],
    colmap_executable: str,
    gpu_index: str,
    copy_policy: str,
    keep_work: bool,
    source_path: Path,
) -> None:
    if point_cloud_mode not in {"sparse", "dense"}:
        raise ValueError(f"Unsupported point-cloud mode: {point_cloud_mode}")
    output_dir = output_root / f"{count}_views"
    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists and will not be overwritten: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    write_name_list(output_dir / "train_views.txt", selected_names)
    write_name_list(output_dir / "test_views.txt", test_names)

    work_dir = output_dir / "_work"
    work_dir.mkdir()
    seed_model = work_dir / "seed_model"
    triangulated_model = work_dir / "triangulated"
    triangulated_model.mkdir()
    triangulated_text_model = work_dir / "triangulated_txt"
    triangulated_text_model.mkdir()
    merged_model = work_dir / "merged_model"
    filtered_database = work_dir / "database.db"
    final_text_model = work_dir / "final_model_txt"
    final_text_model.mkdir()
    command_log: List[List[str]] = [list(command) for command in shared_command_log]
    succeeded = False
    workspace_model_files: List[Path] = []

    try:
        print(
            f"\nPreparing {count}-view {point_cloud_mode} initialization "
            f"in {output_dir}"
        )
        print("Training views:")
        for name in selected_names:
            print(f"  {name}")

        write_seed_model(
            full_text_model, seed_model, image_records, selected_names
        )
        shutil.copy2(database_path, filtered_database)
        database_stats = filter_database(filtered_database, selected_names)
        print(
            "Filtered database: "
            f"{database_stats['remaining_images']} images, "
            f"pairs={database_stats['remaining_pairs']}"
        )
        geometry_stats = database_stats["remaining_pairs"].get(
            "two_view_geometries", {}
        )
        if geometry_stats.get("correspondences", 0) == 0:
            raise RuntimeError(
                f"The {count} selected views have zero geometrically verified "
                "correspondences, so a sparse SfM point cloud cannot be "
                "triangulated without changing the view selection."
            )

        run_command(
            [
                colmap_executable,
                "point_triangulator",
                "--database_path",
                str(filtered_database),
                "--image_path",
                str(image_path),
                "--input_path",
                str(seed_model),
                "--output_path",
                str(triangulated_model),
                "--clear_points",
                "1",
                "--refine_intrinsics",
                "0",
                "--Mapper.fix_existing_images",
                "1",
                "--Mapper.ba_refine_focal_length",
                "0",
                "--Mapper.ba_refine_principal_point",
                "0",
                "--Mapper.ba_refine_extra_params",
                "0",
            ],
            command_log,
        )

        run_command(
            [
                colmap_executable,
                "model_converter",
                "--input_path",
                str(triangulated_model),
                "--output_path",
                str(triangulated_text_model),
                "--output_type",
                "TXT",
            ],
            command_log,
        )
        write_merged_model(
            full_text_model,
            triangulated_text_model,
            merged_model,
            image_records,
            selected_names,
        )

        if undistort_output:
            run_command(
                [
                    colmap_executable,
                    "image_undistorter",
                    "--image_path",
                    str(image_path),
                    "--input_path",
                    str(merged_model),
                    "--output_path",
                    str(output_dir),
                    "--output_type",
                    "COLMAP",
                    "--copy_policy",
                    copy_policy,
                ],
                command_log,
            )
            sparse_model = normalize_sparse_layout(output_dir)
        else:
            sparse_model = output_dir / "sparse" / "0"
            sparse_model.mkdir(parents=True)
            run_command(
                [
                    colmap_executable,
                    "model_converter",
                    "--input_path",
                    str(merged_model),
                    "--output_path",
                    str(sparse_model),
                    "--output_type",
                    "BIN",
                ],
                command_log,
            )
            materialize_images(
                image_path,
                output_dir / "images",
                all_names,
                copy_policy,
            )
            if point_cloud_mode == "dense":
                initialize_pinhole_dense_workspace(
                    output_dir / "stereo", selected_names
                )

        run_command(
            [
                colmap_executable,
                "model_converter",
                "--input_path",
                str(sparse_model),
                "--output_path",
                str(final_text_model),
                "--output_type",
                "TXT",
            ],
            command_log,
        )
        registered_count, point_count = count_text_model(final_text_model)
        if registered_count != len(all_names):
            raise RuntimeError(
                f"Expected {len(all_names)} retained registered images, "
                f"got {registered_count}"
            )
        if point_count == 0:
            raise RuntimeError(
                f"Triangulation produced zero points for the {count}-view setting"
            )

        dense_point_count = None
        if point_cloud_mode == "dense":
            patch_match_config = output_dir / "stereo" / "patch-match.cfg"
            write_patch_match_config(patch_match_config, selected_names)
            workspace_model_files = materialize_patch_match_model(sparse_model)
            fused_ply = work_dir / "fused.ply"
            run_command(
                [
                    colmap_executable,
                    "patch_match_stereo",
                    "--workspace_path",
                    str(output_dir),
                    "--workspace_format",
                    "COLMAP",
                    "--config_path",
                    str(patch_match_config),
                    "--PatchMatchStereo.geom_consistency",
                    "1",
                    "--PatchMatchStereo.gpu_index",
                    gpu_index,
                ],
                command_log,
            )
            run_command(
                [
                    colmap_executable,
                    "stereo_fusion",
                    "--workspace_path",
                    str(output_dir),
                    "--workspace_format",
                    "COLMAP",
                    "--input_type",
                    "geometric",
                    "--output_path",
                    str(fused_ply),
                ],
                command_log,
            )
            dense_point_count = read_dense_ply_vertex_count(fused_ply)
            shutil.copy2(fused_ply, sparse_model / "points3D.ply")

        initialization_point_count = (
            dense_point_count
            if dense_point_count is not None
            else point_count
        )
        initialization_point_cloud = (
            "sparse/0/points3D.ply"
            if dense_point_count is not None
            else "sparse/0/points3D.bin"
        )

        manifest = {
            "protocol": (
                "known_pose_sparse_view_dense_mvs"
                if point_cloud_mode == "dense"
                else "known_pose_sparse_view_sfm"
            ),
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "source_path": str(source_path),
            "pose_model_path": str(pose_model_path),
            "database_path_used": str(database_path),
            "image_path": str(image_path),
            "input_layout": input_layout,
            "point_cloud_mode": point_cloud_mode,
            "input_camera_models": sorted(
                read_camera_model_names(full_text_model / "cameras.txt")
            ),
            "image_undistorter_used": undistort_output,
            "n_views": count,
            "train_views": list(selected_names),
            "test_views": list(test_names),
            "unused_views": sorted(
                set(all_names) - set(selected_names) - set(test_names)
            ),
            "registered_images": registered_count,
            "sparse_points": point_count,
            "dense_points": dense_point_count,
            "initialization_points": initialization_point_count,
            "initialization_point_cloud": initialization_point_cloud,
            "poses_fixed_during_triangulation": True,
            "all_known_poses_retained_for_evaluation": True,
            "dense_stereo_used": point_cloud_mode == "dense",
            "dense_stereo_limited_to_training_views": (
                point_cloud_mode == "dense"
            ),
            "database_filtered_to_training_views": True,
            "features_recomputed_from_converted_images": features_recomputed,
            "commands": command_log,
        }
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        if dense_point_count is None:
            print(
                f"Finished {count}-view sparse model: "
                f"registered_images={registered_count}, "
                f"sparse_points={point_count}"
            )
            print(f"COLMAP model and sparse SfM point cloud: {sparse_model}")
        else:
            print(
                f"Finished {count}-view dense model: "
                f"registered_images={registered_count}, "
                f"sparse_points={point_count}, "
                f"dense_points={dense_point_count}"
            )
            print(f"COLMAP model and dense 3DGS point cloud: {sparse_model}")
        succeeded = True
    finally:
        if succeeded and not keep_work:
            shutil.rmtree(work_dir)
            for workspace_model_file in workspace_model_files:
                workspace_model_file.unlink()
            stereo_dir = output_dir / "stereo"
            if stereo_dir.is_dir():
                shutil.rmtree(stereo_dir)


def main() -> int:
    args = parse_args()

    source_path = require_path(args.source_path, "Source scene", "directory")
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else source_path
    )
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = (
        args.log_file.expanduser().resolve()
        if args.log_file is not None
        else output_root / "prepare_sparse_views.log"
    )
    log_handle, original_stdout, original_stderr = enable_output_tee(log_path)

    try:
        print("=" * 80)
        print(
            f"Sparse-view {args.point_cloud_mode} preprocessing started at "
            f"{dt.datetime.now().astimezone().isoformat()}"
        )
        print(f"Log file: {log_path}")

        raw_pose_model = source_path / "distorted" / "sparse" / "0"
        raw_database = source_path / "distorted" / "database.db"
        raw_images = source_path / "input"
        raw_defaults_available = (
            raw_pose_model.is_dir()
            and raw_database.is_file()
            and raw_images.is_dir()
        )
        if args.input_layout == "auto":
            if args.database_path is not None or raw_defaults_available:
                input_layout = "raw"
            else:
                input_layout = "converted"
        else:
            input_layout = args.input_layout

        if input_layout == "raw":
            pose_default = raw_pose_model
            image_default = raw_images
            database_path = require_path(
                args.database_path
                if args.database_path is not None
                else raw_database,
                "COLMAP database",
                "file",
            )
        else:
            if args.database_path is not None:
                raise ValueError(
                    "--database-path is only valid with --input-layout raw"
                )
            pose_default = source_path / "sparse" / "0"
            image_default = source_path / "images"
            database_path = None

        pose_model_path = require_path(
            args.pose_model_path
            if args.pose_model_path is not None
            else pose_default,
            "Complete COLMAP pose model",
            "directory",
        )
        image_path = require_path(
            args.image_path
            if args.image_path is not None
            else image_default,
            "COLMAP image directory",
            "directory",
        )
        print(f"Resolved input layout: {input_layout}")
        if input_layout == "converted":
            print(
                "Original COLMAP database is unavailable or converted mode was "
                "requested; SIFT features/matches will be recomputed only for "
                "the union of selected training views."
            )
        if not args.n_views:
            raise ValueError("At least one --n-views value is required")
        view_counts = sorted(set(args.n_views))
        if any(count <= 0 for count in view_counts):
            raise ValueError(f"All view counts must be positive: {view_counts}")
        test_list = (
            require_path(args.test_list, "Test image list", "file")
            if args.test_list is not None
            else None
        )

        with tempfile.TemporaryDirectory(prefix="prepare_sparse_views_") as temp_dir:
            full_text_model = Path(temp_dir) / "full_model_txt"
            full_text_model.mkdir()
            initial_commands: List[List[str]] = []
            run_command(
                [
                    args.colmap_executable,
                    "model_converter",
                    "--input_path",
                    str(pose_model_path),
                    "--output_path",
                    str(full_text_model),
                    "--output_type",
                    "TXT",
                ],
                initial_commands,
            )
            image_records = read_model_images(full_text_model / "images.txt")
            camera_model_names = read_camera_model_names(
                full_text_model / "cameras.txt"
            )
            input_images_undistorted = camera_model_names.issubset(
                {"PINHOLE", "SIMPLE_PINHOLE"}
            )
            undistort_output = input_layout == "raw" or not input_images_undistorted
            all_names = sorted(image_records)
            test_names, selections = select_view_names(
                all_names,
                view_counts,
                args.hold_every,
                test_list,
            )

            print(f"Registered images in complete model: {len(all_names)}")
            print(f"Input camera models: {sorted(camera_model_names)}")
            print(f"Output needs image undistortion: {undistort_output}")
            print(f"Test images: {len(test_names)}")
            for count in view_counts:
                print(f"{count}-view selection: {', '.join(selections[count])}")

            if args.dry_run:
                print(
                    f"Dry run complete for {args.point_cloud_mode} mode; "
                    "no sparse-view models were created."
                )
                return 0

            shared_command_log = list(initial_commands)
            if input_layout == "converted":
                union_names = sorted(
                    {
                        name
                        for selected_names in selections.values()
                        for name in selected_names
                    }
                )
                database_path = Path(temp_dir) / "selected_features.db"
                database_stats = build_feature_database(
                    database_path=database_path,
                    image_list_path=Path(temp_dir) / "feature_images.txt",
                    image_path=image_path,
                    full_text_model=full_text_model,
                    image_records=image_records,
                    selected_names=union_names,
                    colmap_executable=args.colmap_executable,
                    gpu_index=args.gpu_index,
                    use_gpu=args.use_gpu,
                    command_log=shared_command_log,
                )
                print(
                    f"Recomputed feature database for {len(union_names)} unique "
                    f"selected views: {database_stats}"
                )

            assert database_path is not None
            for count in view_counts:
                prepare_view_count(
                    count=count,
                    selected_names=selections[count],
                    test_names=test_names,
                    all_names=all_names,
                    output_root=output_root,
                    full_text_model=full_text_model,
                    image_records=image_records,
                    database_path=database_path,
                    image_path=image_path,
                    pose_model_path=pose_model_path,
                    input_layout=input_layout,
                    point_cloud_mode=args.point_cloud_mode,
                    undistort_output=undistort_output,
                    features_recomputed=input_layout == "converted",
                    shared_command_log=shared_command_log,
                    colmap_executable=args.colmap_executable,
                    gpu_index=args.gpu_index,
                    copy_policy=args.copy_policy,
                    keep_work=args.keep_work,
                    source_path=source_path,
                )

        print(
            f"All requested sparse-view {args.point_cloud_mode} models were "
            "prepared successfully."
        )
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Any output directory created before the failure is intentionally "
            "kept for inspection. Move or remove that exact directory before "
            "retrying.",
            file=sys.stderr,
        )
        return 1
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
