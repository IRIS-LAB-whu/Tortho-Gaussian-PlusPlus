#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import atexit
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from os import makedirs

import numpy as np
import torch
import torchvision
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from dummy_camera import DummyCamera, DummyPipeline
from gaussian_renderer import GaussianModel, render
from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos
from utils.general_utils import safe_state


_RENDER_OUTPUT = None


def terminal_summary(message):
    """Write a concise status only to the terminal, outside safe_state's wrapper."""
    if _RENDER_OUTPUT is not None and not _RENDER_OUTPUT["quiet"]:
        print(message, file=_RENDER_OUTPUT["stdout"], flush=True)


def render_progress_stream():
    return sys.stderr if _RENDER_OUTPUT is None else _RENDER_OUTPUT["progress"]


def enable_render_log(model_path, quiet=False):
    """Keep full diagnostics in the log; expose summaries, progress and errors."""
    global _RENDER_OUTPUT
    if _RENDER_OUTPUT is not None:
        _RENDER_OUTPUT["close"]()
    os.makedirs(model_path, exist_ok=True)
    log_file = open(
        os.path.join(model_path, "gravity_direction_estimation.log"), "a", encoding="utf-8"
    )
    original_stdout, original_stderr = sys.stdout, sys.stderr
    original_excepthook = sys.excepthook
    closed = False

    class TeeStream:
        def __init__(self, stream, echo=False):
            self.stream = stream
            self.echo = echo
            self.pending_carriage_return = False

        def write(self, text):
            result = self.stream.write(text) if self.echo else len(text)
            if text == "\r":
                self.pending_carriage_return = True
                return result
            camera_progress = re.fullmatch(r"Reading camera (\d+)/(\d+)", text)
            if camera_progress is not None:
                self.pending_carriage_return = False
                if camera_progress.group(1) != camera_progress.group(2):
                    return result
            if self.pending_carriage_return:
                log_file.write("\r")
                self.pending_carriage_return = False
            log_file.write(text)
            log_file.flush()
            return result

        def flush(self):
            self.stream.flush()
            log_file.flush()

        def __getattr__(self, name):
            return getattr(self.stream, name)

    progress_stream = TeeStream(original_stderr, echo=True)

    def report_exception(exc_type, exception, traceback):
        # Preserve the full traceback in the log without flooding the terminal.
        stderr = sys.stderr
        try:
            sys.stderr = TeeStream(original_stderr)
            original_excepthook(exc_type, exception, traceback)
        finally:
            sys.stderr = stderr
            print("untrust", file=original_stderr, flush=True)

    def close_log():
        global _RENDER_OUTPUT
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = original_stdout, original_stderr
            sys.excepthook = original_excepthook
            _RENDER_OUTPUT = None
            log_file.close()

    sys.stdout = TeeStream(original_stdout)
    sys.stderr = TeeStream(original_stderr)
    sys.excepthook = report_exception
    _RENDER_OUTPUT = {
        "stdout": original_stdout,
        "progress": progress_stream,
        "quiet": quiet,
        "close": close_log,
    }
    atexit.register(close_log)
    return close_log


WORLD_VIEW_SCALE = 0.13 * 1.2
ORTHO_MARGIN = 1.03
TARGET_LONG_EDGE = 8000
ORTHO_DEPTH_SAMPLE_COUNT = 1000000
ORTHO_NEAR_CULL_Z = 1e-6
ORTHO_ADAPTIVE_DEPTH_PERCENTILE = 0.0
ORTHO_ADAPTIVE_NEAR_MARGIN = 1.05
CAMERA_FORWARD_NORMAL_SAMPLE_COUNT = 10
CAMERA_FORWARD_NORMAL_SAMPLE_SEED = 0
CAMERA_CENTER_PLANE_RANSAC_ITERATIONS = 1024
CAMERA_CENTER_PLANE_RANSAC_SEED = 17
CAMERA_CENTER_PLANE_MIN_INLIER_RATIO = 0.6
CAMERA_CENTER_PLANE_SCENE_THRESHOLD_RATIO = 0.01
CAMERA_CENTER_PLANE_NN_THRESHOLD_MULTIPLIER = 1.5
CAMERA_CENTER_PLANE_MIN_THRESHOLD = 1e-6
CAMERA_CENTER_PLANE_CONFIDENCE_MAX_MEDIAN_THRESHOLD_RATIO = 0.5
CAMERA_CENTER_PLANE_CONFIDENCE_MAX_P90_THRESHOLD_RATIO = 1.0
CAMERA_CENTER_PLANE_CONFIDENCE_MIN_SPREAD_RATIO = 0.05
CAMERA_CENTER_PLANE_CONFIDENCE_MAX_THICKNESS_RATIO = 0.15
NADIR_MAX_ANGLE_DEGREES = 8.0
NADIR_CLUSTER_MIN_SEPARATION_DEGREES = 10.0
NADIR_CLUSTER_MAX_ITERATIONS = 20
GRAVITY_OUTLIER_MAD_MULTIPLIER = 3.5
GRAVITY_OUTLIER_MIN_THRESHOLD_DEGREES = 10.0
GRAVITY_OUTLIER_MIN_INLIER_RATIO = 0.5
GRAVITY_STRICT_MAD_MULTIPLIER = 2.0
GRAVITY_STRICT_MIN_THRESHOLD_DEGREES = 3.0
GRAVITY_STRICT_KEEP_QUANTILE = 85.0
GRAVITY_STRICT_MIN_KEEP_RATIO = 0.6
GRAVITY_CONFIDENCE_MIN_CAMERA_COUNT = 3
GRAVITY_CONFIDENCE_MIN_INLIER_COUNT = 3
GRAVITY_CONFIDENCE_MIN_INLIER_RATIO = 0.5
GRAVITY_CONFIDENCE_MAX_MEDIAN_ERROR_DEGREES = 10.0
GRAVITY_CONFIDENCE_MAX_P90_ERROR_DEGREES = 16.0
GRAVITY_CONFIDENCE_MIN_RESULTANT_LENGTH = 0.97
GRAVITY_VISUAL_INLIER_COLOR = "#2a9d8f"
GRAVITY_VISUAL_OUTLIER_COLOR = "#d62828"
GRAVITY_VISUAL_STRICT_REJECT_COLOR = "#7c3aed"
GRAVITY_VISUAL_AVERAGE_COLOR = "#f59e0b"
DEFAULT_PERSPECTIVE_FIELDS_PATH = os.environ.get(
    "PERSPECTIVE_FIELDS_PATH",
    os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "PerspectiveFields")
    ),
)
DEFAULT_PERSPECTIVE_FIELDS_VERSION = "Paramnet-360Cities-edina-uncentered"
PERSPECTIVE_FIELDS_RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "perspective_fields_predict.py",
)


def rotation_matrix_x(angle_radians):
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ],
        dtype=np.float64,
    )


def rotation_matrix_z(angle_radians):
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"Cannot normalize invalid vector: {vector}")
    return vector / norm


def gravity_direction_world(R, roll_degrees, pitch_degrees):
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)

    correction = rotation_matrix_z(-roll) @ rotation_matrix_x(
        -(pitch + math.pi / 2.0)
    )
    camera_forward_after_alignment = correction @ np.array([0.0, 0.0, 1.0])
    return normalize_vector(R @ camera_forward_after_alignment)


def project_axis_onto_plane(axis, normal):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    normal = normalize_vector(normal)
    projected = axis - np.dot(axis, normal) * normal
    norm = np.linalg.norm(projected)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return projected / norm


def camera_pose_from_world_gravity_direction(
    reference_R,
    selected_R,
    selected_T,
    gravity_direction,
):
    reference_R = np.asarray(reference_R, dtype=np.float64).reshape(3, 3)
    selected_R = np.asarray(selected_R, dtype=np.float64).reshape(3, 3)
    selected_T = np.asarray(selected_T, dtype=np.float64).reshape(3)

    camera_center = -selected_R @ selected_T
    z_axis = normalize_vector(gravity_direction)
    x_axis = project_axis_onto_plane(reference_R[:, 0], z_axis)
    if x_axis is None:
        x_axis = project_axis_onto_plane(reference_R[:, 1], z_axis)
    if x_axis is None:
        fallback_axis = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(z_axis)))]
        x_axis = project_axis_onto_plane(fallback_axis, z_axis)

    y_axis = normalize_vector(np.cross(z_axis, x_axis))
    gravity_aligned_R = np.stack([x_axis, y_axis, z_axis], axis=1)
    gravity_aligned_T = -gravity_aligned_R.T @ camera_center

    return (
        gravity_aligned_R.astype(np.float32),
        gravity_aligned_T.astype(np.float32),
    )


def gravity_aligned_camera_pose(R, T, roll_degrees, pitch_degrees):
    """
    Remove the predicted roll/pitch while retaining yaw, then point camera +z down.

    R is the transposed COLMAP world-to-camera rotation used by 3DGS, so it is
    also the camera-to-world rotation for column vectors.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T = np.asarray(T, dtype=np.float64).reshape(3)
    roll = math.radians(roll_degrees)
    pitch = math.radians(pitch_degrees)

    camera_center = -R @ T
    correction = rotation_matrix_z(-roll) @ rotation_matrix_x(
        -(pitch + math.pi / 2.0)
    )
    gravity_aligned_R = R @ correction
    gravity_aligned_T = -gravity_aligned_R.T @ camera_center

    return (
        gravity_aligned_R.astype(np.float32),
        gravity_aligned_T.astype(np.float32),
    )


def camera_prediction_key(name, camera_index, camera):
    return f"{name}:{camera_index}:{camera.image_name}"


def run_subprocess_with_live_output(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[PerspectiveFields] {line}", end="")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(
            f"PerspectiveFields subprocess exited with code {return_code}."
        )


def validate_perspective_fields_config(perspective_fields_path):
    perspective_fields_path = os.path.abspath(
        os.path.expanduser(perspective_fields_path)
    )
    if not os.path.isdir(perspective_fields_path):
        raise FileNotFoundError(
            "PerspectiveFields project directory does not exist: "
            f"{perspective_fields_path}"
        )
    if not os.path.isfile(PERSPECTIVE_FIELDS_RUNNER):
        raise FileNotFoundError(
            f"PerspectiveFields runner does not exist: {PERSPECTIVE_FIELDS_RUNNER}"
        )
    return perspective_fields_path


def validate_roll_pitch(roll_degrees, pitch_degrees, key):
    roll_degrees = float(roll_degrees)
    pitch_degrees = float(pitch_degrees)

    #下面两排别删
    #roll_degrees=0.0
    #pitch_degrees=-90.0

    if not np.isfinite([roll_degrees, pitch_degrees]).all():
        raise ValueError(
            "PerspectiveFields returned non-finite roll/pitch values: "
            f"key={key}, roll={roll_degrees}, pitch={pitch_degrees}"
        )
    return roll_degrees, pitch_degrees


def predict_roll_pitch_batch(
    camera_items,
    perspective_fields_path,
    version,
    device,
):
    if not camera_items:
        return {}

    perspective_fields_path = validate_perspective_fields_config(
        perspective_fields_path
    )
    keys = [str(key) for key, _ in camera_items]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate camera keys found while building PF cache.")

    with tempfile.TemporaryDirectory(prefix="perspective_fields_") as temp_dir:
        input_path = os.path.join(temp_dir, "images.json")
        result_path = os.path.join(temp_dir, "predictions.json")
        image_payload = []
        for key, camera in camera_items:
            image_path = getattr(camera, "image_path", None)
            if not image_path:
                raise ValueError(
                    f"Camera {getattr(camera, 'image_name', '<unknown>')} has no image_path for PerspectiveFields."
                )
            image_path = os.path.abspath(os.path.expanduser(str(image_path)))
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"PerspectiveFields input image does not exist: {image_path}")
            image_payload.append({"key": str(key), "path": image_path})

        with open(input_path, "w", encoding="utf-8") as input_file:
            json.dump({"images": image_payload}, input_file)

        command = [
            sys.executable,
            "-u",
            PERSPECTIVE_FIELDS_RUNNER,
            "--input-json",
            input_path,
            "--output",
            result_path,
            "--perspective-fields-path",
            perspective_fields_path,
            "--version",
            version,
            "--device",
            device,
            "--batch-size",
            "1",
        ]
        print(
            f"Running PerspectiveFields once for {len(camera_items)} image(s) "
            f"with the current Python environment: {sys.executable}"
        )
        run_subprocess_with_live_output(command)

        if not os.path.isfile(result_path):
            raise FileNotFoundError(
                "PerspectiveFields subprocess did not create its result file: "
                f"{result_path}"
            )
        with open(result_path, "r", encoding="utf-8") as result_file:
            prediction_payload = json.load(result_file)

    predictions = prediction_payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("PerspectiveFields batch output is missing 'predictions'.")

    roll_pitch_cache = {}
    for prediction in predictions:
        key = str(prediction["key"])
        roll_pitch_cache[key] = validate_roll_pitch(
            prediction["roll"],
            prediction["pitch"],
            key,
        )

    missing_keys = sorted(set(keys) - set(roll_pitch_cache.keys()))
    if missing_keys:
        raise ValueError(
            "PerspectiveFields batch output is missing predictions for keys: "
            f"{missing_keys}"
        )
    return roll_pitch_cache


def predict_roll_pitch(
    camera,
    perspective_fields_path,
    version,
    device,
):
    roll_pitch_cache = predict_roll_pitch_batch(
        [("__single__", camera)],
        perspective_fields_path,
        version,
        device,
    )
    roll_degrees, pitch_degrees = roll_pitch_cache["__single__"]

    return roll_degrees, pitch_degrees


def camera_orthogonal_bounding_rectangle(xyz, R, T, include_depth_sample=False):
    """
    Compute the Gaussian bounding rectangle in the selected camera plane.

    R/T follow getWorld2View2's row-vector convention:
        xyz_camera = xyz_world @ R + T
    """
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().float()
    else:
        xyz = torch.tensor(xyz, dtype=torch.float32)

    R = torch.as_tensor(R, dtype=xyz.dtype)
    T = torch.as_tensor(T, dtype=xyz.dtype)

    xyz_cam = xyz @ R + T.unsqueeze(0)
    filtered_xyz_cam = xyz_cam[torch.isfinite(xyz_cam).all(dim=1)]

    if filtered_xyz_cam.shape[0] == 0:
        raise ValueError("No finite Gaussians available for orthographic bounding box.")

    min_xy = filtered_xyz_cam[:, :2].min(dim=0).values
    max_xy = filtered_xyz_cam[:, :2].max(dim=0).values
    center_xy = (min_xy + max_xy) / 2.0

    rectangle = {
        "width": float((max_xy[0] - min_xy[0]).item()),
        "height": float((max_xy[1] - min_xy[1]).item()),
        "center_xy": center_xy,
        "min_xy": min_xy,
        "max_xy": max_xy,
        "num_bbox_gaussians": int(filtered_xyz_cam.shape[0]),
        "num_total_gaussians": int(xyz_cam.shape[0]),
    }
    if include_depth_sample:
        depth_values = filtered_xyz_cam[:, 2]
        if ORTHO_ADAPTIVE_DEPTH_PERCENTILE <= 0.0:
            finite_depths = depth_values[torch.isfinite(depth_values)]
            positive_depths = finite_depths[finite_depths > 1e-8]
            if positive_depths.numel() > 0:
                rectangle["depth_min_positive"] = positive_depths.min().clone()
        stride = max(
            1,
            int(math.ceil(depth_values.numel() / ORTHO_DEPTH_SAMPLE_COUNT)),
        )
        rectangle["depth_sample"] = depth_values[::stride].clone()

    return rectangle


def find_center_camera(views):
    if len(views) == 0:
        return None, None

    camera_centers = np.stack([
        view.camera_center.detach().cpu().numpy()
        for view in views
    ])
    center_position = np.mean(camera_centers, axis=0)
    distances = np.linalg.norm(camera_centers - center_position, axis=1)
    closest_camera_index = int(np.argmin(distances))
    return closest_camera_index, camera_centers[closest_camera_index]


def sample_camera_forward_views(views):
    if len(views) <= CAMERA_FORWARD_NORMAL_SAMPLE_COUNT:
        return list(views)
    rng = np.random.default_rng(CAMERA_FORWARD_NORMAL_SAMPLE_SEED)
    sample_indices = rng.choice(
        len(views),
        size=CAMERA_FORWARD_NORMAL_SAMPLE_COUNT,
        replace=False,
    )
    return [views[int(index)] for index in sample_indices]


def collect_camera_forward_directions(views):
    directions = []
    sampled_views = sample_camera_forward_views(views)
    for view in sampled_views:
        R = np.asarray(view.R, dtype=np.float64).reshape(3, 3)
        forward = R[:, 2]
        norm = np.linalg.norm(forward)
        if np.isfinite(norm) and norm > 1e-12:
            directions.append(forward / norm)

    if not directions:
        raise ValueError(
            "No finite camera forward directions are available to orient "
            "the camera-center plane normal."
        )
    return np.vstack(directions)


def orient_plane_normal_to_camera_forwards(normal, views):
    normal = normalize_vector(normal)
    forward_directions = collect_camera_forward_directions(views)
    signed_cosines = forward_directions @ normal
    median_cosine = float(np.median(signed_cosines))
    mean_cosine = float(np.mean(signed_cosines))

    flipped = median_cosine < 0.0 or (
        abs(median_cosine) <= 1e-12 and mean_cosine < 0.0
    )
    if flipped:
        normal = -normal

    return normal


def fit_camera_center_plane_svd(camera_centers):
    centroid = np.mean(camera_centers, axis=0)
    centered = camera_centers - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)

    if singular_values.shape[0] < 3:
        raise ValueError("Camera-center plane fitting failed.")
    rank_tolerance = max(1e-8, float(singular_values[0]) * 1e-6)
    if float(singular_values[1]) <= rank_tolerance:
        raise ValueError(
            "Camera centers are nearly collinear; cannot fit a stable plane normal."
        )

    return centroid, normalize_vector(vh[-1]), singular_values


def camera_center_plane_residuals(camera_centers, normal, point_on_plane=None):
    normal = normalize_vector(normal)
    if point_on_plane is None:
        offset = float(np.median(camera_centers @ normal))
        return np.abs((camera_centers @ normal) - offset)

    point_on_plane = np.asarray(point_on_plane, dtype=np.float64).reshape(3)
    return np.abs((camera_centers - point_on_plane) @ normal)


def camera_center_ransac_threshold(camera_centers):
    centered = camera_centers - np.mean(camera_centers, axis=0)
    scene_extent = float(np.max(np.linalg.norm(centered, axis=1)))
    scene_threshold = scene_extent * CAMERA_CENTER_PLANE_SCENE_THRESHOLD_RATIO
    threshold = scene_threshold

    if camera_centers.shape[0] > 1:
        pairwise_distances = np.linalg.norm(
            camera_centers[:, None, :] - camera_centers[None, :, :],
            axis=2,
        )
        pairwise_distances[pairwise_distances <= 1e-12] = np.inf
        nearest_distances = np.min(pairwise_distances, axis=1)
        finite_nearest = nearest_distances[np.isfinite(nearest_distances)]
        if finite_nearest.size > 0:
            nn_threshold = (
                float(np.median(finite_nearest))
                * CAMERA_CENTER_PLANE_NN_THRESHOLD_MULTIPLIER
            )
            threshold = min(scene_threshold, nn_threshold)

    return max(CAMERA_CENTER_PLANE_MIN_THRESHOLD, threshold)


def candidate_plane_normal_from_points(points):
    first_edge = points[1] - points[0]
    second_edge = points[2] - points[0]
    normal = np.cross(first_edge, second_edge)
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    return normal / norm


def robust_camera_center_plane_inlier_mask(camera_centers):
    camera_count = camera_centers.shape[0]
    threshold = camera_center_ransac_threshold(camera_centers)
    min_inlier_count = max(
        3,
        int(math.ceil(camera_count * CAMERA_CENTER_PLANE_MIN_INLIER_RATIO)),
    )

    best = None
    try:
        _, pca_normal, _ = fit_camera_center_plane_svd(camera_centers)
        residuals = camera_center_plane_residuals(camera_centers, pca_normal)
        inlier_mask = residuals <= threshold
        best = (
            int(np.count_nonzero(inlier_mask)),
            (
                -float(np.median(residuals[inlier_mask]))
                if np.any(inlier_mask)
                else -float("inf")
            ),
            -float(np.median(residuals)),
            inlier_mask,
            pca_normal,
            "all-camera PCA",
        )
    except ValueError:
        best = None

    rng = np.random.default_rng(CAMERA_CENTER_PLANE_RANSAC_SEED)
    for _ in range(CAMERA_CENTER_PLANE_RANSAC_ITERATIONS):
        sample_indices = rng.choice(camera_count, size=3, replace=False)
        normal = candidate_plane_normal_from_points(camera_centers[sample_indices])
        if normal is None:
            continue

        residuals = camera_center_plane_residuals(camera_centers, normal)
        inlier_mask = residuals <= threshold
        inlier_count = int(np.count_nonzero(inlier_mask))
        median_inlier_residual = (
            float(np.median(residuals[inlier_mask]))
            if inlier_count > 0
            else float("inf")
        )
        score = (
            inlier_count,
            -median_inlier_residual,
            -float(np.median(residuals)),
            inlier_mask,
            normal,
            "RANSAC",
        )
        if best is None or score[:3] > best[:3]:
            best = score

    if best is None:
        return None, threshold, 0, min_inlier_count, "none"

    inlier_count, _, _, inlier_mask, _, source = best
    if inlier_count < min_inlier_count:
        return None, threshold, inlier_count, min_inlier_count, source

    return inlier_mask, threshold, inlier_count, min_inlier_count, source


def evaluate_camera_center_plane_confidence(
    total_count,
    inlier_count,
    threshold,
    inlier_residuals,
    singular_values,
    used_fallback,
):
    singular_values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    inlier_residuals = np.asarray(inlier_residuals, dtype=np.float64).reshape(-1)
    inlier_ratio = 0.0 if total_count == 0 else inlier_count / total_count
    median_residual = (
        float(np.median(inlier_residuals))
        if inlier_residuals.size > 0
        else float("inf")
    )
    p90_residual = (
        float(np.percentile(inlier_residuals, 90.0))
        if inlier_residuals.size > 0
        else float("inf")
    )

    largest_singular = float(singular_values[0])
    middle_singular = float(singular_values[1])
    smallest_singular = float(singular_values[2])
    spread_ratio = (
        middle_singular / largest_singular
        if largest_singular > 1e-12
        else 0.0
    )
    thickness_ratio = (
        smallest_singular / middle_singular
        if middle_singular > 1e-12
        else float("inf")
    )

    rejection_reasons = []
    if used_fallback:
        rejection_reasons.append("robust inlier selection fell back to all centers")
    if inlier_ratio < CAMERA_CENTER_PLANE_MIN_INLIER_RATIO:
        rejection_reasons.append(
            f"inlier_ratio={inlier_ratio:.6f} < "
            f"{CAMERA_CENTER_PLANE_MIN_INLIER_RATIO:.6f}"
        )
    if median_residual > (
        threshold * CAMERA_CENTER_PLANE_CONFIDENCE_MAX_MEDIAN_THRESHOLD_RATIO
    ):
        rejection_reasons.append(
            f"median_residual={median_residual:.6f} > "
            f"{threshold * CAMERA_CENTER_PLANE_CONFIDENCE_MAX_MEDIAN_THRESHOLD_RATIO:.6f}"
        )
    if p90_residual > (
        threshold * CAMERA_CENTER_PLANE_CONFIDENCE_MAX_P90_THRESHOLD_RATIO
    ):
        rejection_reasons.append(
            f"p90_residual={p90_residual:.6f} > "
            f"{threshold * CAMERA_CENTER_PLANE_CONFIDENCE_MAX_P90_THRESHOLD_RATIO:.6f}"
        )
    if spread_ratio < CAMERA_CENTER_PLANE_CONFIDENCE_MIN_SPREAD_RATIO:
        rejection_reasons.append(
            f"spread_ratio={spread_ratio:.6f} < "
            f"{CAMERA_CENTER_PLANE_CONFIDENCE_MIN_SPREAD_RATIO:.6f}"
        )
    if thickness_ratio > CAMERA_CENTER_PLANE_CONFIDENCE_MAX_THICKNESS_RATIO:
        rejection_reasons.append(
            f"thickness_ratio={thickness_ratio:.6f} > "
            f"{CAMERA_CENTER_PLANE_CONFIDENCE_MAX_THICKNESS_RATIO:.6f}"
        )

    return {
        "label": "trust" if not rejection_reasons else "untrust",
        "inlier_ratio": float(inlier_ratio),
        "median_residual": median_residual,
        "p90_residual": p90_residual,
        "spread_ratio": float(spread_ratio),
        "thickness_ratio": float(thickness_ratio),
        "rejection_reasons": rejection_reasons,
    }


def fit_camera_forward_aligned_camera_center_plane_normal(views, return_confidence=False):
    if len(views) < 3:
        raise ValueError(
            "At least 3 cameras are required to fit a camera-center plane."
        )

    camera_centers = np.stack([
        view.camera_center.detach().cpu().numpy()
        for view in views
    ]).astype(np.float64)
    finite_mask = np.isfinite(camera_centers).all(axis=1)
    finite_camera_centers = camera_centers[finite_mask]
    if finite_camera_centers.shape[0] < 3:
        raise ValueError(
            "At least 3 finite camera centers are required to fit a plane."
        )

    (
        inlier_mask,
        ransac_threshold,
        ransac_inlier_count,
        min_inlier_count,
        ransac_source,
    ) = robust_camera_center_plane_inlier_mask(finite_camera_centers)
    fitting_camera_centers = finite_camera_centers
    used_fallback = inlier_mask is None
    confidence_inlier_count = ransac_inlier_count
    if inlier_mask is not None:
        fitting_camera_centers = finite_camera_centers[inlier_mask]
        confidence_inlier_count = fitting_camera_centers.shape[0]
    else:
        print(
            "[WARN] Robust camera-center plane fitting found too few inliers "
            f"({ransac_inlier_count}/{finite_camera_centers.shape[0]}, "
            f"need {min_inlier_count}); falling back to all finite camera centers."
        )

    centroid, normal, singular_values = fit_camera_center_plane_svd(fitting_camera_centers)
    residuals = camera_center_plane_residuals(finite_camera_centers, normal, centroid)
    inlier_residuals = camera_center_plane_residuals(
        fitting_camera_centers,
        normal,
        centroid,
    )
    confidence = evaluate_camera_center_plane_confidence(
        finite_camera_centers.shape[0],
        confidence_inlier_count,
        ransac_threshold,
        inlier_residuals,
        singular_values,
        used_fallback,
    )
    normal = orient_plane_normal_to_camera_forwards(normal, views)
    print(
        "Camera-center plane fitting: "
        f"source={ransac_source}, "
        f"inliers={fitting_camera_centers.shape[0]}/{finite_camera_centers.shape[0]}, "
        f"threshold={ransac_threshold:.6f}, "
        f"all_median_residual={float(np.median(residuals)):.6f}, "
        f"all_p90_residual={float(np.percentile(residuals, 90.0)):.6f}, "
        f"singular_values=({singular_values[0]:.6f}, "
        f"{singular_values[1]:.6f}, {singular_values[2]:.6f})"
    )
    print(
        "Camera-center plane confidence: "
        f"{confidence['label']} "
        f"(inlier_ratio={confidence['inlier_ratio']:.6f}, "
        f"inlier_median_residual={confidence['median_residual']:.6f}, "
        f"inlier_p90_residual={confidence['p90_residual']:.6f}, "
        f"spread_ratio={confidence['spread_ratio']:.6f}, "
        f"thickness_ratio={confidence['thickness_ratio']:.6f})"
    )
    for reason in confidence["rejection_reasons"]:
        print(f"[WARN] Camera-center plane confidence rejection reason: {reason}")
    print(
        "Camera-center plane normal: "
        f"({normal[0]:.10f}, {normal[1]:.10f}, {normal[2]:.10f})"
    )
    print("===== Method 2 camera-center plane normal end =====")
    print("")

    normal = normal.astype(np.float32)
    if return_confidence:
        return normal, confidence
    terminal_summary(
        "method2 (camera plane): "
        + ("trust" if camera_plane_confidence_is_trusted(confidence) else "untrust")
    )
    return normal


def camera_forward_direction(camera):
    R = np.asarray(camera.R, dtype=np.float64).reshape(3, 3)
    return normalize_vector(R[:, 2])


def split_angle_clusters(angles_degrees):
    angles_degrees = np.asarray(angles_degrees, dtype=np.float64).reshape(-1)
    if angles_degrees.shape[0] < 2:
        return None
    if float(np.max(angles_degrees) - np.min(angles_degrees)) < 1e-9:
        return None

    centers = np.array(
        [float(np.min(angles_degrees)), float(np.max(angles_degrees))],
        dtype=np.float64,
    )
    labels = None
    for _ in range(NADIR_CLUSTER_MAX_ITERATIONS):
        distances = np.abs(angles_degrees[:, None] - centers[None, :])
        next_labels = np.argmin(distances, axis=1)
        if np.count_nonzero(next_labels == 0) == 0:
            return None
        if np.count_nonzero(next_labels == 1) == 0:
            return None

        next_centers = np.array(
            [
                float(np.mean(angles_degrees[next_labels == 0])),
                float(np.mean(angles_degrees[next_labels == 1])),
            ],
            dtype=np.float64,
        )
        if labels is not None and np.array_equal(next_labels, labels):
            centers = next_centers
            break
        labels = next_labels
        centers = next_centers

    order = np.argsort(centers)
    remapped_labels = np.zeros_like(labels)
    for new_label, old_label in enumerate(order):
        remapped_labels[labels == old_label] = new_label
    centers = centers[order]

    if float(centers[1] - centers[0]) < NADIR_CLUSTER_MIN_SEPARATION_DEGREES:
        return None

    return remapped_labels, centers


def camera_item_description(record):
    return (
        f"{record['split_name']} camera index={record['camera_index']}, "
        f"image={record['image_name']}"
    )


def print_nadir_cluster_summary(records, angles_degrees, labels, centers, has_nadir):
    for cluster_label in sorted(set(int(label) for label in labels)):
        cluster_angles = angles_degrees[labels == cluster_label]
        cluster_records = [
            record for record, label in zip(records, labels)
            if int(label) == cluster_label
        ]
        if has_nadir and cluster_label == 0:
            cluster_name = "nadir"
        elif centers is not None and cluster_label == 0:
            cluster_name = "near-plane-oblique"
        else:
            cluster_name = "oblique"
        sample_record = cluster_records[0]
        center_text = ""
        if centers is not None:
            center_text = f", center={centers[cluster_label]:.6f} deg"
        print(
            f"[Nadir view {cluster_name}] count={len(cluster_records)}"
            f"{center_text}, mean_angle={float(np.mean(cluster_angles)):.6f} deg, "
            f"min_angle={float(np.min(cluster_angles)):.6f} deg, "
            f"max_angle={float(np.max(cluster_angles)):.6f} deg, "
            f"sample={camera_item_description(sample_record)}"
        )




def find_final_splat_nadir_candidates(camera_items, reference_direction, reference_name):
    if not camera_items:
        return []

    reference_direction = normalize_vector(reference_direction)
    records = []
    directions = []
    for split_name, camera_index, camera in camera_items:
        direction = camera_forward_direction(camera)
        angle_degrees = float(angular_distance_degrees([direction], reference_direction)[0])
        records.append(
            {
                "split_name": split_name,
                "camera_index": camera_index,
                "image_name": camera.image_name,
                "camera": camera,
                "angle_degrees": angle_degrees,
                "key": (split_name, int(camera_index), str(camera.image_name)),
            }
        )
        directions.append(direction)

    directions = np.vstack(directions)
    angles_degrees = np.asarray(
        [record["angle_degrees"] for record in records],
        dtype=np.float64,
    )

    print("")
    print(f"===== Final splat: {reference_name} =====")
    print(f"Final splat nadir search enabled for {reference_name}.")
    print(
        f"Final splat {reference_name} direction for nadir clustering: "
        f"({reference_direction[0]:.10f}, {reference_direction[1]:.10f}, "
        f"{reference_direction[2]:.10f})"
    )

    split_result = split_angle_clusters(angles_degrees)
    if split_result is None:
        labels = np.zeros(angles_degrees.shape[0], dtype=np.int64)
        centers = None
        median_angle = float(np.median(angles_degrees))
        has_nadir = median_angle <= NADIR_MAX_ANGLE_DEGREES
        cluster_mode = "pure_nadir" if has_nadir else "pure_oblique"
        print(
            f"Final splat nadir clustering produced one class for {reference_name}: "
            f"mode={cluster_mode}, median_angle={median_angle:.6f} deg, "
            f"nadir_threshold={NADIR_MAX_ANGLE_DEGREES:.6f} deg"
        )
    else:
        labels, centers = split_result
        has_nadir = float(centers[0]) <= NADIR_MAX_ANGLE_DEGREES
        cluster_mode = "nadir_oblique" if has_nadir else "oblique_only"
        print(
            f"Final splat nadir clustering produced two classes for {reference_name}: "
            f"mode={cluster_mode}, centers=({centers[0]:.6f}, "
            f"{centers[1]:.6f}) deg, "
            f"separation={float(centers[1] - centers[0]):.6f} deg, "
            f"nadir_threshold={NADIR_MAX_ANGLE_DEGREES:.6f} deg"
        )

    print_nadir_cluster_summary(records, angles_degrees, labels, centers, has_nadir)

    if not has_nadir:
        print(f"Final splat: no nadir camera class found for {reference_name}.")
        return []

    nadir_indices = np.flatnonzero(labels == 0)
    print(f"Final splat nadir image names for {reference_name}:")
    candidates = []
    for nadir_index in nadir_indices:
        record = records[int(nadir_index)]
        print(f"  {record['image_name']}")
        candidates.append(
            {
                "key": record["key"],
                "record": record,
                "direction": directions[int(nadir_index)].astype(np.float32),
            }
        )
    return candidates


def average_final_splat_directions(base_directions, nadir_candidates):
    directions = [normalize_vector(direction) for direction in base_directions]
    directions.extend(
        normalize_vector(candidate["direction"])
        for candidate in nadir_candidates
    )
    return normalized_mean_direction(np.vstack(directions)).astype(np.float32)


def camera_plane_confidence_is_trusted(confidence):
    return (
        confidence.get("label") == "trust"
        and len(confidence.get("rejection_reasons", [])) == 0
    )


def choose_final_splat_direction(
    camera_items,
    average_direction,
    average_confidence,
    plane_direction,
    plane_confidence,
):
    average_trusted = bool(average_confidence.get("is_trusted", False))
    plane_trusted = camera_plane_confidence_is_trusted(plane_confidence)
    print(
        "Final splat confidence summary: "
        f"method1_average_gravity trusted={average_trusted}, "
        f"method2_camera_plane_down_normal trusted={plane_trusted}"
    )
    terminal_summary("method1 (average gravity): " + ("trust" if average_trusted else "untrust"))
    terminal_summary("method2 (camera plane): " + ("trust" if plane_trusted else "untrust"))

    if average_trusted and not plane_trusted:
        nadir_candidates = find_final_splat_nadir_candidates(
            camera_items,
            average_direction,
            "method1 average gravity",
        )
        if nadir_candidates:
            final_direction = average_final_splat_directions(
                [average_direction],
                nadir_candidates,
            )
            source = (
                "final_splat method1 trusted; averaged method1 gravity with "
                f"{len(nadir_candidates)} nadir camera direction(s)"
            )
        else:
            final_direction = normalize_vector(average_direction).astype(np.float32)
            source = (
                "final_splat method1 trusted; no nadir cameras found, "
                "using method1 average gravity"
            )
    elif plane_trusted and not average_trusted:
        nadir_candidates = find_final_splat_nadir_candidates(
            camera_items,
            plane_direction,
            "method2 camera-plane normal",
        )
        if nadir_candidates:
            final_direction = average_final_splat_directions(
                [plane_direction],
                nadir_candidates,
            )
            source = (
                "final_splat method2 trusted; averaged method2 gravity with "
                f"{len(nadir_candidates)} nadir camera direction(s)"
            )
        else:
            final_direction = normalize_vector(plane_direction).astype(np.float32)
            source = (
                "final_splat method2 trusted; no nadir cameras found, "
                "using method2 camera-plane normal"
            )
    elif average_trusted and plane_trusted:
        average_nadir_candidates = find_final_splat_nadir_candidates(
            camera_items,
            average_direction,
            "method1 average gravity",
        )
        plane_nadir_candidates = find_final_splat_nadir_candidates(
            camera_items,
            plane_direction,
            "method2 camera-plane normal",
        )
        plane_candidates_by_key = {
            candidate["key"]: candidate
            for candidate in plane_nadir_candidates
        }
        intersected_nadir_candidates = [
            candidate
            for candidate in average_nadir_candidates
            if candidate["key"] in plane_candidates_by_key
        ]
        if intersected_nadir_candidates:
            print("Final splat intersected nadir image names:")
            for candidate in intersected_nadir_candidates:
                print(f"  {candidate['record']['image_name']}")
            final_direction = average_final_splat_directions(
                [average_direction, plane_direction],
                intersected_nadir_candidates,
            )
            source = (
                "final_splat both methods trusted; averaged method1/method2 "
                f"directions with {len(intersected_nadir_candidates)} "
                "intersected nadir camera direction(s)"
            )
        else:
            print("Final splat nadir intersection is empty.")
            average_nadir_count = len(average_nadir_candidates)
            plane_nadir_count = len(plane_nadir_candidates)
            if average_nadir_count != plane_nadir_count:
                if average_nadir_count > plane_nadir_count:
                    selected_method = "method1"
                    selected_direction = average_direction
                    selected_candidates = average_nadir_candidates
                else:
                    selected_method = "method2"
                    selected_direction = plane_direction
                    selected_candidates = plane_nadir_candidates
                final_direction = average_final_splat_directions(
                    [selected_direction],
                    selected_candidates,
                )
                source = (
                    "final_splat both methods trusted; no intersected nadir cameras, "
                    f"selected {selected_method} with more nadir cameras "
                    f"(method1={average_nadir_count}, method2={plane_nadir_count}); "
                    f"averaged only {selected_method} direction with its "
                    f"{len(selected_candidates)} nadir camera direction(s)"
                )
            else:
                final_direction = average_final_splat_directions(
                    [average_direction, plane_direction],
                    [],
                )
                source = (
                    "final_splat both methods trusted; no intersected nadir cameras, "
                    "averaged method1 and method2 directions"
                )
    else:
        raise ValueError(
            "Final splat direction is untrusted: method1 and method2 are both untrusted."
        )

    final_direction = normalize_vector(final_direction).astype(np.float32)
    print("")
    print("===== Final splat: final output =====")
    print(
        "Final splat render direction: "
        f"({final_direction[0]:.10f}, {final_direction[1]:.10f}, "
        f"{final_direction[2]:.10f})"
    )
    print(f"Final splat render direction source: {source}")
    return final_direction, source


def untrusted_final_splat_confidence(reason):
    return {
        "label": "untrusted",
        "is_trusted": False,
        "rejection_reasons": [reason],
    }


def resolve_final_splat_direction(
    model_path,
    iteration,
    camera_items,
    perspective_fields_path,
    perspective_fields_version,
    perspective_fields_device,
):
    if not camera_items:
        print("[WARN] Final splat: no cameras available.")
        return None, "final_splat fallback: no cameras available"

    views = [camera for _, _, camera in camera_items]

    center_camera_index, _ = find_center_camera(views)
    prediction_items = [
        (camera_prediction_key("final_splat", camera_index, camera), camera)
        for camera_index, camera in enumerate(views)
    ]
    roll_pitch_cache = predict_roll_pitch_batch(
        prediction_items,
        perspective_fields_path,
        perspective_fields_version,
        perspective_fields_device,
    )
    visualization_path = None
    (
        average_direction,
        _,
        _,
        average_confidence,
    ) = predict_average_gravity_direction(
        "final_splat",
        views,
        center_camera_index,
        perspective_fields_path,
        perspective_fields_version,
        perspective_fields_device,
        visualization_path,
        roll_pitch_cache,
        return_confidence=True,
        save_visualization=False,
    )

    plane_direction = None
    plane_confidence = untrusted_final_splat_confidence(
        "method2 camera-plane normal was not computed"
    )
    if len(views) < 3:
        reason = (
            f"only {len(views)} cameras; need at least 3 for method2 "
            "camera-plane normal"
        )
        print(f"[WARN] Final splat method2 is untrusted: {reason}")
        plane_confidence = untrusted_final_splat_confidence(reason)
    else:
        try:
            plane_direction, plane_confidence = (
                fit_camera_forward_aligned_camera_center_plane_normal(
                    views,
                    return_confidence=True,
                )
            )
        except ValueError as error:
            reason = str(error)
            print(f"[WARN] Final splat method2 failed: {reason}")
            plane_confidence = untrusted_final_splat_confidence(reason)

    return choose_final_splat_direction(
        camera_items,
        average_direction,
        average_confidence,
        plane_direction,
        plane_confidence,
    )


def collect_required_prediction_items(
    name,
    views,
    average_gravity_direction,
):
    if len(views) == 0:
        return []

    if average_gravity_direction:
        camera_indices = range(len(views))
    else:
        closest_camera_index, _ = find_center_camera(views)
        if closest_camera_index is None:
            return []
        camera_indices = [closest_camera_index]

    return [
        (camera_prediction_key(name, camera_index, views[camera_index]), views[camera_index])
        for camera_index in camera_indices
    ]


def normalized_mean_direction(directions):
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    mean_direction = np.mean(directions, axis=0)
    norm = np.linalg.norm(mean_direction)
    if not np.isfinite(norm) or norm <= 1e-12:
        return directions[0].copy()
    return mean_direction / norm


def angular_distance_degrees(directions, reference_direction):
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    reference_direction = normalize_vector(reference_direction)
    dots = np.clip(directions @ reference_direction, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def robust_average_gravity_directions(directions):
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if directions.shape[0] == 0:
        raise ValueError("No gravity directions are available for averaging.")
    directions = np.vstack([normalize_vector(direction) for direction in directions])

    if directions.shape[0] <= 2:
        inlier_mask = np.ones(directions.shape[0], dtype=bool)
        average_direction = normalized_mean_direction(directions)
        angular_errors = angular_distance_degrees(directions, average_direction)
        return average_direction, inlier_mask, {
            "angular_errors_degrees": angular_errors,
            "final_angular_errors_degrees": angular_errors,
            "coarse_inlier_mask": inlier_mask.copy(),
            "threshold_degrees": 180.0,
            "strict_threshold_degrees": 180.0,
            "median_error_degrees": float(np.median(angular_errors)),
            "mad_degrees": 0.0,
            "strict_median_error_degrees": float(np.median(angular_errors)),
            "strict_mad_degrees": 0.0,
            "medoid_index": 0,
            "fallback_used": False,
            "strict_fallback_used": False,
        }

    dot_matrix = np.clip(directions @ directions.T, -1.0, 1.0)
    pairwise_angles = np.degrees(np.arccos(dot_matrix))
    median_pairwise_angles = np.median(pairwise_angles, axis=1)
    medoid_index = int(np.argmin(median_pairwise_angles))
    angular_errors = pairwise_angles[medoid_index]
    median_error = float(np.median(angular_errors))
    mad = float(np.median(np.abs(angular_errors - median_error)))
    robust_sigma = 1.4826 * mad
    threshold_degrees = max(
        GRAVITY_OUTLIER_MIN_THRESHOLD_DEGREES,
        median_error + GRAVITY_OUTLIER_MAD_MULTIPLIER * robust_sigma,
    )

    coarse_inlier_mask = angular_errors <= threshold_degrees
    min_inlier_count = max(
        1,
        int(math.ceil(directions.shape[0] * GRAVITY_OUTLIER_MIN_INLIER_RATIO)),
    )
    fallback_used = False
    if int(np.count_nonzero(coarse_inlier_mask)) < min_inlier_count:
        fallback_used = True
        sorted_indices = np.argsort(angular_errors)
        coarse_inlier_mask = np.zeros(directions.shape[0], dtype=bool)
        coarse_inlier_mask[sorted_indices[:min_inlier_count]] = True
        threshold_degrees = float(angular_errors[sorted_indices[min_inlier_count - 1]])

    provisional_average = normalized_mean_direction(directions[coarse_inlier_mask])
    final_angular_errors = angular_distance_degrees(directions, provisional_average)
    coarse_errors_to_average = final_angular_errors[coarse_inlier_mask]
    strict_fallback_used = False

    if coarse_errors_to_average.shape[0] <= 2:
        inlier_mask = coarse_inlier_mask.copy()
        strict_threshold_degrees = 180.0
        strict_median_error = float(np.median(coarse_errors_to_average))
        strict_mad = 0.0
    else:
        strict_median_error = float(np.median(coarse_errors_to_average))
        strict_mad = float(
            np.median(np.abs(coarse_errors_to_average - strict_median_error))
        )
        strict_sigma = 1.4826 * strict_mad
        strict_mad_threshold = (
            strict_median_error + GRAVITY_STRICT_MAD_MULTIPLIER * strict_sigma
        )
        strict_quantile_threshold = float(
            np.percentile(coarse_errors_to_average, GRAVITY_STRICT_KEEP_QUANTILE)
        )
        strict_threshold_degrees = min(
            threshold_degrees,
            max(
                GRAVITY_STRICT_MIN_THRESHOLD_DEGREES,
                min(strict_mad_threshold, strict_quantile_threshold),
            ),
        )
        inlier_mask = coarse_inlier_mask & (final_angular_errors <= strict_threshold_degrees)
        coarse_inlier_count = int(np.count_nonzero(coarse_inlier_mask))
        min_strict_count = max(
            1,
            int(math.ceil(coarse_inlier_count * GRAVITY_STRICT_MIN_KEEP_RATIO)),
        )
        if int(np.count_nonzero(inlier_mask)) < min_strict_count:
            strict_fallback_used = True
            coarse_indices = np.flatnonzero(coarse_inlier_mask)
            sorted_coarse_indices = coarse_indices[
                np.argsort(final_angular_errors[coarse_indices])
            ]
            inlier_mask = np.zeros(directions.shape[0], dtype=bool)
            inlier_mask[sorted_coarse_indices[:min_strict_count]] = True
            strict_threshold_degrees = float(
                final_angular_errors[sorted_coarse_indices[min_strict_count - 1]]
            )

    average_direction = normalized_mean_direction(directions[inlier_mask])
    return average_direction, inlier_mask, {
        "angular_errors_degrees": angular_errors,
        "final_angular_errors_degrees": final_angular_errors,
        "coarse_inlier_mask": coarse_inlier_mask,
        "threshold_degrees": float(threshold_degrees),
        "strict_threshold_degrees": float(strict_threshold_degrees),
        "median_error_degrees": median_error,
        "mad_degrees": mad,
        "strict_median_error_degrees": strict_median_error,
        "strict_mad_degrees": strict_mad,
        "medoid_index": medoid_index,
        "fallback_used": fallback_used,
        "strict_fallback_used": strict_fallback_used,
    }


def evaluate_average_gravity_confidence(
    directions,
    inlier_mask,
    average_direction,
    stats,
):
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    directions = np.vstack([normalize_vector(direction) for direction in directions])
    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    average_direction = normalize_vector(average_direction)

    total_count = int(directions.shape[0])
    inlier_count = int(np.count_nonzero(inlier_mask))
    min_required_inliers = max(
        GRAVITY_CONFIDENCE_MIN_INLIER_COUNT,
        int(math.ceil(total_count * GRAVITY_CONFIDENCE_MIN_INLIER_RATIO)),
    )

    if inlier_count > 0:
        inlier_directions = directions[inlier_mask]
        inlier_errors = angular_distance_degrees(
            inlier_directions,
            average_direction,
        )
        inlier_median_error = float(np.median(inlier_errors))
        inlier_p90_error = float(np.percentile(inlier_errors, 90.0))
        inlier_max_error = float(np.max(inlier_errors))
        resultant_length = float(np.linalg.norm(np.mean(inlier_directions, axis=0)))
    else:
        inlier_median_error = float("inf")
        inlier_p90_error = float("inf")
        inlier_max_error = float("inf")
        resultant_length = 0.0

    inlier_ratio = 0.0 if total_count == 0 else inlier_count / total_count
    rejection_reasons = []
    caution_reasons = []
    if total_count < GRAVITY_CONFIDENCE_MIN_CAMERA_COUNT:
        rejection_reasons.append(
            f"only {total_count} cameras; need at least "
            f"{GRAVITY_CONFIDENCE_MIN_CAMERA_COUNT}"
        )
    if inlier_count < min_required_inliers:
        rejection_reasons.append(
            f"only {inlier_count} trusted directions; need at least "
            f"{min_required_inliers}"
        )
    if inlier_ratio < GRAVITY_CONFIDENCE_MIN_INLIER_RATIO:
        rejection_reasons.append(
            f"inlier ratio {inlier_ratio:.3f} is below "
            f"{GRAVITY_CONFIDENCE_MIN_INLIER_RATIO:.3f}"
        )
    if inlier_median_error > GRAVITY_CONFIDENCE_MAX_MEDIAN_ERROR_DEGREES:
        rejection_reasons.append(
            f"inlier median error {inlier_median_error:.3f} deg exceeds "
            f"{GRAVITY_CONFIDENCE_MAX_MEDIAN_ERROR_DEGREES:.3f} deg"
        )
    if inlier_p90_error > GRAVITY_CONFIDENCE_MAX_P90_ERROR_DEGREES:
        rejection_reasons.append(
            f"inlier p90 error {inlier_p90_error:.3f} deg exceeds "
            f"{GRAVITY_CONFIDENCE_MAX_P90_ERROR_DEGREES:.3f} deg"
        )
    if resultant_length < GRAVITY_CONFIDENCE_MIN_RESULTANT_LENGTH:
        rejection_reasons.append(
            f"resultant length {resultant_length:.6f} is below "
            f"{GRAVITY_CONFIDENCE_MIN_RESULTANT_LENGTH:.6f}"
        )
    if stats.get("fallback_used", False):
        rejection_reasons.append("coarse outlier fallback was needed")
    if stats.get("strict_fallback_used", False):
        caution_reasons.append("strict inlier fallback was needed")

    is_trusted = len(rejection_reasons) == 0

    return {
        "is_trusted": is_trusted,
        "label": "trusted" if is_trusted else "untrusted",
        "total_count": total_count,
        "inlier_count": inlier_count,
        "min_required_inliers": min_required_inliers,
        "inlier_ratio": float(inlier_ratio),
        "inlier_median_error_degrees": inlier_median_error,
        "inlier_p90_error_degrees": inlier_p90_error,
        "inlier_max_error_degrees": inlier_max_error,
        "inlier_resultant_length": resultant_length,
        "rejection_reasons": rejection_reasons,
        "caution_reasons": caution_reasons,
    }


def project_direction_for_visualization(direction, origin, scale):
    x, y, z = normalize_vector(direction)
    azimuth = math.radians(40.0)
    elevation = math.radians(25.0)

    projected_x = math.cos(azimuth) * x - math.sin(azimuth) * y
    rotated_y = math.sin(azimuth) * x + math.cos(azimuth) * y
    projected_y = -(
        math.sin(elevation) * rotated_y + math.cos(elevation) * z
    )

    return origin[0] + scale * projected_x, origin[1] + scale * projected_y


def svg_arrow(start, end, color, stroke_width, arrow_length, arrow_width, opacity):
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return ""

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux

    base_x = ex - ux * arrow_length
    base_y = ey - uy * arrow_length
    line_end_x = ex - ux * (arrow_length * 0.7)
    line_end_y = ey - uy * (arrow_length * 0.7)
    p1 = (ex, ey)
    p2 = (base_x + px * arrow_width, base_y + py * arrow_width)
    p3 = (base_x - px * arrow_width, base_y - py * arrow_width)

    return (
        f'<line x1="{sx:.3f}" y1="{sy:.3f}" x2="{line_end_x:.3f}" '
        f'y2="{line_end_y:.3f}" stroke="{color}" '
        f'stroke-width="{stroke_width:.3f}" stroke-linecap="round" '
        f'opacity="{opacity:.3f}" />\n'
        f'<polygon points="{p1[0]:.3f},{p1[1]:.3f} '
        f'{p2[0]:.3f},{p2[1]:.3f} {p3[0]:.3f},{p3[1]:.3f}" '
        f'fill="{color}" opacity="{opacity:.3f}" />\n'
    )


def write_gravity_rejection_visualization(
    output_path,
    directions,
    inlier_mask,
    average_direction,
    stats,
):
    width = 980
    height = 760
    origin = (width / 2.0, height / 2.0 + 45.0)
    scale = 280.0
    axis_scale = 95.0
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    inlier_mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    coarse_inlier_mask = np.asarray(
        stats.get("coarse_inlier_mask", inlier_mask),
        dtype=bool,
    ).reshape(-1)
    confidence = stats.get("confidence")
    coarse_rejected_count = int(np.count_nonzero(~coarse_inlier_mask))
    strict_rejected_count = int(np.count_nonzero(coarse_inlier_mask & ~inlier_mask))
    inlier_count = int(np.count_nonzero(inlier_mask))

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n',
        '<rect width="100%" height="100%" fill="#ffffff" />\n',
        '<text x="28" y="40" font-family="Arial, sans-serif" '
        'font-size="22" fill="#222222">Robust world-space gravity directions</text>\n',
        '<text x="28" y="66" font-family="Arial, sans-serif" '
        'font-size="13" fill="#666666">'
        f'kept={inlier_count}, strict rejected={strict_rejected_count}, '
        f'coarse rejected={coarse_rejected_count}, '
        f'coarse threshold={stats["threshold_degrees"]:.3f} deg, '
        f'strict threshold={stats["strict_threshold_degrees"]:.3f} deg</text>\n',
    ]
    if confidence is not None:
        svg_parts.append(
            '<text x="28" y="88" font-family="Arial, sans-serif" '
            'font-size="13" fill="#666666">'
            f'confidence={confidence["label"]}, '
            f'inlier ratio={confidence["inlier_ratio"]:.3f}, '
            f'inlier median={confidence["inlier_median_error_degrees"]:.3f} deg, '
            f'inlier p90={confidence["inlier_p90_error_degrees"]:.3f} deg, '
            f'resultant={confidence["inlier_resultant_length"]:.6f}</text>\n'
        )
    svg_parts.append(
        f'<circle cx="{origin[0]:.3f}" cy="{origin[1]:.3f}" '
        'r="4" fill="#111111" />\n'
    )

    legend_items = [
        (GRAVITY_VISUAL_INLIER_COLOR, "kept directions"),
        (GRAVITY_VISUAL_STRICT_REJECT_COLOR, "strict rejected directions"),
        (GRAVITY_VISUAL_OUTLIER_COLOR, "coarse outlier directions"),
        (GRAVITY_VISUAL_AVERAGE_COLOR, "average direction"),
    ]
    legend_x = 28.0
    legend_y = 122.0 if confidence is not None else 96.0
    for item_index, (color, label) in enumerate(legend_items):
        y = legend_y + item_index * 24.0
        svg_parts.append(
            f'<line x1="{legend_x:.1f}" y1="{y:.1f}" '
            f'x2="{legend_x + 32.0:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round" />\n'
        )
        svg_parts.append(
            f'<text x="{legend_x + 42.0:.1f}" y="{y + 5.0:.1f}" '
            'font-family="Arial, sans-serif" font-size="13" '
            f'fill="#444444">{label}</text>\n'
        )

    axes = [
        (np.array([1.0, 0.0, 0.0]), "X"),
        (np.array([0.0, 1.0, 0.0]), "Y"),
        (np.array([0.0, 0.0, 1.0]), "Z"),
    ]
    for axis_vector, label in axes:
        end = project_direction_for_visualization(axis_vector, origin, axis_scale)
        svg_parts.append(svg_arrow(origin, end, "#a0a0a0", 1.5, 12.0, 5.5, 0.65))
        svg_parts.append(
            f'<text x="{end[0] + 6:.3f}" y="{end[1] + 5:.3f}" '
            'font-family="Arial, sans-serif" font-size="13" '
            f'fill="#777777">{label}</text>\n'
        )

    for direction, is_coarse_inlier, is_inlier in zip(
        directions,
        coarse_inlier_mask,
        inlier_mask,
    ):
        end = project_direction_for_visualization(direction, origin, scale)
        if is_inlier:
            color = GRAVITY_VISUAL_INLIER_COLOR
            stroke_width = 2.2
            opacity = 0.74
        elif is_coarse_inlier:
            color = GRAVITY_VISUAL_STRICT_REJECT_COLOR
            stroke_width = 2.1
            opacity = 0.82
        else:
            color = GRAVITY_VISUAL_OUTLIER_COLOR
            stroke_width = 2.0
            opacity = 0.86
        svg_parts.append(svg_arrow(origin, end, color, stroke_width, 13.0, 6.0, opacity))

    average_end = project_direction_for_visualization(
        average_direction,
        origin,
        scale * 1.08,
    )
    svg_parts.append(
        svg_arrow(
            origin,
            average_end,
            GRAVITY_VISUAL_AVERAGE_COLOR,
            5.0,
            18.0,
            9.0,
            1.0,
        )
    )

    svg_parts.append("</svg>\n")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as svg_file:
        svg_file.writelines(svg_parts)


def predict_average_gravity_direction(
    name,
    views,
    selected_camera_index,
    perspective_fields_path,
    perspective_fields_version,
    perspective_fields_device,
    visualization_path,
    roll_pitch_cache,
    return_confidence=False,
    save_visualization=True,
):
    camera_indices = list(range(len(views)))
    print(
        f"Average gravity direction enabled for {name}: "
        f"predicting all {len(camera_indices)} cameras"
    )

    directions = []
    selected_reference_R = None
    selected_roll_pitch = None
    prediction_records = []

    for average_index, camera_index in enumerate(camera_indices, start=1):
        camera = views[camera_index]
        camera_R = np.asarray(camera.R, dtype=np.float32).reshape(3, 3)
        camera_T = np.asarray(camera.T, dtype=np.float32).reshape(3)
        prediction_key = camera_prediction_key(name, camera_index, camera)
        roll_degrees, pitch_degrees = roll_pitch_cache[prediction_key]
        direction = gravity_direction_world(camera_R, roll_degrees, pitch_degrees)
        directions.append(direction)
        prediction_records.append(
            {
                "camera_index": camera_index,
                "image_name": camera.image_name,
                "roll_degrees": roll_degrees,
                "pitch_degrees": pitch_degrees,
            }
        )

        if camera_index == selected_camera_index:
            selected_reference_R, _ = gravity_aligned_camera_pose(
                camera_R,
                camera_T,
                roll_degrees,
                pitch_degrees,
            )
            selected_roll_pitch = (roll_degrees, pitch_degrees)

        print(
            f"[Average gravity {average_index}/{len(camera_indices)}] "
            f"camera index={camera_index}, image={camera.image_name}, "
            f"roll={roll_degrees:.6f} deg, pitch={pitch_degrees:.6f} deg, "
            f"gravity=({direction[0]:.10f}, {direction[1]:.10f}, "
            f"{direction[2]:.10f})"
        )

    directions = np.vstack(directions)
    average_direction, inlier_mask, stats = robust_average_gravity_directions(directions)
    confidence = evaluate_average_gravity_confidence(
        directions,
        inlier_mask,
        average_direction,
        stats,
    )
    stats["confidence"] = confidence
    angular_errors = stats["angular_errors_degrees"]
    final_angular_errors = stats["final_angular_errors_degrees"]
    coarse_inlier_mask = stats["coarse_inlier_mask"]
    inlier_count = int(np.count_nonzero(inlier_mask))
    coarse_rejected_indices = [
        record["camera_index"]
        for record, is_coarse_inlier in zip(prediction_records, coarse_inlier_mask)
        if not is_coarse_inlier
    ]
    strict_rejected_indices = [
        record["camera_index"]
        for record, is_coarse_inlier, is_inlier in zip(
            prediction_records,
            coarse_inlier_mask,
            inlier_mask,
        )
        if is_coarse_inlier and not is_inlier
    ]
    print(
        f"Robust gravity rejection: kept={inlier_count}/{len(directions)}, "
        f"strict_rejected={len(strict_rejected_indices)}, "
        f"coarse_rejected={len(coarse_rejected_indices)}, "
        f"coarse_threshold={stats['threshold_degrees']:.6f} deg, "
        f"strict_threshold={stats['strict_threshold_degrees']:.6f} deg, "
        f"medoid_camera_index={prediction_records[stats['medoid_index']]['camera_index']}"
    )
    if stats["fallback_used"]:
        print(
            "[WARN] Gravity outlier threshold kept too few directions; "
            "used closest-directions fallback."
        )
    if stats["strict_fallback_used"]:
        print(
            "[WARN] Strict gravity threshold kept too few directions; "
            "used closest-inlier fallback."
        )
    print(f"Coarse rejected gravity camera indices: {coarse_rejected_indices}")
    print(f"Strict rejected gravity camera indices: {strict_rejected_indices}")
    for record, is_coarse_inlier, is_inlier, angular_error, final_angular_error in zip(
        prediction_records,
        coarse_inlier_mask,
        inlier_mask,
        angular_errors,
        final_angular_errors,
    ):
        if is_inlier:
            status = "INLIER"
        elif is_coarse_inlier:
            status = "STRICT_REJECTED"
        else:
            status = "COARSE_REJECTED"
        print(
            f"[{status}] camera index={record['camera_index']}, "
            f"image={record['image_name']}, "
            f"angle_to_medoid={angular_error:.6f} deg, "
            f"angle_to_provisional_average={final_angular_error:.6f} deg"
        )

    print(
        f"Robust average world gravity direction: "
        f"({average_direction[0]:.10f}, {average_direction[1]:.10f}, "
        f"{average_direction[2]:.10f})"
    )
    print(
        f"Average gravity confidence: trusted={confidence['is_trusted']}, "
        f"inliers={confidence['inlier_count']}/{confidence['total_count']}, "
        f"inlier_ratio={confidence['inlier_ratio']:.6f}, "
        f"inlier_median_error={confidence['inlier_median_error_degrees']:.6f} deg, "
        f"inlier_p90_error={confidence['inlier_p90_error_degrees']:.6f} deg, "
        f"inlier_resultant={confidence['inlier_resultant_length']:.6f}"
    )
    for reason in confidence["caution_reasons"]:
        print(f"[INFO] Average gravity confidence caution: {reason}")
    for reason in confidence["rejection_reasons"]:
        print(f"[WARN] Average gravity confidence rejection reason: {reason}")
    if confidence["is_trusted"]:
        print("Average gravity direction is trusted for diagnostics.")
    else:
        print(
            "[WARN] Average gravity direction is untrusted for diagnostics, "
            "but it will still be used for final render."
        )
    if save_visualization:
        write_gravity_rejection_visualization(
            visualization_path,
            directions,
            inlier_mask,
            average_direction,
            stats,
        )
        print(f"Saved gravity rejection visualization: {visualization_path}")
    print("===== Method 1 average gravity end =====")
    print("")

    if return_confidence:
        return average_direction, selected_reference_R, selected_roll_pitch, confidence
    terminal_summary(
        "method1 (average gravity): "
        + ("trust" if confidence["is_trusted"] else "untrust")
    )
    return average_direction, selected_reference_R, selected_roll_pitch


def select_ortho_view_scale(rectangle, base_scale, adaptive_view_scale):
    if not adaptive_view_scale:
        return float(base_scale)

    depth_sample = rectangle.get("depth_sample")
    if depth_sample is None or depth_sample.numel() == 0:
        print(
            "[WARN] Adaptive ortho view scale requested, but no depth sample "
            "is available; using default scale."
        )
        return float(base_scale)

    depth_sample = depth_sample.detach().cpu().float()
    finite_depths = depth_sample[torch.isfinite(depth_sample)]
    if finite_depths.numel() == 0:
        print(
            "[WARN] Adaptive ortho view scale requested, but all sampled depths "
            "are invalid; using default scale."
        )
        return float(base_scale)

    positive_depths = finite_depths[finite_depths > 1e-8]
    if positive_depths.numel() == 0:
        print(
            "[WARN] Adaptive ortho view scale requested, but no sampled depth "
            "is positive; using default scale."
        )
        return float(base_scale)

    if ORTHO_ADAPTIVE_DEPTH_PERCENTILE <= 0.0 and "depth_min_positive" in rectangle:
        robust_depth = float(rectangle["depth_min_positive"].item())
    else:
        quantile = ORTHO_ADAPTIVE_DEPTH_PERCENTILE / 100.0
        robust_depth = float(torch.quantile(positive_depths, quantile).item())
    if not np.isfinite(robust_depth) or robust_depth <= 1e-8:
        print(
            "[WARN] Adaptive ortho view scale found an invalid robust depth; "
            "using default scale."
        )
        return float(base_scale)

    required_scale = (
        ORTHO_NEAR_CULL_Z * ORTHO_ADAPTIVE_NEAR_MARGIN / robust_depth
    )
    selected_scale = max(float(base_scale), float(required_scale))
    default_keep_ratio = float(
        (finite_depths * float(base_scale) > ORTHO_NEAR_CULL_Z)
        .float()
        .mean()
        .item()
    )
    selected_keep_ratio = float(
        (finite_depths * selected_scale > ORTHO_NEAR_CULL_Z)
        .float()
        .mean()
        .item()
    )

    print(
        "Adaptive ortho view scale: "
        f"default={float(base_scale):.10f}, selected={selected_scale:.10f}, "
        f"depth_p{ORTHO_ADAPTIVE_DEPTH_PERCENTILE:.1f}={robust_depth:.10f}, "
        f"default_estimated_keep={default_keep_ratio * 100.0:.3f}%, "
        f"selected_estimated_keep={selected_keep_ratio * 100.0:.3f}%"
    )
    return selected_scale


def build_camera_centered_ortho_bounds(rectangle, world_view_scale=WORLD_VIEW_SCALE):
    min_xy = rectangle["min_xy"].detach().cpu().numpy()
    max_xy = rectangle["max_xy"].detach().cpu().numpy()
    extent = np.maximum(max_xy - min_xy, 1e-6)
    padding = extent * (ORTHO_MARGIN - 1.0) * 0.5

    left = float((min_xy[0] - padding[0]) * world_view_scale)
    right = float((max_xy[0] + padding[0]) * world_view_scale)
    bottom = float((min_xy[1] - padding[1]) * world_view_scale)
    top = float((max_xy[1] + padding[1]) * world_view_scale)

    return left, right, bottom, top


def image_size_from_rectangle(rectangle):
    if rectangle["width"] <= 1e-8 or rectangle["height"] <= 1e-8:
        raise ValueError(
            f"Degenerate orthographic rectangle: "
            f"width={rectangle['width']}, height={rectangle['height']}"
        )

    aspect_ratio = rectangle["width"] / rectangle["height"]
    if aspect_ratio >= 1.0:
        width = TARGET_LONG_EDGE
        height = max(1, int(round(TARGET_LONG_EDGE / aspect_ratio)))
    else:
        height = TARGET_LONG_EDGE
        width = max(1, int(round(TARGET_LONG_EDGE * aspect_ratio)))

    return width, height


def generate_unique_filename(base_path, prefix):
    idx = 0
    while True:
        filename = f"{prefix}_{idx:05d}.png"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return filepath
        idx += 1


def generate_unique_filepath(base_path, prefix, extension):
    idx = 0
    while True:
        filename = f"{prefix}_{idx:05d}{extension}"
        filepath = os.path.join(base_path, filename)
        if not os.path.exists(filepath):
            return filepath
        idx += 1




def resolve_orthographic_camera_pose(
    name,
    selected_camera,
    selected_camera_index,
    selected_camera_center,
    perspective_fields_path,
    perspective_fields_version,
    perspective_fields_device,
    closest_camera=False,
    average_gravity_direction=None,
    average_reference_R=None,
    selected_roll_pitch=None,
    roll_pitch_cache=None,
    plane_normal_direction=None,
    render_direction_source=None,
):
    selected_R = np.asarray(selected_camera.R, dtype=np.float32).reshape(3, 3)
    selected_T = np.asarray(selected_camera.T, dtype=np.float32).reshape(3)

    if plane_normal_direction is not None:
        roll_degrees = None
        pitch_degrees = None
        if average_reference_R is None:
            average_reference_R = selected_R
        R, T = camera_pose_from_world_gravity_direction(
            average_reference_R,
            selected_R,
            selected_T,
            plane_normal_direction,
        )
    elif average_gravity_direction is None:
        if roll_pitch_cache is None:
            roll_degrees, pitch_degrees = predict_roll_pitch(
                selected_camera,
                perspective_fields_path,
                perspective_fields_version,
                perspective_fields_device,
            )
        else:
            prediction_key = camera_prediction_key(
                name,
                selected_camera_index,
                selected_camera,
            )
            roll_degrees, pitch_degrees = roll_pitch_cache[prediction_key]
        R, T = gravity_aligned_camera_pose(
            selected_R,
            selected_T,
            roll_degrees,
            pitch_degrees,
        )
    else:
        roll_degrees = None
        pitch_degrees = None
        if selected_roll_pitch is not None:
            roll_degrees, pitch_degrees = selected_roll_pitch
        if average_reference_R is None:
            average_reference_R = selected_R
        R, T = camera_pose_from_world_gravity_direction(
            average_reference_R,
            selected_R,
            selected_T,
            average_gravity_direction,
        )

    if closest_camera:
        print(f"Closest camera index for {name}: {selected_camera_index}")
    else:
        print(f"Selected camera index for {name}: {selected_camera_index}")
    print(f"Selected camera image: {selected_camera.image_name}")
    print(f"Selected camera center: {selected_camera_center}")
    if plane_normal_direction is not None:
        plane_normal_direction = normalize_vector(plane_normal_direction)
        if render_direction_source is None:
            render_direction_source = "camera-forward-aligned normal fitted from camera centers"
        print(f"Using {render_direction_source}.")
        print(
            f"Final render direction: "
            f"({plane_normal_direction[0]:.10f}, "
            f"{plane_normal_direction[1]:.10f}, "
            f"{plane_normal_direction[2]:.10f})"
        )
    elif average_gravity_direction is None:
        print(
            f"PerspectiveFields prediction: roll={roll_degrees:.6f} deg, "
            f"pitch={pitch_degrees:.6f} deg"
        )
    else:
        print("Using averaged world gravity direction for final render.")
        if roll_degrees is not None and pitch_degrees is not None:
            print(
                f"Selected camera PerspectiveFields prediction: "
                f"roll={roll_degrees:.6f} deg, pitch={pitch_degrees:.6f} deg"
            )
        average_gravity_direction = normalize_vector(average_gravity_direction)
        print(
            f"Final world gravity direction: "
            f"({average_gravity_direction[0]:.10f}, "
            f"{average_gravity_direction[1]:.10f}, "
            f"{average_gravity_direction[2]:.10f})"
        )
    if plane_normal_direction is None:
        print("Gravity-aligned virtual camera R:")
    else:
        print("Plane-normal virtual camera R:")
    print(R)
    if plane_normal_direction is None:
        print("Gravity-aligned virtual camera T:")
    else:
        print("Plane-normal virtual camera T:")
    print(T)

    return R, T


def render_orthographic_camera(
    name,
    selected_camera,
    selected_camera_index,
    selected_camera_center,
    gaussians,
    background,
    perspective_fields_path,
    perspective_fields_version,
    perspective_fields_device,
    rendering_filepath,
    closest_camera=False,
    average_gravity_direction=None,
    average_reference_R=None,
    selected_roll_pitch=None,
    roll_pitch_cache=None,
    plane_normal_direction=None,
    render_direction_source=None,
    adaptive_view_scale=False,
):
    R, T = resolve_orthographic_camera_pose(
        name,
        selected_camera,
        selected_camera_index,
        selected_camera_center,
        perspective_fields_path,
        perspective_fields_version,
        perspective_fields_device,
        closest_camera=closest_camera,
        average_gravity_direction=average_gravity_direction,
        average_reference_R=average_reference_R,
        selected_roll_pitch=selected_roll_pitch,
        roll_pitch_cache=roll_pitch_cache,
        plane_normal_direction=plane_normal_direction,
        render_direction_source=render_direction_source,
    )

    rectangle = camera_orthogonal_bounding_rectangle(
        gaussians.get_xyz,
        R,
        T,
        include_depth_sample=adaptive_view_scale,
    )
    world_view_scale = select_ortho_view_scale(
        rectangle,
        WORLD_VIEW_SCALE,
        adaptive_view_scale,
    )
    ortho_bounds = build_camera_centered_ortho_bounds(rectangle, world_view_scale)
    left, right, bottom, top = ortho_bounds

    ortho_width = right - left
    ortho_height = top - bottom
    FoVx = 2 * math.atan(ortho_width / 2.0)
    FoVy = 2 * math.atan(ortho_height / 2.0)
    W, H = image_size_from_rectangle(rectangle)

    center_xy = rectangle["center_xy"].detach().cpu().numpy()
    print(
        f"Auto ortho bbox: width={rectangle['width']:.6f}, "
        f"height={rectangle['height']:.6f}, "
        f"center_xy=({center_xy[0]:.6f}, {center_xy[1]:.6f}), "
        f"points={rectangle['num_bbox_gaussians']}/{rectangle['num_total_gaussians']}"
    )
    print(
        f"Camera-centered ortho bounds after view scale: "
        f"left={left:.6f}, right={right:.6f}, "
        f"bottom={bottom:.6f}, top={top:.6f}"
    )
    print(f"World/view scale: {world_view_scale:.10f}")
    print(f"Output size: W={W}, H={H}")
    print(f"FoVx={FoVx:.6f}, FoVy={FoVy:.6f}")

    mycam = DummyCamera(
        R,
        T,
        FoVx,
        FoVy,
        W,
        H,
        ortho_bounds=ortho_bounds,
        view_scale=world_view_scale,
    )
    pipeline = DummyPipeline()

    terminal_summary(
        "Final render direction: "
        f"({R[0, 2]:.10f}, {R[1, 2]:.10f}, {R[2, 2]:.10f})"
    )
    terminal_summary("Detailed gravity direction estimation: see gravity_direction_estimation.log.")
    for _ in tqdm(range(1), desc="Rendering progress", file=render_progress_stream()):
        result = render(mycam, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(result, rendering_filepath)
    return R


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    background,
    perspective_fields_path,
    perspective_fields_version,
    perspective_fields_device,
    average_gravity_direction,
    camera_plane_down_normal,
    roll_pitch_cache=None,
    plane_normal_direction=None,
    render_direction_reference_R=None,
    render_direction_source=None,
    final_splat_direction=None,
    final_splat_source=None,
    visualize_support_score=False,
):
    render_path = os.path.join(model_path, "TDOM")
    makedirs(render_path, exist_ok=True)

    if camera_plane_down_normal and plane_normal_direction is None and len(views) > 0:
        plane_normal_direction = fit_camera_forward_aligned_camera_center_plane_normal(
            views
        )
        render_direction_source = "robust camera-center plane normal"

    closest_camera_index, closest_camera_center = find_center_camera(views)
    if closest_camera_index is None:
        print(f"[WARN] Skip {name}: no cameras available.")
        terminal_summary("No cameras available; rendering skipped.")
        return

    #closest_camera_index=0别删
    selected_camera = views[closest_camera_index]
    average_direction = final_splat_direction
    average_reference_R = None
    selected_roll_pitch = None
    if (
        roll_pitch_cache is None
        and plane_normal_direction is None
        and final_splat_direction is None
    ):
        if average_gravity_direction:
            roll_pitch_cache = predict_roll_pitch_batch(
                [
                    (camera_prediction_key(name, camera_index, camera), camera)
                    for camera_index, camera in enumerate(views)
                ],
                perspective_fields_path,
                perspective_fields_version,
                perspective_fields_device,
            )
        else:
            roll_pitch_cache = predict_roll_pitch_batch(
                [
                    (
                        camera_prediction_key(name, closest_camera_index, selected_camera),
                        selected_camera,
                    )
                ],
                perspective_fields_path,
                perspective_fields_version,
                perspective_fields_device,
            )
    if (
        average_gravity_direction
        and plane_normal_direction is None
        and final_splat_direction is None
    ):
        visualization_path = generate_unique_filepath(
            render_path,
            f"gravity_{name}_iter{iteration}",
            ".svg",
        )
        (
            average_direction,
            average_reference_R,
            selected_roll_pitch,
        ) = predict_average_gravity_direction(
            name,
            views,
            closest_camera_index,
            perspective_fields_path,
            perspective_fields_version,
            perspective_fields_device,
            visualization_path,
            roll_pitch_cache,
        )
    if final_splat_direction is not None and final_splat_source is not None:
        print(f"Final splat render direction source for {name}: {final_splat_source}")
    rendering_prefix = (
        "final_splat"
        if (
            final_splat_direction is not None
            or render_direction_source == "cached render direction from render_direction.txt"
        )
        else f"tdom_{name}_iter{iteration}"
    )
    if visualize_support_score:
        rendering_prefix = "support_score"
    rendering_filepath = generate_unique_filename(render_path, rendering_prefix)
    rendered_R = render_orthographic_camera(
        name,
        selected_camera,
        closest_camera_index,
        closest_camera_center,
        gaussians,
        background,
        perspective_fields_path,
        perspective_fields_version,
        perspective_fields_device,
        rendering_filepath,
        closest_camera=True,
        average_gravity_direction=average_direction,
        average_reference_R=(
            render_direction_reference_R
            if plane_normal_direction is not None
            else average_reference_R
        ),
        selected_roll_pitch=selected_roll_pitch,
        roll_pitch_cache=roll_pitch_cache,
        plane_normal_direction=plane_normal_direction,
        render_direction_source=render_direction_source,
        adaptive_view_scale=False,
    )
    if not os.path.exists(os.path.join(model_path, "render_direction.txt")):
        save_render_direction_cache(model_path, rendered_R)


def load_render_direction_cache(model_path):
    path = os.path.join(model_path, "render_direction.txt")
    if not os.path.exists(path):
        return None
    values = np.loadtxt(path, dtype=np.float64)
    if values.shape != (4, 3) or not np.isfinite(values).all():
        raise ValueError(f"Invalid render direction cache: {path}; expected 4 finite rows of 3 values.")
    direction = normalize_vector(values[0])
    reference_R = values[1:]
    if (
        not np.allclose(reference_R.T @ reference_R, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(reference_R), 1.0, atol=1e-5)
        or not np.allclose(reference_R[:, 2], direction, atol=1e-5)
    ):
        raise ValueError(f"Invalid orientation in {path}; remove the cache to predict again.")
    print(f"Using cached render direction: {path}")
    return direction, reference_R


def save_render_direction_cache(model_path, R):
    path = os.path.join(model_path, "render_direction.txt")
    values = np.vstack((normalize_vector(R[:, 2]), R))
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=model_path, delete=False) as stream:
            temporary_path = stream.name
            np.savetxt(
                stream, values, fmt="%.17g",
                header="World render direction (x y z), followed by 3 rows of reference camera R",
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    print(f"Saved render direction: {path}")


def get_direction_only_args(parser, args_cmdline):
    """Allow pre-training metadata without a saved model configuration."""
    defaults_parser = ArgumentParser(add_help=False)
    ModelParams(defaults_parser)
    merged = vars(defaults_parser.parse_args([])).copy()
    if os.path.exists(os.path.join(args_cmdline.model_path, "cfg_args")):
        args = get_combined_args(parser, args_cmdline=args_cmdline)
    else:
        args = args_cmdline
    merged.update({key: value for key, value in vars(args).items() if value is not None})
    return Namespace(**merged)


def load_direction_cameras(dataset):
    """Load the same train/test camera metadata without reading any point cloud."""
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](
            dataset.source_path, dataset.images, dataset.eval,
            load_images=False, load_point_cloud=False,
        )
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](
            dataset.source_path, dataset.white_background, dataset.eval,
            load_images=False, load_point_cloud=False,
        )
    else:
        raise ValueError(f"Could not recognize scene type: {dataset.source_path}")
    print("Loading Training Camera Metadata")
    train_cameras = cameraList_from_camInfos(
        scene_info.train_cameras, 1.0, dataset, load_images=False,
    )
    print("Loading Test Camera Metadata")
    test_cameras = cameraList_from_camInfos(
        scene_info.test_cameras, 1.0, dataset, load_images=False,
    )
    scene_cameras = sorted(
        train_cameras + test_cameras,
        key=lambda camera: str(camera.image_name),
    )
    print(
        f"Scene cameras for direction estimation: {len(scene_cameras)} "
        f"(train={len(train_cameras)}, test={len(test_cameras)})."
    )
    terminal_summary(f"Loaded {len(scene_cameras)} camera metadata.")
    return scene_cameras


def estimate_render_direction(dataset, perspective_fields_path,
                              perspective_fields_version, perspective_fields_device):
    """Save the default final-splat orientation without loading/rendering Gaussians."""
    with torch.no_grad():
        views = load_direction_cameras(dataset)
        camera_items = [("scene", index, camera) for index, camera in enumerate(views)]
        selected_index, selected_center = find_center_camera(views)
        if selected_index is None:
            raise ValueError("No cameras available for direction estimation.")
        direction, source = resolve_final_splat_direction(
            dataset.model_path, -1, camera_items,
            perspective_fields_path, perspective_fields_version, perspective_fields_device,
        )
        if direction is None:
            raise ValueError("Final-splat direction estimation did not produce a trusted direction.")
        selected_camera = views[selected_index]
        if source is not None:
            print(f"Final splat render direction source for scene: {source}")
        R, _ = resolve_orthographic_camera_pose(
            "scene", selected_camera, selected_index, selected_center,
            perspective_fields_path, perspective_fields_version, perspective_fields_device,
            closest_camera=True, average_gravity_direction=direction,
        )
        save_render_direction_cache(dataset.model_path, R)
        terminal_summary(
            "Final render direction: "
            f"({R[0, 2]:.10f}, {R[1, 2]:.10f}, {R[2, 2]:.10f})"
        )
        terminal_summary("Detailed gravity direction estimation: see gravity_direction_estimation.log.")


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    perspective_fields_path: str,
    perspective_fields_version: str,
    perspective_fields_device: str,
    average_gravity_direction: bool,
    camera_plane_down_normal: bool,
    final_splat: bool,
    visualize_support_score: bool = False,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(
            dataset, gaussians, load_iteration=iteration, shuffle=False, load_images=False,
            point_cloud_filename=(
                "point_cloud_weak_support.ply" if visualize_support_score else "point_cloud.ply"
            ),
        )

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        train_cameras = list(scene.getTrainCameras())
        test_cameras = list(scene.getTestCameras())
        scene_cameras = sorted(
            train_cameras + test_cameras,
            key=lambda camera: str(camera.image_name),
        )
        scene_camera_items = [
            ("scene", camera_index, camera)
            for camera_index, camera in enumerate(scene_cameras)
        ]
        print(
            f"Scene cameras for single TDOM: {len(scene_cameras)} "
            f"(train={len(train_cameras)}, test={len(test_cameras)})."
        )

        cached_direction = load_render_direction_cache(dataset.model_path)
        if cached_direction is None:
            terminal_summary(
                f"Loaded iteration {scene.loaded_iter}: {len(scene_cameras)} camera metadata, "
                f"{gaussians.get_xyz.shape[0]} Gaussians."
            )
        if cached_direction is not None:
            # Skip all direction prediction; use the saved orientation below.
            final_splat = True
        final_splat_direction = None
        final_splat_source = None
        if final_splat and cached_direction is None:
            final_splat_direction, final_splat_source = resolve_final_splat_direction(
                dataset.model_path,
                scene.loaded_iter,
                scene_camera_items,
                perspective_fields_path,
                perspective_fields_version,
                perspective_fields_device,
            )
            if final_splat_direction is None:
                raise ValueError("Final-splat direction estimation did not produce a trusted direction.")

        plane_normal_direction = None
        render_direction_reference_R = None
        render_direction_source = None
        if cached_direction is not None:
            plane_normal_direction, render_direction_reference_R = cached_direction
            render_direction_source = "cached render direction from render_direction.txt"
        if not final_splat and camera_plane_down_normal and scene_cameras:
            plane_normal_direction = (
                fit_camera_forward_aligned_camera_center_plane_normal(scene_cameras)
            )
            render_direction_source = "robust camera-center plane normal"

        prediction_items = []
        if not final_splat and not camera_plane_down_normal:
            prediction_items = collect_required_prediction_items(
                "scene", scene_cameras, average_gravity_direction
            )
        roll_pitch_cache = None
        if not final_splat:
            roll_pitch_cache = predict_roll_pitch_batch(
                prediction_items,
                perspective_fields_path,
                perspective_fields_version,
                perspective_fields_device,
            )

        render_set(
            dataset.model_path,
            "scene",
            scene.loaded_iter,
            scene_cameras,
            gaussians,
            background,
            perspective_fields_path,
            perspective_fields_version,
            perspective_fields_device,
            average_gravity_direction,
            camera_plane_down_normal,
            roll_pitch_cache,
            plane_normal_direction,
            render_direction_reference_R,
            render_direction_source,
            final_splat_direction,
            final_splat_source,
            visualize_support_score,
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Render a single scene orthophoto")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--visualize_support_score", action="store_true",
        help=("Render point_cloud_weak_support.ply from the selected iteration "
              "as TDOM/support_score_*.png. Requires its training export."),
    )
    parser.add_argument(
        "--direction_only", action="store_true",
        help=("Estimate and cache the default final-splat orientation from camera metadata "
              "and images without a trained model or rendering. Other direction modes "
              "are unsupported unless an existing valid direction cache is used."),
    )
    parser.add_argument(
        "--perspective_fields_path",
        default=DEFAULT_PERSPECTIVE_FIELDS_PATH,
        type=str,
    )
    parser.add_argument(
        "--perspective_fields_version",
        default=DEFAULT_PERSPECTIVE_FIELDS_VERSION,
        type=str,
    )
    parser.add_argument(
        "--perspective_fields_device",
        default="cuda",
        choices=("cuda", "cpu"),
    )
    parser.add_argument(
        "--average_gravity_direction",
        action="store_true",
        help=(
            "Predict gravity for all cameras, reject "
            "outliers, and render with the robust averaged direction."
        ),
    )
    parser.add_argument(
        "--camera_plane_down_normal",
        action="store_true",
        help=(
            "Robustly fit a plane using all camera "
            "centers, then orient its normal toward sampled camera forward "
            "directions. This mode does not use PerspectiveFields."
        ),
    )
    final_splat_group = parser.add_mutually_exclusive_group()
    final_splat_group.add_argument(
        "--final_splat",
        dest="final_splat",
        action="store_true",
        help=(
            "Combine average gravity, camera-plane normal confidence, and "
            "nadir-camera agreement to choose the render direction "
            "(default when no other direction mode is selected)."
        ),
    )
    final_splat_group.add_argument(
        "--no_final_splat",
        dest="final_splat",
        action="store_false",
        help="Disable final-splat direction selection; an existing render_direction.txt still takes precedence.",
    )
    parser.set_defaults(final_splat=None)
    args_cmdline = parser.parse_args()
    if args_cmdline.final_splat is None:
        args_cmdline.final_splat = not (
            args_cmdline.average_gravity_direction
            or args_cmdline.camera_plane_down_normal
        )
    if not args_cmdline.model_path:
        parser.error("--model_path/-m is required")
    enable_render_log(args_cmdline.model_path, quiet=args_cmdline.quiet)
    if args_cmdline.direction_only and args_cmdline.visualize_support_score:
        raise ValueError("--direction_only cannot be combined with --visualize_support_score.")
    if args_cmdline.direction_only:
        cached_direction = load_render_direction_cache(args_cmdline.model_path)
        if cached_direction is not None:
            _, R = cached_direction
            terminal_summary(
                "Final render direction: "
                f"({R[0, 2]:.10f}, {R[1, 2]:.10f}, {R[2, 2]:.10f})"
            )
            terminal_summary("Detailed gravity direction estimation: see gravity_direction_estimation.log.")
            raise SystemExit(0)
        args = get_direction_only_args(parser, args_cmdline)
        if not args.source_path:
            raise ValueError("--source_path/-s is required without a render_direction.txt cache.")
        if not args.final_splat or args.average_gravity_direction or args.camera_plane_down_normal:
            raise ValueError("--direction_only supports the default final-splat direction mode only.")
        print("Estimating render direction " + args.model_path)
        safe_state(False)
        estimate_render_direction(
            model.extract(args), args.perspective_fields_path,
            args.perspective_fields_version, args.perspective_fields_device,
        )
        raise SystemExit(0)
    args = get_combined_args(parser, args_cmdline=args_cmdline)

    print("Rendering " + args.model_path)
    if args.final_splat and (
        args.average_gravity_direction or args.camera_plane_down_normal
    ):
        print(
            "[INFO] --average_gravity_direction and --camera_plane_down_normal "
            "are ignored when --final_splat is explicitly selected."
        )
    elif args.camera_plane_down_normal and args.average_gravity_direction:
        print(
            "[INFO] --average_gravity_direction is ignored when "
            "--camera_plane_down_normal is used."
        )
    # --quiet suppresses terminal summaries; the full timestamped log is retained.
    safe_state(False)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.perspective_fields_path,
        args.perspective_fields_version,
        args.perspective_fields_device,
        (
            args.average_gravity_direction
            and not args.camera_plane_down_normal
            and not args.final_splat
        ),
        args.camera_plane_down_normal and not args.final_splat,
        args.final_splat,
        args.visualize_support_score,
    )
