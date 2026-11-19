"""Prepare a render orientation only when final support-mask pruning needs it."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from utils.support_prune_mask import load_render_orientation


RENDER_SCRIPT = Path(__file__).resolve().parents[2] / "Tortho-Gaussian++render" / "render_dummy.py"
DIRECTION_FAILURE_MESSAGE = "预测方向不可信，请关闭 --final_support_prune_mask_thresh 剪枝，重新进行训练。"


def _executable(value):
    candidate = os.path.expanduser(str(value))
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return os.path.abspath(candidate)
    return shutil.which(candidate)


def resolve_render_command(render_python=None):
    """Find TGPP-render without activating or modifying either environment."""
    if render_python is not None:
        executable = _executable(render_python)
        if executable is None:
            raise ValueError("Render Python is not executable: {}. Set --render_python to the TGPP-render Python.".format(render_python))
        return [executable]

    environments = Path(sys.executable).resolve().parent.parent.parent
    if environments.name == "envs":
        sibling = environments / "TGPP-render" / "bin" / "python"
        if sibling.is_file() and os.access(str(sibling), os.X_OK):
            return [str(sibling)]

    conda = _executable(os.environ["CONDA_EXE"]) if os.environ.get("CONDA_EXE") else None
    conda = conda or shutil.which("conda")
    if conda:
        return [conda, "run", "-n", "TGPP-render", "--no-capture-output", "python"]
    raise ValueError("Cannot locate the TGPP-render environment. Set --render_python to its Python executable.")


def _prepare_render_direction(args, model_path, cache_path, log_path):
    if os.path.lexists(str(cache_path)):
        try:
            load_render_orientation(model_path)
        except (OSError, ValueError) as error:
            raise ValueError("Invalid render direction cache {}: {}".format(cache_path, error)) from None
        return str(cache_path)
    command = resolve_render_command(args.render_python)
    if not RENDER_SCRIPT.is_file():
        raise ValueError("Missing render direction entry point: {}".format(RENDER_SCRIPT))
    command += [str(RENDER_SCRIPT), "--direction_only", "-s", os.path.abspath(args.source_path),
                "-m", str(model_path), "--images", args.images,
                "--resolution", str(args.resolution), "--quiet"]
    if args.eval:
        command.append("--eval")
    if args.white_background:
        command.append("--white_background")

    model_path.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise ValueError("Direction estimation failed (exit {}).".format(result.returncode))
    try:
        load_render_orientation(model_path)
    except (OSError, ValueError) as error:
        raise ValueError("Direction cache validation failed: {}".format(error)) from None
    return str(cache_path)


def ensure_render_direction(args):
    """Final pruning overrides skip; ordinary training never inspects direction files."""
    args.skip_direction_estimation = args.final_support_prune_mask_thresh is None
    if args.skip_direction_estimation:
        return None

    model_path = Path(args.model_path).absolute()
    cache_path = model_path / "render_direction.txt"
    log_path = model_path / "gravity_direction_estimation.log"
    try:
        return _prepare_render_direction(args, model_path, cache_path, log_path)
    except (OSError, ValueError) as error:
        try:
            model_path.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write("Direction preparation failed: {}: {}\n".format(type(error).__name__, error))
        except OSError:
            # An unwritable output directory cannot receive diagnostics either.
            pass
        raise ValueError(DIRECTION_FAILURE_MESSAGE) from None
