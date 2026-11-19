#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$task_dir"
exec > >(tee -a "$task_dir/install.log") 2>&1

python -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'
# Use this folder's training rasterizer, never the orthographic version.
# CUDA 11.8 nvcc must be available through PATH / CUDA_HOME.
python -m pip install --no-build-isolation \
    ./submodules/diff-gaussian-rasterization \
    ./submodules/simple-knn \
    ./submodules/fused-ssim
python -c 'import diff_gaussian_rasterization, simple_knn._C, fused_ssim; print("Training extensions imported successfully.")'
