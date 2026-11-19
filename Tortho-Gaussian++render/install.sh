#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$task_dir"
exec > >(tee -a "$task_dir/install.log") 2>&1

python -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)'

# Download from the author instead of redistributing PerspectiveFields.
# Its own LICENSE applies; keep this directory out of the published repository.
perspective_revision=d54be737d6eacfb9d39a2b7079a494924b45bb6c
if [[ ! -f PerspectiveFields/perspective2d/perspectivefields.py ]]; then
    if [[ -e PerspectiveFields ]]; then
        printf '%s\n' 'PerspectiveFields exists but is incomplete; inspect it before retrying.' >&2
        exit 1
    fi
    git clone https://github.com/jinlinyi/PerspectiveFields.git PerspectiveFields
    git -C PerspectiveFields checkout --detach "$perspective_revision"
fi

# environment.yml provides inference imports without upstream demo/training extras.
# The helper imports this local source directory directly, so no second Python is needed.
python - <<'PY'
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path('PerspectiveFields').resolve()))
from perspective2d import PerspectiveFields
from perspective2d.perspectivefields import model_zoo

assert 'Paramnet-360Cities-edina-uncentered' in model_zoo
print('PerspectiveFields imports successfully in the rendering environment.')
print('The first inference downloads its model weights into the PyTorch cache.')
PY

# CUDA 11.8 nvcc must be available through PATH / CUDA_HOME.
# Install only this folder's orthographic rasterizer in this environment.
python -m pip install --no-build-isolation \
    ./submodules/diff-gaussian-rasterization \
    ./submodules/simple-knn
python -c 'import diff_gaussian_rasterization, simple_knn._C; print("Orthographic extensions imported successfully.")'
