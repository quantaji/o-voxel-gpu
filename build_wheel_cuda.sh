#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="3.8"
TORCH_VERSION="2.4.0"
CUDA_VERSION="11.8"
OUT_DIR_ARG="wheels"
ARCH_LIST=""

usage() {
    cat <<'EOF'
Usage:
  ./build_wheel_cuda.sh [options]

Options:
  --python VERSION      Python version. Default: 3.8
  --torch VERSION       PyTorch version. Default: 2.4.0
  --cuda VERSION        CUDA version. Default: 11.8
  --out DIR             Output directory. Default: wheels
  --arch-list LIST      TORCH_CUDA_ARCH_LIST. Default: auto by PyTorch/CUDA version.
  -h, --help            Show this help message.

Example:
  ./build_wheel_cuda.sh --python 3.8 --torch 2.4.0 --cuda 11.8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
    --python)
        PYTHON_VERSION="$2"
        shift 2
        ;;
    --torch)
        TORCH_VERSION="$2"
        shift 2
        ;;
    --cuda)
        CUDA_VERSION="$2"
        shift 2
        ;;
    --out)
        OUT_DIR_ARG="$2"
        shift 2
        ;;
    --arch-list)
        ARCH_LIST="$2"
        shift 2
        ;;
    -h | --help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
        ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

if [[ "$OUT_DIR_ARG" = /* ]]; then
    OUT_DIR="$OUT_DIR_ARG"
else
    OUT_DIR="$REPO_DIR/$OUT_DIR_ARG"
fi
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

torch_cuda_tag() {
    local version="$1"
    echo "cu${version//./}"
}

python_cp_tag() {
    case "$1" in
    3.8) echo "cp38" ;;
    3.9) echo "cp39" ;;
    3.10) echo "cp310" ;;
    3.11) echo "cp311" ;;
    3.12) echo "cp312" ;;
    3.13) echo "cp313" ;;
    3.14) echo "cp314" ;;
    *)
        echo "Unsupported Python version: $1" >&2
        echo "Add a Python tag mapping in python_cp_tag()." >&2
        exit 2
        ;;
    esac
}

docker_image_for_cuda() {
    case "$1" in
    11.8)
        echo "nvidia/cuda:11.8.0-devel-ubuntu22.04"
        ;;
    12.1)
        echo "nvidia/cuda:12.1.0-devel-ubuntu22.04"
        ;;
    12.4)
        echo "nvidia/cuda:12.4.0-devel-ubuntu22.04"
        ;;
    12.6)
        echo "nvidia/cuda:12.6.0-devel-ubuntu22.04"
        ;;
    12.8)
        echo "nvidia/cuda:12.8.0-devel-ubuntu22.04"
        ;;
    12.9)
        echo "nvidia/cuda:12.9.0-devel-ubuntu22.04"
        ;;
    13.0)
        echo "nvidia/cuda:13.0.0-devel-ubuntu22.04"
        ;;
    *)
        echo "Unsupported CUDA version: $1" >&2
        echo "Add a Docker image mapping in docker_image_for_cuda()." >&2
        exit 2
        ;;
    esac
}

default_arch_list_for_torch_cuda() {
    local torch_version="$1"
    local cuda_version="$2"
    local torch_mm="${torch_version%.*}"

    case "$cuda_version" in
    11.8 | 12.1 | 12.4 | 12.6)
        echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
        ;;
    12.8)
        case "$torch_mm" in
        2.7 | 2.8)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;12.0+PTX"
            ;;
        2.9 | 2.10 | 2.11 | 2.12 | 2.13 | 2.14)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0+PTX"
            ;;
        *)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
            ;;
        esac
        ;;
    12.9)
        case "$torch_mm" in
        2.8)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.1;10.3;12.0;12.1+PTX"
            ;;
        2.9 | 2.10 | 2.11 | 2.12 | 2.13 | 2.14)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;12.0;12.1+PTX"
            ;;
        *)
            echo "7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0+PTX"
            ;;
        esac
        ;;
    13.0)
        case "$torch_mm" in
        2.9 | 2.10 | 2.11 | 2.12 | 2.13 | 2.14)
            echo "7.5;8.0;8.6;8.7;8.9;9.0;10.0;10.3;11.0;12.0;12.1+PTX"
            ;;
        *)
            echo "7.5;8.0;8.6;8.7;8.9;9.0+PTX"
            ;;
        esac
        ;;
    *)
        echo "Unsupported CUDA version: $cuda_version" >&2
        echo "Add an arch list mapping in default_arch_list_for_torch_cuda()." >&2
        exit 2
        ;;
    esac
}

command -v docker >/dev/null 2>&1 || {
    echo "docker is required but was not found in PATH." >&2
    exit 1
}
command -v curl >/dev/null 2>&1 || {
    echo "curl is required for the PyTorch wheel preflight check." >&2
    exit 1
}

TORCH_CUDA_TAG="$(torch_cuda_tag "$CUDA_VERSION")"
PYTHON_TAG="$(python_cp_tag "$PYTHON_VERSION")"
TORCH_INDEX_URL="https://download.pytorch.org/whl/$TORCH_CUDA_TAG"
DOCKER_IMAGE="$(docker_image_for_cuda "$CUDA_VERSION")"
OVOXEL_VERSION="0.0.1+torch${TORCH_VERSION}.${TORCH_CUDA_TAG}"
if [[ -z "$ARCH_LIST" ]]; then
    ARCH_LIST="$(default_arch_list_for_torch_cuda "$TORCH_VERSION" "$CUDA_VERSION")"
fi
HOST_UID="$(id -u)"
HOST_GID="$(id -g)"

TORCH_WHEEL_PATTERN="torch-${TORCH_VERSION}(%2B|\\+)${TORCH_CUDA_TAG}-${PYTHON_TAG}-${PYTHON_TAG}-[^\"'<> ]*x86_64\\.whl"
TORCH_INDEX_HTML="$(mktemp)"
trap 'rm -f "$TORCH_INDEX_HTML"' EXIT
curl -fsSL "${TORCH_INDEX_URL}/torch/" -o "$TORCH_INDEX_HTML"
if ! grep -Eq "$TORCH_WHEEL_PATTERN" "$TORCH_INDEX_HTML"; then
    echo "PyTorch wheel not found for this combination:" >&2
    echo "  python=$PYTHON_VERSION ($PYTHON_TAG)" >&2
    echo "  torch=$TORCH_VERSION" >&2
    echo "  cuda=$CUDA_VERSION ($TORCH_CUDA_TAG)" >&2
    echo "Checked: ${TORCH_INDEX_URL}/torch/" >&2
    exit 2
fi

echo "Repository:      $REPO_DIR"
echo "Output:          $OUT_DIR"
echo "Docker image:    $DOCKER_IMAGE"
echo "Python:          $PYTHON_VERSION"
echo "PyTorch:         $TORCH_VERSION"
echo "CUDA:            $CUDA_VERSION ($TORCH_CUDA_TAG)"
echo "o_voxel version: $OVOXEL_VERSION"
echo "Torch index:     $TORCH_INDEX_URL"
echo "CUDA arch list:  $ARCH_LIST"

docker run --rm -i \
    -v "$REPO_DIR:/src:ro" \
    -v "$OUT_DIR:/wheelhouse" \
    -e HOST_UID="$HOST_UID" \
    -e HOST_GID="$HOST_GID" \
    -e PYTHON_VERSION="$PYTHON_VERSION" \
    -e TORCH_VERSION="$TORCH_VERSION" \
    -e CUDA_VERSION="$CUDA_VERSION" \
    -e OVOXEL_VERSION="$OVOXEL_VERSION" \
    -e TORCH_INDEX_URL="$TORCH_INDEX_URL" \
    -e TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
    "$DOCKER_IMAGE" \
    bash -se <<'IN_CONTAINER'
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

apt-get update
apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    bzip2 \
    git \
    build-essential
rm -rf /var/lib/apt/lists/*

curl -fsSL -o /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash /tmp/miniforge.sh -b -p /opt/conda
rm -f /tmp/miniforge.sh

. /opt/conda/etc/profile.d/conda.sh
conda create -y -n wheel-build "python=${PYTHON_VERSION}" pip
conda activate wheel-build

python -m pip install --upgrade pip setuptools wheel ninja numpy
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX_URL"

python --version
python - <<'PY'
import os
import sys
import torch

expected_torch = os.environ["TORCH_VERSION"]
expected_cuda = os.environ["CUDA_VERSION"]
actual_torch = torch.__version__.split("+", 1)[0]
actual_cuda = torch.version.cuda
print(f"torch={torch.__version__}, torch.version.cuda={actual_cuda}")
if actual_torch != expected_torch:
    sys.exit(f"Expected torch {expected_torch}, got {torch.__version__}")
if actual_cuda != expected_cuda:
    sys.exit(f"Expected torch CUDA {expected_cuda}, got {actual_cuda}")
PY
nvcc --version
gcc --version

mkdir -p /tmp/build
cp -a /src/. /tmp/build/o-voxel
cd /tmp/build/o-voxel
rm -rf build dist wheels *.egg-info __pycache__
find . -name __pycache__ -type d -prune -exec rm -rf {} +

export MAX_JOBS="$(nproc)"
export FORCE_CUDA=1

python -m pip wheel . --no-deps --no-build-isolation -w /wheelhouse

chown -R "$HOST_UID:$HOST_GID" /wheelhouse
echo "Built wheels:"
ls -lh /wheelhouse/*.whl
IN_CONTAINER

echo "Wheel output directory: $OUT_DIR"
