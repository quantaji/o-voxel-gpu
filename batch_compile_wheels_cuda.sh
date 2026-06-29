#!/usr/bin/env bash
set -euo pipefail

OUT_DIR_ARG="wheels"
ARCH_LIST=""
DRY_RUN=0
FAIL_FAST=0

usage() {
    cat <<'EOF'
Usage:
  ./batch_compile_wheels_cuda.sh [options]

Options:
  --out DIR             Output directory passed to build_wheel_cuda.sh. Default: wheels
  --arch-list LIST      TORCH_CUDA_ARCH_LIST passed to build_wheel_cuda.sh.
  --dry-run             Print the 225 build commands without running them.
  --keep-going          Accepted for compatibility. This is now the default.
  --fail-fast           Stop immediately after the first failed build.
  -h, --help            Show this help message.

This script enumerates the 225 Linux x86_64 CUDA torch wheels found in the
official PyTorch wheel indexes from torch 2.4.0 through 2.12.1.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT_DIR_ARG="$2"
            shift 2
            ;;
        --arch-list)
            ARCH_LIST="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --keep-going)
            FAIL_FAST=0
            shift
            ;;
        --fail-fast)
            FAIL_FAST=1
            shift
            ;;
        -h|--help)
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
BUILD_SCRIPT="$SCRIPT_DIR/build_wheel_cuda.sh"

if [[ "$OUT_DIR_ARG" = /* ]]; then
    OUT_DIR="$OUT_DIR_ARG"
else
    OUT_DIR="$SCRIPT_DIR/$OUT_DIR_ARG"
fi
mkdir -p "$OUT_DIR"

if [[ ! -x "$BUILD_SCRIPT" ]]; then
    echo "Build script is not executable: $BUILD_SCRIPT" >&2
    echo "Run: chmod +x build_wheel_cuda.sh batch_compile_wheels_cuda.sh" >&2
    exit 1
fi

MATRIX=(
    "2.4.0|11.8|3.8 3.9 3.10 3.11 3.12"
    "2.4.0|12.1|3.8 3.9 3.10 3.11 3.12"
    "2.4.0|12.4|3.8 3.9 3.10 3.11 3.12"
    "2.4.1|11.8|3.8 3.9 3.10 3.11 3.12"
    "2.4.1|12.1|3.8 3.9 3.10 3.11 3.12"
    "2.4.1|12.4|3.8 3.9 3.10 3.11 3.12"
    "2.5.0|11.8|3.9 3.10 3.11 3.12 3.13"
    "2.5.0|12.1|3.9 3.10 3.11 3.12 3.13"
    "2.5.0|12.4|3.9 3.10 3.11 3.12 3.13"
    "2.5.1|11.8|3.9 3.10 3.11 3.12 3.13"
    "2.5.1|12.1|3.9 3.10 3.11 3.12 3.13"
    "2.5.1|12.4|3.9 3.10 3.11 3.12 3.13"
    "2.6.0|11.8|3.9 3.10 3.11 3.12 3.13"
    "2.6.0|12.4|3.9 3.10 3.11 3.12 3.13"
    "2.6.0|12.6|3.9 3.10 3.11 3.12 3.13"
    "2.7.0|11.8|3.9 3.10 3.11 3.12 3.13"
    "2.7.0|12.6|3.9 3.10 3.11 3.12 3.13"
    "2.7.0|12.8|3.9 3.10 3.11 3.12 3.13"
    "2.7.1|11.8|3.9 3.10 3.11 3.12 3.13"
    "2.7.1|12.6|3.9 3.10 3.11 3.12 3.13"
    "2.7.1|12.8|3.9 3.10 3.11 3.12 3.13"
    "2.8.0|12.6|3.9 3.10 3.11 3.12 3.13"
    "2.8.0|12.8|3.9 3.10 3.11 3.12 3.13"
    "2.8.0|12.9|3.9 3.10 3.11 3.12 3.13"
    "2.9.0|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.9.0|12.8|3.10 3.11 3.12 3.13 3.14"
    "2.9.0|12.9|3.10 3.11 3.12 3.13 3.14"
    "2.9.0|13.0|3.10 3.11 3.12 3.13 3.14"
    "2.9.1|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.9.1|12.8|3.10 3.11 3.12 3.13 3.14"
    "2.9.1|12.9|3.10 3.11 3.12 3.13 3.14"
    "2.9.1|13.0|3.10 3.11 3.12 3.13 3.14"
    "2.10.0|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.10.0|12.8|3.10 3.11 3.12 3.13 3.14"
    "2.10.0|12.9|3.10 3.11 3.12 3.13 3.14"
    "2.10.0|13.0|3.10 3.11 3.12 3.13 3.14"
    "2.11.0|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.11.0|12.8|3.10 3.11 3.12 3.13 3.14"
    "2.11.0|12.9|3.10 3.11 3.12 3.13 3.14"
    "2.11.0|13.0|3.10 3.11 3.12 3.13 3.14"
    "2.12.0|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.12.0|13.0|3.10 3.11 3.12 3.13 3.14"
    "2.12.1|12.6|3.10 3.11 3.12 3.13 3.14"
    "2.12.1|12.9|3.10 3.11 3.12 3.13 3.14"
    "2.12.1|13.0|3.10 3.11 3.12 3.13 3.14"
)

build_args=()
if [[ -n "$ARCH_LIST" ]]; then
    build_args+=(--arch-list "$ARCH_LIST")
fi

total=0
for row in "${MATRIX[@]}"; do
    IFS='|' read -r _ _ python_versions <<<"$row"
    for _ in $python_versions; do
        total=$((total + 1))
    done
done

echo "Total wheel builds: $total"
if [[ "$total" -ne 225 ]]; then
    echo "Internal matrix error: expected 225 builds." >&2
    exit 1
fi

failed=()
skipped=0
built=0
index=0
for row in "${MATRIX[@]}"; do
    IFS='|' read -r torch_version cuda_version python_versions <<<"$row"
    for python_version in $python_versions; do
        index=$((index + 1))
        echo
        echo "[$index/$total] torch=$torch_version cuda=$cuda_version python=$python_version"
        torch_cuda_tag="cu${cuda_version//./}"
        python_tag="cp${python_version//./}"
        wheel_pattern="$OUT_DIR"/*"+torch${torch_version}.${torch_cuda_tag}"-"${python_tag}"-"${python_tag}"-*x86_64.whl
        existing_wheels=($wheel_pattern)
        if [[ -e "${existing_wheels[0]}" ]]; then
            skipped=$((skipped + 1))
            echo "  skip existing: ${existing_wheels[0]}"
            continue
        fi
        cmd=(
            "$BUILD_SCRIPT"
            --python "$python_version"
            --torch "$torch_version"
            --cuda "$cuda_version"
            --out "$OUT_DIR_ARG"
            "${build_args[@]}"
        )
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '  %q' "${cmd[@]}"
            echo
            continue
        fi
        if ! "${cmd[@]}"; then
            failed+=("torch=$torch_version cuda=$cuda_version python=$python_version")
            if [[ "$FAIL_FAST" -eq 1 ]]; then
                echo "Stopping after failed build." >&2
                exit 1
            fi
        else
            built=$((built + 1))
        fi
    done
done

if [[ "${#failed[@]}" -gt 0 ]]; then
    echo
    echo "Failed builds (${#failed[@]}):" >&2
    printf '  %s\n' "${failed[@]}" >&2
fi

echo
echo "Batch complete: total=$total built=$built skipped=$skipped failed=${#failed[@]}"
