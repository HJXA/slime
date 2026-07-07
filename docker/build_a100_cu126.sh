#!/usr/bin/env bash

set -Eeuo pipefail

# 打印命令使用说明，便于在 A100 机器上直接构建迁移镜像。
usage() {
  cat <<'USAGE'
用法:
  bash docker/build_a100_cu126.sh [选项]

目标:
  构建 A100 / CUDA 12.6 / cu126 slime Docker 镜像。

常用环境变量:
  IMAGE_TAG          默认 slime:a100-cu126
  CUDA_IMAGE         默认 nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
  APP_USER           默认 slime
  APP_UID            默认当前用户 uid
  APP_GID            默认当前用户 gid
  BASE_DIR           默认 /home/${APP_USER}/slime_a100_cu126
  ENV_NAME           默认 slime-a100-cu126
  PATCH_VERSION      默认 v0.5.9.a100
  SLIME_A100_REPO    默认 https://github.com/jason9693/slime-a100.git
  SLIME_A100_REF     默认 main
  MAX_JOBS           默认 32

示例:
  bash docker/build_a100_cu126.sh
  IMAGE_TAG=my-slime:a100-cu126 MAX_JOBS=16 bash docker/build_a100_cu126.sh

运行验证:
  docker run --rm --gpus all --ipc=host --shm-size=16g my-slime:a100-cu126 verify-a100-cu126
USAGE
}

# 支持简单 help，不触发 docker build。
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

IMAGE_TAG="${IMAGE_TAG:-slime:a100-cu126}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04}"
APP_USER="${APP_USER:-slime}"
APP_UID="${APP_UID:-$(id -u)}"
APP_GID="${APP_GID:-$(id -g)}"
BASE_DIR="${BASE_DIR:-/home/${APP_USER}/slime_a100_cu126}"
ENV_NAME="${ENV_NAME:-slime-a100-cu126}"
PATCH_VERSION="${PATCH_VERSION:-v0.5.9.a100}"
SLIME_A100_REPO="${SLIME_A100_REPO:-https://github.com/jason9693/slime-a100.git}"
SLIME_A100_REF="${SLIME_A100_REF:-main}"
MAX_JOBS="${MAX_JOBS:-32}"

echo "[信息] 构建镜像 ${IMAGE_TAG}"
echo "[信息] CUDA_IMAGE=${CUDA_IMAGE}"
echo "[信息] APP_USER=${APP_USER}, APP_UID=${APP_UID}, APP_GID=${APP_GID}"
echo "[信息] BASE_DIR=${BASE_DIR}"
echo "[信息] PATCH_VERSION=${PATCH_VERSION}"

# 使用仓库根目录作为 build context，确保 Dockerfile 能复制构建脚本和验证脚本。
docker build \
  -f docker/Dockerfile.a100-cu126 \
  --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
  --build-arg APP_USER="${APP_USER}" \
  --build-arg APP_UID="${APP_UID}" \
  --build-arg APP_GID="${APP_GID}" \
  --build-arg BASE_DIR="${BASE_DIR}" \
  --build-arg ENV_NAME="${ENV_NAME}" \
  --build-arg PATCH_VERSION="${PATCH_VERSION}" \
  --build-arg SLIME_A100_REPO="${SLIME_A100_REPO}" \
  --build-arg SLIME_A100_REF="${SLIME_A100_REF}" \
  --build-arg MAX_JOBS="${MAX_JOBS}" \
  -t "${IMAGE_TAG}" \
  .

echo "[信息] 镜像构建完成：${IMAGE_TAG}"
echo "[信息] 在 A100 机器上运行验证："
echo "docker run --rm --gpus all --ipc=host --shm-size=16g ${IMAGE_TAG} verify-a100-cu126"
