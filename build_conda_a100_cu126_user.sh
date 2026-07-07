#!/usr/bin/env bash

set -Eeuo pipefail

# 输出普通进度信息，方便长时间构建时定位当前阶段。
log() {
  printf '[信息] %s\n' "$*"
}

# 输出错误并退出，避免失败后继续污染后续安装状态。
die() {
  printf '[错误] %s\n' "$*" >&2
  exit 1
}

# 打印使用说明，明确默认路径全部位于当前用户目录下。
usage() {
  cat <<'USAGE'
用法:
  bash build_conda_a100_cu126_user.sh [选项]

目标:
  为 A100 / SM80 构建 CUDA 12.6 / cu126 slime 环境，并把仓库、micromamba、
  conda env、日志和激活脚本默认放到当前用户 HOME 下。
  如果 /root 下已有资源且当前用户有权限正常使用，脚本会优先复用，避免重复下载。

常用选项:
  --prepare-only     只克隆 A100 fork 并生成 cu126/user-home 构建脚本，不执行构建
  --update-repo      如果 A100 fork 已存在，则 fetch 后切到 SLIME_A100_REF
  --no-verify        构建完成后不运行 Python 导入验证
  --force-build      即使构建完成标识已存在，也重新执行构建阶段
  --force-verify     即使验证完成标识已存在，也重新执行验证阶段
  --reset-resume     清空断点标识后重新判断各阶段
  -h, --help         显示本说明

可覆盖环境变量:
  BASE_DIR           默认 ${HOME}/slime_a100_cu126
  ENV_NAME           默认 slime-a100-cu126
  SLIME_A100_REPO    默认 https://github.com/jason9693/slime-a100.git
  SLIME_A100_REF     默认 main
  PATCH_VERSION      默认 v0.5.9.a100
  MAX_JOBS           默认 32
  TORCH_CUDA_ARCH_LIST 默认 8.0
  REUSE_ROOT_EXISTING 默认 1；设为 0 时不复用 /root 下的已有资源
  ROOT_BASE_DIR      默认 /root；用于查找可复用的系统级资源
  INSTALL_LOG        默认 ${BASE_DIR}/install_a100_cu126_<时间戳>.log
  BUILD_LOG          默认 ${BASE_DIR}/build_cuda126.log
  RESUME_DIR         默认 ${BASE_DIR}/.resume_a100_cu126
  RESUME_ENABLED     默认 1；设为 0 时忽略断点标识
  PIP_GIT_SOURCE_DIR 默认 ${BASE_DIR}/sources/pip_git；GitHub pip 包源码缓存
  RETRY_ATTEMPTS     默认 5；GitHub pip 安装失败时的最大重试次数
  RETRY_DELAY_SECONDS 默认 30；每次重试前等待秒数

示例:
  bash build_conda_a100_cu126_user.sh
  BASE_DIR=${HOME}/envs/slime_a100_cu126 bash build_conda_a100_cu126_user.sh
  bash build_conda_a100_cu126_user.sh --prepare-only
USAGE
}

RUN_BUILD=1
UPDATE_REPO=0
RUN_VERIFY=1
FORCE_BUILD=0
FORCE_VERIFY=0
RESET_RESUME=0

# 解析命令行选项，默认直接构建，按需只生成脚本或跳过验证。
while [ "$#" -gt 0 ]; do
  case "$1" in
    --prepare-only)
      RUN_BUILD=0
      RUN_VERIFY=0
      shift
      ;;
    --update-repo)
      UPDATE_REPO=1
      shift
      ;;
    --no-verify)
      RUN_VERIFY=0
      shift
      ;;
    --force-build)
      FORCE_BUILD=1
      shift
      ;;
    --force-verify)
      FORCE_VERIFY=1
      shift
      ;;
    --reset-resume)
      RESET_RESUME=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "未知参数：$1"
      ;;
  esac
done

# 检查基础命令是否存在；缺少系统依赖时提前报错，避免构建到一半失败。
for cmd in git curl tar python3; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    die "缺少命令 ${cmd}，请先安装系统基础依赖。"
  fi
done

BASE_DIR_INPUT="${BASE_DIR:-${HOME}/slime_a100_cu126}"
ENV_NAME="${ENV_NAME:-slime-a100-cu126}"
SLIME_A100_REPO="${SLIME_A100_REPO:-https://github.com/jason9693/slime-a100.git}"
SLIME_A100_REF="${SLIME_A100_REF:-main}"
PATCH_VERSION="${PATCH_VERSION:-v0.5.9.a100}"
MAX_JOBS="${MAX_JOBS:-32}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
REUSE_ROOT_EXISTING="${REUSE_ROOT_EXISTING:-1}"
ROOT_BASE_DIR="${ROOT_BASE_DIR:-/root}"
RETRY_ATTEMPTS="${RETRY_ATTEMPTS:-5}"
RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS:-30}"

# 解析 BASE_DIR 的真实路径，并强制默认位于当前用户 HOME 下。
mkdir -p "${BASE_DIR_INPUT}"
BASE_DIR="$(cd "${BASE_DIR_INPUT}" && pwd -P)"
HOME_DIR="$(cd "${HOME}" && pwd -P)"

# 避免误把大型依赖装到系统级目录；如确有需要，可显式设置 ALLOW_OUTSIDE_HOME=1。
case "${BASE_DIR}" in
  "${HOME_DIR}"|"${HOME_DIR}"/*)
    ;;
  *)
    if [ "${ALLOW_OUTSIDE_HOME:-0}" != "1" ]; then
      die "BASE_DIR=${BASE_DIR} 不在当前用户 HOME=${HOME_DIR} 下；如确认要这样做，请设置 ALLOW_OUTSIDE_HOME=1。"
    fi
    ;;
esac

SLIME_DIR="${SLIME_DIR:-${BASE_DIR}/slime-a100}"
if [ -n "${MAMBA_ROOT_PREFIX+x}" ]; then
  MAMBA_ROOT_PREFIX_SET_BY_USER=1
else
  MAMBA_ROOT_PREFIX_SET_BY_USER=0
fi
MAMBA_EXE="${MAMBA_EXE:-${BASE_DIR}/bin/micromamba}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-${BASE_DIR}/micromamba}"
GENERATED_SCRIPT="${GENERATED_SCRIPT:-${SLIME_DIR}/build_conda.a100.cuda126.user.sh}"
ACTIVATE_FILE="${ACTIVATE_FILE:-${BASE_DIR}/activate_slime_a100_cu126.sh}"
BUILD_LOG="${BUILD_LOG:-${BASE_DIR}/build_cuda126.log}"
INSTALL_LOG="${INSTALL_LOG:-${BASE_DIR}/install_a100_cu126_$(date +%Y%m%d_%H%M%S).log}"
RESUME_DIR="${RESUME_DIR:-${BASE_DIR}/.resume_a100_cu126}"
RESUME_ENABLED="${RESUME_ENABLED:-1}"
PIP_GIT_SOURCE_DIR="${PIP_GIT_SOURCE_DIR:-${BASE_DIR}/sources/pip_git}"

export BASE_DIR
export ENV_NAME
export PATCH_VERSION
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS
export MAMBA_ROOT_PREFIX
export MAMBA_EXE
export RESUME_DIR
export RESUME_ENABLED
export PIP_GIT_SOURCE_DIR
export RETRY_ATTEMPTS
export RETRY_DELAY_SECONDS

# 从这里开始记录完整安装过程：既实时打印到终端，也写入总日志文件。
touch "${INSTALL_LOG}" || die "无法写入安装日志：${INSTALL_LOG}"
exec > >(tee -a "${INSTALL_LOG}") 2>&1
trap 'status=$?; if [ "${status}" -ne 0 ]; then printf "[错误] 脚本异常退出，退出码=%s；请查看安装日志：%s\n" "${status}" "${INSTALL_LOG}" >&2; fi' EXIT

# 返回阶段完成标识文件路径；阶段名只在脚本内部生成，避免用户输入污染路径。
stage_marker() {
  local stage="$1"
  printf '%s/%s.done\n' "${RESUME_DIR}" "${stage}"
}

# 判断阶段是否已经完成；只有显式完成标识存在时才允许跳过。
stage_done() {
  local stage="$1"
  [ "${RESUME_ENABLED}" = "1" ] && [ -f "$(stage_marker "${stage}")" ]
}

# 阶段成功后写入完成标识；失败、中断或半成品都不会写这个文件。
mark_stage_done() {
  local stage="$1"
  local marker
  marker="$(stage_marker "${stage}")"
  if [ "${RESUME_ENABLED}" = "1" ]; then
    mkdir -p "${RESUME_DIR}"
    {
      printf 'stage=%s\n' "${stage}"
      printf 'completed_at=%s\n' "$(date +%Y%m%d_%H%M%S)"
      printf 'base_dir=%s\n' "${BASE_DIR}"
      printf 'env_name=%s\n' "${ENV_NAME}"
      printf 'patch_version=%s\n' "${PATCH_VERSION}"
    } > "${marker}"
  fi
}

# 强制重跑某个阶段前清理对应标识，避免旧验证结果覆盖新构建结果。
clear_stage_done() {
  local stage="$1"
  if [ "${RESUME_ENABLED}" = "1" ]; then
    rm -f "$(stage_marker "${stage}")"
  fi
}

# 清空断点标识只允许发生在 BASE_DIR 下，避免误删用户传入的任意目录。
if [ "${RESET_RESUME}" = "1" ]; then
  case "${RESUME_DIR}" in
    "${BASE_DIR}"/*)
      rm -rf "${RESUME_DIR}"
      ;;
    *)
      die "RESUME_DIR=${RESUME_DIR} 不在 BASE_DIR=${BASE_DIR} 下，拒绝执行 --reset-resume。"
      ;;
  esac
fi
mkdir -p "${RESUME_DIR}"

log "BASE_DIR=${BASE_DIR}"
log "SLIME_DIR=${SLIME_DIR}"
log "ENV_NAME=${ENV_NAME}"
log "PATCH_VERSION=${PATCH_VERSION}"
log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
log "REUSE_ROOT_EXISTING=${REUSE_ROOT_EXISTING}"
log "RETRY_ATTEMPTS=${RETRY_ATTEMPTS}"
log "RETRY_DELAY_SECONDS=${RETRY_DELAY_SECONDS}"
log "RESUME_ENABLED=${RESUME_ENABLED}"
log "RESUME_DIR=${RESUME_DIR}"
log "PIP_GIT_SOURCE_DIR=${PIP_GIT_SOURCE_DIR}"
log "INSTALL_LOG=${INSTALL_LOG}"
log "BUILD_LOG=${BUILD_LOG}"

PREPARE_STAGE="prepare_${ENV_NAME}_${PATCH_VERSION}"
BUILD_STAGE="build_${ENV_NAME}_${PATCH_VERSION}"
VERIFY_STAGE="verify_${ENV_NAME}_${PATCH_VERSION}"

# A100 是 SM80；这里给出提醒，不强制检查 GPU，方便在登录节点先生成脚本。
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv || true
else
  log "当前 shell 找不到 nvidia-smi；如果在登录节点生成脚本，可以忽略。"
fi

# 判断目录是否足够“可正常使用”：构建会 checkout、pip install、apply patch，因此需要读写执行权限。
can_reuse_writable_dir() {
  local path="$1"
  [ -d "${path}" ] && [ -r "${path}" ] && [ -w "${path}" ] && [ -x "${path}" ]
}

# 将 /root 下可读写的已有目录软链到当前用户工作区，避免重复 clone 大仓库。
reuse_root_dir_if_available() {
  local name="$1"
  local src="${ROOT_BASE_DIR}/${name}"
  local dst="${BASE_DIR}/${name}"

  if [ "${REUSE_ROOT_EXISTING}" != "1" ]; then
    return 0
  fi
  if [ -e "${dst}" ]; then
    return 0
  fi
  if can_reuse_writable_dir "${src}"; then
    log "复用 ${src}，并映射为 ${dst}"
    ln -s "${src}" "${dst}"
  elif [ -e "${src}" ]; then
    log "发现 ${src}，但当前用户没有完整读写执行权限，跳过复用。"
  fi
}

# 优先选择可执行的已有 micromamba；如果 /root 已有可用二进制，就不再下载到用户目录。
find_reusable_micromamba() {
  local candidate
  if [ "${REUSE_ROOT_EXISTING}" = "1" ]; then
    for candidate in \
      "${ROOT_BASE_DIR}/micromamba/bin/micromamba" \
      "${ROOT_BASE_DIR}/.local/bin/micromamba" \
      "${ROOT_BASE_DIR}/bin/micromamba"; do
      if [ -x "${candidate}" ]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    done
  fi
  if command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
    return 0
  fi
  return 1
}

# 若 /root 下已有可写 micromamba root，就复用其包缓存和环境目录；否则仍落在当前用户目录。
if [ "${REUSE_ROOT_EXISTING}" = "1" ] && [ "${MAMBA_ROOT_PREFIX_SET_BY_USER}" = "0" ]; then
  ROOT_MAMBA_PREFIX="${ROOT_BASE_DIR}/micromamba"
  if can_reuse_writable_dir "${ROOT_MAMBA_PREFIX}"; then
    MAMBA_ROOT_PREFIX="${ROOT_MAMBA_PREFIX}"
    export MAMBA_ROOT_PREFIX
    log "复用已有 MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX}"
  elif [ -e "${ROOT_MAMBA_PREFIX}" ]; then
    log "发现 ${ROOT_MAMBA_PREFIX}，但当前用户没有完整读写执行权限，继续使用 ${MAMBA_ROOT_PREFIX}"
  fi
fi

# 在用户目录安装 micromamba；若系统或 /root 已有可执行版本，则直接复用不重复下载。
if [ -x "${MAMBA_EXE}" ]; then
  log "复用已有 micromamba：${MAMBA_EXE}"
elif REUSABLE_MAMBA="$(find_reusable_micromamba)"; then
  MAMBA_EXE="${REUSABLE_MAMBA}"
  export MAMBA_EXE
  log "复用可执行 micromamba：${MAMBA_EXE}"
else
  log "下载 micromamba 到用户目录。"
  mkdir -p "${BASE_DIR}/bin" "${BASE_DIR}/downloads/micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    -o "${BASE_DIR}/downloads/micromamba/micromamba.tar.bz2"
  tar -xjf "${BASE_DIR}/downloads/micromamba/micromamba.tar.bz2" \
    -C "${BASE_DIR}/downloads/micromamba" bin/micromamba
  install -m 755 "${BASE_DIR}/downloads/micromamba/bin/micromamba" "${MAMBA_EXE}"
fi

export PATH="$(dirname "${MAMBA_EXE}"):${PATH}"

# 先映射可能已经存在的 /root 资源，后续构建脚本看到 BASE_DIR 下已有目录就会直接使用。
reuse_root_dir_if_available "slime-a100"
reuse_root_dir_if_available "sglang"
reuse_root_dir_if_available "Megatron-LM"
reuse_root_dir_if_available "slime"

# 克隆或复用 A100 fork；该 fork 提供 v0.5.9.a100 patch set。
if [ ! -d "${SLIME_DIR}/.git" ]; then
  log "克隆 A100 fork：${SLIME_A100_REPO}"
  git clone "${SLIME_A100_REPO}" "${SLIME_DIR}"
  git -C "${SLIME_DIR}" checkout "${SLIME_A100_REF}"
else
  log "复用已有 A100 fork：${SLIME_DIR}"
  if [ "${UPDATE_REPO}" = "1" ]; then
    log "按要求更新 A100 fork 并切换到 ${SLIME_A100_REF}。"
    git -C "${SLIME_DIR}" fetch --all --tags
    git -C "${SLIME_DIR}" checkout "${SLIME_A100_REF}"
  fi
fi

A100_BUILD_SCRIPT="${SLIME_DIR}/build_conda.a100.sh"
PATCH_DIR="${SLIME_DIR}/docker/patch/${PATCH_VERSION}"

# A100 构建依赖 fork 中的专用脚本和 patch；缺失时直接报错。
if [ ! -f "${A100_BUILD_SCRIPT}" ]; then
  die "未找到 ${A100_BUILD_SCRIPT}；请确认 SLIME_A100_REPO 指向包含 A100 构建脚本的仓库。"
fi

# 缺少 v0.5.9.a100 patch 时不能继续，否则构建出来不是完整 A100 路径。
if [ ! -d "${PATCH_DIR}" ]; then
  die "未找到 ${PATCH_DIR}；A100/CU126 构建需要 PATCH_VERSION=${PATCH_VERSION}。"
fi

cp "${A100_BUILD_SCRIPT}" "${GENERATED_SCRIPT}"

# 将 A100 fork 原脚本改写为 cu126 与当前用户目录版本。
python3 - "${GENERATED_SCRIPT}" <<'PY'
import os
import re
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text()

base_dir = os.environ["BASE_DIR"]
env_name = os.environ["ENV_NAME"]
patch_version = os.environ["PATCH_VERSION"]
mamba_root = os.environ["MAMBA_ROOT_PREFIX"]
mamba_exe = os.environ["MAMBA_EXE"]
resume_dir = os.environ["RESUME_DIR"]
resume_enabled = os.environ["RESUME_ENABLED"]
retry_attempts = os.environ["RETRY_ATTEMPTS"]
retry_delay_seconds = os.environ["RETRY_DELAY_SECONDS"]
pip_git_source_dir = os.environ["PIP_GIT_SOURCE_DIR"]

# 将旧脚本中的系统级用户目录前缀统一改为当前用户的 BASE_DIR。
old_root = "/" + "root"
text = text.replace(old_root, base_dir)

# 将 CUDA 与 PyTorch wheel 源统一改为 CUDA 12.6 / cu126。
replacements = {
    "nvidia/label/cuda-12.9.1": "nvidia/label/cuda-12.6.3",
    "cuda=12.9.1": "cuda=12.6.3",
    "cuda=12.9": "cuda=12.6.3",
    "cuda-nvtx=12.9.79": "cuda-nvtx",
    "cuda-nvtx-dev=12.9.79": "cuda-nvtx-dev",
    "cu129": "cu126",
    "cuda 12.9": "cuda 12.6",
}
for src, dst in replacements.items():
    text = text.replace(src, dst)

# 固定 cuda-python 到 12.6.0，避免 pip 拉到 13.x 或其他 CUDA 版本。
text = re.sub(r"cuda-python==[0-9][^\s\\]*", "cuda-python==12.6.0", text)

# PyTorch cu126 官方组合使用 2.9.1 / 0.24.1 / 2.9.1。
text = re.sub(r"torch==[0-9][^\s\\]*", "torch==2.9.1", text)
text = re.sub(r"torchaudio==[0-9][^\s\\]*", "torchaudio==2.9.1", text)
text = re.sub(r"torchvision==[0-9][^\s\\]*", "torchvision==0.24.1", text)
text = text.replace(
    "torch==2.9.1 torchvision torchaudio==2.9.1",
    "torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1",
)

# 环境名从默认 slime 改为独立名称，避免覆盖已有环境。
text = re.sub(r"micromamba create -n slime\b", f"micromamba create -n {env_name}", text)
text = re.sub(r"micromamba activate slime\b", f"micromamba activate {env_name}", text)
text = re.sub(r"micromamba install -n slime\b", f"micromamba install -n {env_name}", text)
text = text.replace("envs/slime", f"envs/{env_name}")

# 强制 patch 版本使用 A100 patch set。
text = re.sub(
    r'export PATCH_VERSION=.*',
    f'export PATCH_VERSION="${{PATCH_VERSION:-{patch_version}}}"',
    text,
)

# 避免原脚本再次交互式安装 micromamba；外层脚本已经准备了用户目录版本。
text = re.sub(
    r"yes\s+''\s+\|\s+\"\$\{SHELL\}\"\s+<\(curl -L micro\.mamba\.pm/install\.sh\)",
    ': "micromamba 已由外层脚本安装到用户目录"',
    text,
)

# 注入用户目录、micromamba 和 A100 运行环境变量。
marker = "# A100_CU126_USER_INJECTED"
injected = f'''set -ex

{marker}
export BASE_DIR="${{BASE_DIR:-{base_dir}}}"
export PATCH_VERSION="${{PATCH_VERSION:-{patch_version}}}"
export MAMBA_ROOT_PREFIX="${{MAMBA_ROOT_PREFIX:-{mamba_root}}}"
export PATH="{Path(mamba_exe).parent}:${{PATH}}"
eval "$("{mamba_exe}" shell hook -s bash)"
export TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST:-8.0}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${{CUDA_DEVICE_MAX_CONNECTIONS:-1}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export SLIME_FAST_CLEAR_MEMORY="${{SLIME_FAST_CLEAR_MEMORY:-1}}"
export NCCL_NVLS_ENABLE="${{NCCL_NVLS_ENABLE:-0}}"
export MAX_JOBS="${{MAX_JOBS:-32}}"
export RESUME_DIR="${{RESUME_DIR:-{resume_dir}}}"
export RESUME_ENABLED="${{RESUME_ENABLED:-{resume_enabled}}}"
export GENERATED_RESUME_DIR="${{RESUME_DIR}}/generated"
export PIP_GIT_SOURCE_DIR="${{PIP_GIT_SOURCE_DIR:-{pip_git_source_dir}}}"
export RETRY_ATTEMPTS="${{RETRY_ATTEMPTS:-{retry_attempts}}}"
export RETRY_DELAY_SECONDS="${{RETRY_DELAY_SECONDS:-{retry_delay_seconds}}}"
export GIT_CONFIG_COUNT="${{GIT_CONFIG_COUNT:-1}}"
export GIT_CONFIG_KEY_0="${{GIT_CONFIG_KEY_0:-http.version}}"
export GIT_CONFIG_VALUE_0="${{GIT_CONFIG_VALUE_0:-HTTP/1.1}}"
mkdir -p "${{GENERATED_RESUME_DIR}}" "${{PIP_GIT_SOURCE_DIR}}"

# 返回生成脚本内部步骤的完成标识路径。
generated_stage_marker() {{
  local stage="$1"
  printf '%s/%s.done\\n' "${{GENERATED_RESUME_DIR}}" "${{stage}}"
}}

# 成功步骤会写完成标识；重跑时只跳过这些明确完成过的步骤。
mark_generated_stage_done() {{
  local stage="$1"
  if [ "${{RESUME_ENABLED}}" = "1" ]; then
    printf 'stage=%s\\ncompleted_at=%s\\n' "${{stage}}" "$(date +%Y%m%d_%H%M%S)" > "$(generated_stage_marker "${{stage}}")"
  fi
}}

# 执行一个可断点续跑步骤；失败不会写标识，下一次重跑会继续尝试。
run_generated_step() {{
  local stage="$1"
  local status=0
  shift

  # 只有明确完成的步骤才能跳过，目录存在但没有标识不会被误判为成功。
  if [ "${{RESUME_ENABLED}}" = "1" ] && [ -f "$(generated_stage_marker "${{stage}}")" ]; then
    echo "[信息] 跳过已完成步骤：${{stage}}"
    return 0
  fi

  # 命令成功后立即写标识，网络中断或编译失败时保留原始失败码。
  if "$@"; then
    mark_generated_stage_done "${{stage}}"
    return 0
  else
    status="$?"
    return "${{status}}"
  fi
}}

# 对 GitHub 源码安装加重试，降低集群网络/TLS 短断导致整轮构建失败的概率。
retry_command() {{
  local attempt=1
  local status=0

  # 循环执行同一命令；例如 `pip install git+https://github.com/x/y.git@sha --no-deps` 失败后会等待再试。
  while true; do
    "$@" && return 0
    status="$?"

    # 达到最大次数后保留原始退出码，方便日志定位真实失败命令。
    if [ "${{attempt}}" -ge "${{RETRY_ATTEMPTS}}" ]; then
      echo "[错误] 命令重试 ${{RETRY_ATTEMPTS}} 次后仍失败：$*" >&2
      return "${{status}}"
    fi

    echo "[警告] 命令失败，${{RETRY_DELAY_SECONDS}} 秒后重试 $((attempt + 1))/${{RETRY_ATTEMPTS}}：$*" >&2
    sleep "${{RETRY_DELAY_SECONDS}}"
    attempt="$((attempt + 1))"
  done
}}

# 先把 GitHub 源码包拉到持久目录，再从本地目录安装，避免 pip 每次都在 /tmp 重新 clone。
install_github_package() {{
  local git_url="$1"
  local git_ref="$2"
  local local_name="$3"
  local src_dir="${{PIP_GIT_SOURCE_DIR}}/${{local_name}}"
  local tmp_dir="${{src_dir}}.tmp.$$"
  local status=0
  shift 3

  # 已有 git 目录时复用并 fetch；例如 `git+https://github.com/a/b.git@sha` 会复用 `${{PIP_GIT_SOURCE_DIR}}/a__b`。
  if [ -d "${{src_dir}}/.git" ]; then
    echo "[信息] 复用 GitHub 源码目录：${{src_dir}}"
    retry_command git -C "${{src_dir}}" fetch --all --tags || return "$?"
  elif [ -e "${{src_dir}}" ]; then
    echo "[错误] ${{src_dir}} 已存在但不是 git 仓库，请人工检查后再重跑。" >&2
    return 1
  else
    echo "[信息] 克隆 GitHub 源码：${{git_url}} -> ${{src_dir}}"
    retry_command git clone "${{git_url}}" "${{tmp_dir}}" || return "$?"
    mv "${{tmp_dir}}" "${{src_dir}}" || return "$?"
  fi

  # 对固定 commit 或 tag 再单独 fetch 一次；若对象已经存在，则允许继续 checkout。
  if ! retry_command git -C "${{src_dir}}" fetch origin "${{git_ref}}"; then
    git -C "${{src_dir}}" rev-parse --verify "${{git_ref}}^{{commit}}" >/dev/null 2>&1 || return "$?"
  fi
  git -C "${{src_dir}}" checkout -f "${{git_ref}}" || return "$?"

  # 有子模块时一并更新，避免本地安装阶段才发现源码不完整。
  if [ -f "${{src_dir}}/.gitmodules" ]; then
    retry_command git -C "${{src_dir}}" submodule update --init --recursive || return "$?"
  fi

  # 将原始 pip 参数保留下来，只把 git+URL 替换成本地源码目录。
  "$@" "${{src_dir}}"
  status="$?"
  return "${{status}}"
}}
'''
if marker not in text:
    text = text.replace("set -ex", injected, 1)

# 克隆命令改为可重入：已有仓库时复用目录，缺失时才重新拉取。
text = text.replace(
    "git clone https://github.com/sgl-project/sglang.git",
    'if [ -d sglang/.git ]; then echo "[信息] 复用已有 sglang 仓库"; mark_generated_stage_done clone_sglang; else run_generated_step clone_sglang retry_command git clone https://github.com/sgl-project/sglang.git; fi',
)
text = text.replace(
    "git clone https://github.com/deepseek-ai/DeepEP.git",
    'if [ -d DeepEP/.git ]; then echo "[信息] 复用已有 DeepEP 仓库"; mark_generated_stage_done clone_deepep; else run_generated_step clone_deepep retry_command git clone https://github.com/deepseek-ai/DeepEP.git; fi',
)

# Megatron 原脚本把 clone/checkout/install 串在一起；拆开后才能在已有目录上继续执行。
text = re.sub(
    r"git clone https://github\.com/NVIDIA/Megatron-LM\.git --recursive\s*&&\s*\\?\s*cd Megatron-LM/\s*&&\s*git checkout \${MEGATRON_COMMIT}\s*&&\s*\\?\s*pip install -e \.",
    'if [ -d Megatron-LM/.git ]; then echo "[信息] 复用已有 Megatron-LM 仓库"; mark_generated_stage_done clone_megatron; else run_generated_step clone_megatron retry_command git clone https://github.com/NVIDIA/Megatron-LM.git --recursive; fi\ncd Megatron-LM/\ngit checkout -f ${MEGATRON_COMMIT}\ngit submodule update --init --recursive\nrun_generated_step pip_megatron_editable pip install -e .',
    text,
)

# 第三方仓库可能已经打过 patch；重跑时先回到固定 commit，再由后续 patch 步骤重新应用。
text = text.replace("git checkout ${SGLANG_COMMIT}", "git checkout -f ${SGLANG_COMMIT}")
text = text.replace("git checkout ${DEEPEP_COMMIT}", "git checkout -f ${DEEPEP_COMMIT}")
text = text.replace("git checkout ${MEGATRON_COMMIT}", "git checkout -f ${MEGATRON_COMMIT}")

# 用户态安装不能写 /etc/profile.d；运行时统一使用外层生成的 activate 脚本。
text = re.sub(r"SLIME_ENV_SH=/etc/profile\.d/slime_env\.sh", 'SLIME_ENV_SH="${BASE_DIR}/slime_env.sh"', text)
text = re.sub(r'RC_FILES="[^"]*"', 'RC_FILES="${BASE_DIR}/.bashrc"', text)

# 将 GitHub 源码 pip 安装改为“先 clone 到持久目录，再从本地安装”的断点步骤。
def github_pip_stage(command: str) -> str:
    """把 pip GitHub 安装命令转成稳定的阶段名。"""
    key = re.sub(r"[^A-Za-z0-9]+", "_", command).strip("_").lower()
    return "pip_github_" + key[:96]


def parse_github_spec(spec: str) -> tuple[str, str, str]:
    """把 git+https 规格拆成 git URL、ref 和本地目录名。"""
    raw = spec.removeprefix("git+")
    if "@" in raw:
        git_url, git_ref = raw.rsplit("@", 1)
    else:
        git_url, git_ref = raw, "HEAD"
    repo_path = re.sub(r"^https://github\.com/", "", git_url).removesuffix(".git")
    # 将 `https://github.com/fzyzcjy/torch_memory_saver.git@sha` 转成目录名 `fzyzcjy__torch_memory_saver`。
    local_name = re.sub(r"[^A-Za-z0-9._-]+", "_", repo_path.replace("/", "__")).strip("_")
    return git_url, git_ref, local_name


def wrap_github_pip(match: re.Match[str]) -> str:
    """把 `pip install git+https://github.com/...` 改成本地源码安装。"""
    indent = match.group(1)
    command = match.group(2)
    parts = shlex.split(command)
    git_index = next((idx for idx, part in enumerate(parts) if part.startswith("git+https://github.com/")), -1)
    if git_index < 0:
        return match.group(0)
    git_url, git_ref, local_name = parse_github_spec(parts[git_index])
    pip_parts = parts[:git_index] + parts[git_index + 1 :]
    quoted = " ".join(shlex.quote(part) for part in [git_url, git_ref, local_name, *pip_parts])
    return f"{indent}run_generated_step {github_pip_stage(command)} install_github_package {quoted}"


text = re.sub(
    r"(?m)^(\s*)((?:python -m )?pip install [^\n]*git\+https://github\.com/[^\n]+)$",
    wrap_github_pip,
    text,
)

# 常见重型 wheel/源码安装也写步骤标识，避免一次成功后下一次续跑重复构建。
text = text.replace(
    "MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation",
    'run_generated_step pip_flash_attn env MAX_JOBS="${MAX_JOBS:-64}" pip -v install flash-attn==2.7.4.post1 --no-build-isolation',
)
text = text.replace(
    'pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"',
    'run_generated_step pip_transformer_engine pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"',
)
text = text.replace(
    "pip install cmake ninja",
    "run_generated_step pip_cmake_ninja pip install cmake ninja",
)

path.write_text(text)
PY

chmod +x "${GENERATED_SCRIPT}"
log "已生成构建脚本：${GENERATED_SCRIPT}"

# 写出激活脚本，后续训练时只需 source 这一份文件。
cat > "${ACTIVATE_FILE}" <<EOF
#!/usr/bin/env bash
export BASE_DIR="${BASE_DIR}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}"
export PATH="$(dirname "${MAMBA_EXE}"):\${PATH}"
eval "\$("${MAMBA_EXE}" shell hook -s bash)"
micromamba activate "${ENV_NAME}"
export CUDA_HOME="\${CONDA_PREFIX}"
export SLIME_ROOT="${SLIME_DIR}"
export PYTHONPATH="${BASE_DIR}/Megatron-LM:\${PYTHONPATH:-}"
export PATCH_VERSION="${PATCH_VERSION}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SLIME_FAST_CLEAR_MEMORY=1
export NCCL_NVLS_ENABLE=0
EOF
chmod +x "${ACTIVATE_FILE}"
log "已生成激活脚本：${ACTIVATE_FILE}"
mark_stage_done "${PREPARE_STAGE}"

# 默认执行构建；prepare-only 模式只生成脚本，方便用户先审查。
if [ "${RUN_BUILD}" = "1" ]; then
  # 只有构建阶段明确完成过，且用户没有要求强制重跑时，才跳过这段长构建。
  if stage_done "${BUILD_STAGE}" && [ "${FORCE_BUILD}" != "1" ]; then
    log "检测到构建完成标识，跳过构建阶段：$(stage_marker "${BUILD_STAGE}")"
  else
    log "开始执行 A100/CU126 构建，日志追加写入 ${BUILD_LOG}"
    clear_stage_done "${BUILD_STAGE}"
    clear_stage_done "${VERIFY_STAGE}"
    {
      printf '\n[信息] ===== 构建阶段开始 %s =====\n' "$(date +%Y%m%d_%H%M%S)"
      (
        cd "${SLIME_DIR}"
        BASE_DIR="${BASE_DIR}" \
        PATCH_VERSION="${PATCH_VERSION}" \
        MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
        CUDA_DEVICE_MAX_CONNECTIONS=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        SLIME_FAST_CLEAR_MEMORY=1 \
        NCCL_NVLS_ENABLE=0 \
        MAX_JOBS="${MAX_JOBS}" \
        RESUME_DIR="${RESUME_DIR}" \
        RESUME_ENABLED="${RESUME_ENABLED}" \
        RETRY_ATTEMPTS="${RETRY_ATTEMPTS}" \
        RETRY_DELAY_SECONDS="${RETRY_DELAY_SECONDS}" \
        bash "${GENERATED_SCRIPT}"
      )
      printf '[信息] ===== 构建阶段完成 %s =====\n' "$(date +%Y%m%d_%H%M%S)"
    } 2>&1 | tee -a "${BUILD_LOG}"
    mark_stage_done "${BUILD_STAGE}"
  fi
else
  log "已按 --prepare-only 跳过实际构建。"
fi

# 构建完成后做完整导入验证，确认 Python 侧看到的是 cu126，并且运行在 A100/SM80 上。
if [ "${RUN_VERIFY}" = "1" ]; then
  # 验证成功也写独立标识；已经通过的环境重复运行脚本时不用再枚举 GPU 和导入包。
  if stage_done "${VERIFY_STAGE}" && [ "${FORCE_VERIFY}" != "1" ]; then
    log "检测到验证完成标识，跳过验证阶段：$(stage_marker "${VERIFY_STAGE}")"
  else
    log "开始验证 A100/CU126 环境。"
    clear_stage_done "${VERIFY_STAGE}"
    # shellcheck disable=SC1090
    source "${ACTIVATE_FILE}"
    log "验证使用 python=$(command -v python)"
    python -V
    log "CUDA_HOME=${CUDA_HOME:-}"
    log "SLIME_ROOT=${SLIME_ROOT:-}"
    log "PYTHONPATH=${PYTHONPATH:-}"
    log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-}"
    log "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-}"
    log "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}"
    log "SLIME_FAST_CLEAR_MEMORY=${SLIME_FAST_CLEAR_MEMORY:-}"
    log "NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-}"
    python - <<'PY'
import importlib
import os
import sys

import torch

errors: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"[通过] {message}")
    else:
        print(f"[失败] {message}")
        errors.append(message)


print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
check(torch.version.cuda is not None and torch.version.cuda.startswith("12.6"), "PyTorch 使用 CUDA 12.6")
check(torch.cuda.is_available(), "PyTorch 可以访问 CUDA GPU")

has_a100 = False
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("gpu count:", device_count)
    for idx in range(device_count):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        print(f"gpu[{idx}]: {name}, capability={capability}")
        if "A100" in name and capability == (8, 0):
            has_a100 = True
check(has_a100, "至少检测到一张 NVIDIA A100 / SM80 GPU")

expected_env = {
    "TORCH_CUDA_ARCH_LIST": "8.0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "SLIME_FAST_CLEAR_MEMORY": "1",
    "NCCL_NVLS_ENABLE": "0",
}
for key, expected in expected_env.items():
    actual = os.environ.get(key)
    print(f"{key}={actual}")
    check(actual == expected, f"{key}={expected}")

for module_name in ("slime", "sglang", "transformer_engine", "deep_ep", "megatron"):
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[通过] {module_name} 导入成功，version={version}")
    except Exception as exc:
        print(f"[失败] {module_name} 导入失败: {exc!r}")
        errors.append(f"{module_name} 导入失败")

try:
    router = importlib.import_module("sglang_router")
    router_version = getattr(router, "__version__", "unknown")
    print(f"sglang_router version={router_version}")
    check("slime" in str(router_version), "sglang_router 版本包含 slime 标识")
except Exception as exc:
    print(f"[失败] sglang_router 导入失败: {exc!r}")
    errors.append("sglang_router 导入失败")

if errors:
    print("\n验证失败项:")
    for item in errors:
        print(f"- {item}")
    sys.exit(1)

print("\nA100/CU126 环境验证全部通过。")
PY
    mark_stage_done "${VERIFY_STAGE}"
  fi
else
  log "已跳过验证。"
fi

log "完成。后续使用环境前执行：source ${ACTIVATE_FILE}"
log "总安装日志：${INSTALL_LOG}"
log "构建子日志：${BUILD_LOG}"
