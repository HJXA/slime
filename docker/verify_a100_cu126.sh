#!/usr/bin/env bash

set -Eeuo pipefail

BASE_DIR="${BASE_DIR:-/home/slime/slime_a100_cu126}"
ACTIVATE_FILE="${ACTIVATE_FILE:-${BASE_DIR}/activate_slime_a100_cu126.sh}"

# 容器运行时必须先激活构建脚本生成的环境。
if [ ! -f "${ACTIVATE_FILE}" ]; then
  echo "[错误] 找不到激活脚本：${ACTIVATE_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ACTIVATE_FILE}"

echo "[信息] python=$(command -v python)"
python -V
echo "[信息] CUDA_HOME=${CUDA_HOME:-}"
echo "[信息] SLIME_ROOT=${SLIME_ROOT:-}"
echo "[信息] PYTHONPATH=${PYTHONPATH:-}"

python - <<'PY'
import importlib
import os
import sys

import torch

errors: list[str] = []


def check(condition: bool, message: str) -> None:
    """记录单项验证结果；失败项会累计到最后统一退出。"""
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
# 只有容器能访问 CUDA 时才枚举 GPU，避免无 GPU 构建节点产生额外异常。
if torch.cuda.is_available():
    # 逐张检查可见 GPU；A100 的计算能力是 SM80，对应 capability=(8, 0)。
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        capability = torch.cuda.get_device_capability(idx)
        print(f"gpu[{idx}]: {name}, capability={capability}")
        # 同时检查名称和算力，避免把非 A100 的 SM80 兼容设备误认为目标环境。
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
# 逐项确认关键运行变量，保证容器运行时和从 0 部署脚本的默认约束一致。
for key, expected in expected_env.items():
    actual = os.environ.get(key)
    print(f"{key}={actual}")
    check(actual == expected, f"{key}={expected}")

# 逐个导入核心包；这些包覆盖 slime 训练、SGLang 后端、TE、DeepEP 和 Megatron。
for module_name in ("slime", "sglang", "transformer_engine", "deep_ep", "megatron"):
    # 导入失败通常说明构建脚本、patch 或 PYTHONPATH 没有正确落盘。
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[通过] {module_name} 导入成功，version={version}")
    except Exception as exc:
        print(f"[失败] {module_name} 导入失败: {exc!r}")
        errors.append(f"{module_name} 导入失败")

# sglang_router 是独立二进制扩展，单独检查 slime 定制版本标识。
try:
    router = importlib.import_module("sglang_router")
    router_version = getattr(router, "__version__", "unknown")
    print(f"sglang_router version={router_version}")
    check("slime" in str(router_version), "sglang_router 版本包含 slime 标识")
except Exception as exc:
    print(f"[失败] sglang_router 导入失败: {exc!r}")
    errors.append("sglang_router 导入失败")

# 任何失败项都让容器验证命令返回非零，方便调度系统或 CI 捕获。
if errors:
    print("\n验证失败项:")
    for item in errors:
        print(f"- {item}")
    sys.exit(1)

print("\nA100/CU126 Docker 环境验证全部通过。")
PY
