# Qwen2.5 数学 GRPO 最小教程

这份教程用于从零跑通一个最小数学领域 GRPO 链路：使用 `Qwen2.5-0.5B-Instruct` 作为 actor 和 rollout 模型，使用公开数学数据集作为 prompt 数据，使用 slime 内置 `math` rule-based reward 做答案校验，然后通过 GRPO 更新模型。

这里的“最小”指的是先把 **模型下载、数据读取、reward 校验、rollout、训练、评估、checkpoint** 串起来。第一次建议只跑几轮，不追求效果；等链路稳定后，再放大模型、数据量、rollout 长度、训练轮数和 GPU 资源。

## 目录

```text
docs/zh/get_started/qwen2.5_math_grpo.md  # 本教程，说明 Qwen2.5 数学 GRPO 最小链路
docs/zh/get_started/quick_start.md        # slime 通用快速开始，包含完整参数块说明
docs/zh/get_started/usage.md              # slime 训练、rollout、数据格式和 GRPO 参数说明
docs/zh/get_started/customization.md      # custom reward、custom rollout 等扩展接口说明
.claude/skills/                           # slime 仓库默认技能，指导常见扩展和审核任务
build_conda.sh                            # 从 0 部署 conda/micromamba 环境、SGLang、Megatron 和 slime
docker/                                   # Docker 镜像构建、补丁和容器环境相关文件
scripts/models/qwen2.5-0.5B.sh            # Qwen2.5-0.5B 的 Megatron 模型结构参数
scripts/models/qwen2.5-1.5B.sh            # Qwen2.5-1.5B 的 Megatron 模型结构参数
scripts/models/qwen2.5-3B.sh              # Qwen2.5-3B 的 Megatron 模型结构参数
scripts/models/qwen2.5-7B.sh              # Qwen2.5-7B 的 Megatron 模型结构参数
slime/rollout/rm_hub/__init__.py          # `--rm-type math` 等 reward 分发入口
slime/rollout/rm_hub/math_utils.py        # boxed 数学答案抽取与等价性校验逻辑
train.py                                  # 同步 GRPO 训练入口
train_async.py                            # fully-async 训练入口，本教程先不使用
```

## 目标链路

本教程最终要跑通下面这条最短闭环：

```text
数学 prompt 数据
  -> Qwen2.5 rollout 生成多个答案
  -> `math` reward 检查答案是否匹配 label
  -> GRPO 根据同一 prompt 的多条 response 做组内相对优势
  -> Megatron 训练 actor
  -> slime 将新权重同步给 SGLang rollout
  -> 进入下一轮 rollout
```

先用 `Qwen2.5-0.5B-Instruct` 的原因是它最容易跑通工程链路。确认无误后，切到 `Qwen2.5-1.5B`、`Qwen2.5-3B` 或 `Qwen2.5-7B` 只需要换模型路径、模型参数脚本和 GPU 并行配置。

## 第 0 步：确认机器与环境

推荐先在 Linux GPU 服务器或 Docker 容器中运行，不建议在 Mac 上实际执行 CUDA 训练。Mac 上只做脚本和参数审查即可。

### 0.1 Docker 快速环境

最省心的方式是使用 slime 官方 Docker 镜像：

```bash
docker pull slimerl/slime:latest

docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

进入容器后，确认 slime、Megatron 和 GPU 可用：

```bash
cd /root/slime
pip install -e . --no-deps

python -c "import slime; print('slime 可以导入')"
nvidia-smi
ls /root/Megatron-LM
```

如果 `ls /root/Megatron-LM` 不存在，需要先按当前环境的方式安装或挂载 Megatron-LM，因为 slime 的默认训练后端会通过 `PYTHONPATH=/root/Megatron-LM` 读取 Megatron 参数和训练实现。

### 0.2 从 0 开始部署裸机 conda 环境

如果不能使用 Docker，可以在一台干净的 Linux GPU 机器上用仓库自带的 `build_conda.sh` 从 0 部署。这个脚本会安装 micromamba 环境、CUDA 12.9 运行库、SGLang、Megatron-LM、slime 依赖、SGLang router、Apex、Transformer Engine、flash-attn、mbridge，并应用 slime 对 SGLang 和 Megatron 的补丁。

这条路径比 Docker 慢，也更依赖网络和编译环境。建议先在新机器、独立账号或一次性容器里做，避免污染已有 Python/CUDA 环境。

#### 0.2.1 检查系统前置条件

先确认 GPU 驱动、编译工具和网络都可用：

```bash
nvidia-smi
which git
which curl
which gcc || true
which g++ || true
```

如果是 Ubuntu/Debian 系统，可以先安装常用系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl wget ca-certificates \
  build-essential cmake ninja-build pkg-config \
  libaio-dev numactl
```

其中：

| 依赖 | 用途 |
| --- | --- |
| NVIDIA 驱动 | 让 PyTorch、SGLang、Megatron 能看到 GPU。 |
| `git` / `curl` / `wget` | 下载 slime、SGLang、Megatron 和 Python 包。 |
| `build-essential` / `cmake` / `ninja-build` | 编译 Apex、flash-attn、torch_memory_saver 等 native 扩展。 |
| `libaio-dev` / `numactl` | 部分训练和推理依赖会用到的系统库或调度工具。 |

#### 0.2.2 拉取 slime 仓库

选择一个干净的根目录。下面以 `/root` 为例；如果你在普通用户目录部署，可以把 `BASE_DIR` 换成自己的路径，但要注意 `build_conda.sh` 当前更贴近 `/root` 或容器环境的使用方式。

```bash
export BASE_DIR=/root
mkdir -p ${BASE_DIR}
cd ${BASE_DIR}

git clone https://github.com/THUDM/slime.git
cd ${BASE_DIR}/slime
```

如果你已经在自己的 slime 仓库中，只需要进入仓库根目录：

```bash
cd /path/to/slime
export BASE_DIR=/root
```

#### 0.2.3 执行一键环境安装

运行：

```bash
cd ${BASE_DIR}/slime
BASE_DIR=${BASE_DIR} bash build_conda.sh
```

脚本主要会做这些事：

```text
1. 安装 micromamba，并创建名为 slime 的 Python 3.12 环境。
2. 安装 CUDA 12.9、cuDNN、NCCL、Rust 等基础依赖。
3. 克隆并安装固定 commit 的 SGLang。
4. 安装 PyTorch cu129、sglang-kernel、sgl-deep-gemm 和 SGLang router。
5. 编译安装 flash-attn、Apex、Transformer Engine、torch_memory_saver 等训练依赖。
6. 克隆并安装固定 commit 的 Megatron-LM。
7. 安装 slime 的 requirements，并以 editable 方式安装 slime。
8. 编译 slime 内部 int4_qat kernel。
9. 对 SGLang 和 Megatron 应用 `docker/patch/latest/` 下的补丁。
```

整个过程可能需要几十分钟到数小时，取决于网络、CPU 编译速度和 pip/conda 镜像速度。如果中途因为网络失败退出，通常可以重新执行同一条命令；脚本会复用已经克隆的目录。

#### 0.2.4 激活环境并验证

安装完成后，重新打开 shell，或手动加载 micromamba：

```bash
source ~/.bashrc
micromamba activate slime
```

然后验证关键组件：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import slime; print('slime 可以导入')"
python -c "import sglang; print('sglang 可以导入')"
python -c "import sglang_router; print(sglang_router.__version__)"

test -d ${BASE_DIR}/Megatron-LM
PYTHONPATH=${BASE_DIR}/Megatron-LM python -c "import megatron; print('Megatron 可以导入')"
```

期望结果：

```text
torch.cuda.is_available() 返回 True
slime 可以导入
sglang 可以导入
sglang_router 版本字符串里包含 slime
Megatron 可以导入
```

#### 0.2.5 设置本教程需要的默认路径

如果你按 `/root` 部署，后面的命令可以继续使用：

```bash
export SLIME_ROOT=/root/slime
export PYTHONPATH=/root/Megatron-LM
```

如果你把 `BASE_DIR` 改成了别的位置，则同步改成：

```bash
export SLIME_ROOT=${BASE_DIR}/slime
export PYTHONPATH=${BASE_DIR}/Megatron-LM
```

后文的 `hf download`、checkpoint 转换和训练脚本都默认使用这两个路径。只要 `SLIME_ROOT` 和 `PYTHONPATH` 对应正确，Docker 路径和裸机路径在训练命令层面没有本质区别。

#### 0.2.6 裸机部署常见问题

| 问题 | 处理方式 |
| --- | --- |
| `torch.cuda.is_available()` 是 `False` | 先看 `nvidia-smi` 是否正常；如果驱动或 CUDA runtime 不匹配，优先换 Docker，或按机器驱动重新对齐 CUDA/PyTorch 版本。 |
| `flash-attn`、`Apex` 或 `torch_memory_saver` 编译失败 | 检查 `gcc/g++/ninja/cmake` 是否存在，确认当前 shell 已激活 `slime` 环境，并保留完整编译日志。 |
| `git clone` 或 pip 下载很慢 | 配置公司或学校内部镜像；不要手动改 slime 代码里的版本号，先只替换下载源。 |
| SGLang 或 Megatron patch 冲突 | 确认 `build_conda.sh` 中的 `SGLANG_COMMIT`、`MEGATRON_COMMIT`、`PATCH_VERSION` 没有被随意改动。 |
| 普通用户没有 `/root` 权限 | 使用自己有权限的 `BASE_DIR`，并检查脚本里少量 `/root` 假设；最稳妥的方式仍是在容器或专用账号中部署。 |

## 第 1 步：确定本次最小实验配置

先固定这些变量，后面的命令都复用它们：

```bash
export SLIME_ROOT=/root/slime
export MODEL_NAME=Qwen2.5-0.5B-Instruct
export MODEL_DIR=/root/models/${MODEL_NAME}
export TRAIN_DATA=/root/datasets/dapo-math-17k/dapo-math-17k.tiny.jsonl
export EVAL_DATA=/root/datasets/aime-2024/aime-2024.tiny.jsonl
export SAVE_DIR=/root/outputs/qwen2.5-0.5b-math-grpo/slime_ckpt
export NUM_GPUS=2
```

各变量的含义如下：

| 变量 | 含义 |
| --- | --- |
| `SLIME_ROOT` | slime 仓库路径。 |
| `MODEL_NAME` | Hugging Face 上的 Qwen2.5 模型名。 |
| `MODEL_DIR` | Hugging Face 格式模型保存路径，也是 SGLang rollout 加载 tokenizer 和 config 的路径。 |
| `TRAIN_DATA` | 训练 prompt 数据路径。第一次只取少量样本做 smoke run。 |
| `EVAL_DATA` | 评估 prompt 数据路径。第一次只取少量样本做 smoke run。 |
| `SAVE_DIR` | Megatron actor checkpoint 保存路径。 |
| `NUM_GPUS` | 本教程最小脚本默认使用 2 张 GPU：1 张 actor，1 张 rollout。 |

如果只有 1 张 GPU，也可以尝试 `--colocate` 训推一体，但第一次教程更推荐 2 张 GPU 分离 actor 和 rollout，问题更少、日志更清楚。

## 第 2 步：下载 Qwen2.5 模型

下载 Hugging Face 格式模型：

```bash
mkdir -p /root/models

hf download Qwen/${MODEL_NAME} \
  --local-dir ${MODEL_DIR}
```

检查关键文件是否存在：

```bash
ls ${MODEL_DIR}
test -f ${MODEL_DIR}/config.json
test -f ${MODEL_DIR}/tokenizer_config.json
```

这里的 Hugging Face 模型会被用于两件事：

1. SGLang rollout 阶段读取 tokenizer、chat template、模型结构配置和初始权重。
2. Megatron checkpoint 转换时作为源权重。

## 第 3 步：下载数学数据

本教程使用 `dapo-math-17k` 训练，用 `aime-2024` 做最小评估：

```bash
mkdir -p /root/datasets/dapo-math-17k /root/datasets/aime-2024

hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/datasets/dapo-math-17k

hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/datasets/aime-2024
```

第一次只取很少的数据，避免一上来就跑长任务：

```bash
head -n 64 /root/datasets/dapo-math-17k/dapo-math-17k.jsonl > ${TRAIN_DATA}
head -n 16 /root/datasets/aime-2024/aime-2024.jsonl > ${EVAL_DATA}
```

查看一条训练样本：

```bash
python - <<'PY'
import json
import os

path = os.environ["TRAIN_DATA"]
with open(path, "r", encoding="utf-8") as f:
    row = json.loads(next(f))

print(row.keys())
print(json.dumps(row, ensure_ascii=False, indent=2)[:1200])
PY
```

本教程后面的训练参数会使用：

```bash
--prompt-data ${TRAIN_DATA}
--input-key prompt
--label-key label
--apply-chat-template
```

这表示每条样本里：

| 字段 | 作用 |
| --- | --- |
| `prompt` | 输入给模型的问题，通常是 OpenAI messages 形式。 |
| `label` | 标准答案，用于 reward 校验。 |
| `--apply-chat-template` | 让 tokenizer 按 Qwen2.5 的 chat template 把 messages 转成模型输入。 |

## 第 4 步：确认 reward 规则

最小数学 GRPO 不需要额外训练 reward model，直接用：

```bash
--rm-type math
```

当前 `math` reward 的核心规则是：

1. 从模型 response 的最后一个 `\boxed{...}` 中抽取答案。
2. 如果 `label` 自身包含 `\boxed{...}`，也会先抽取其中答案。
3. 用数学归一化和 SymPy 等价性判断比较二者。
4. 正确返回 `1`，错误返回 `0`。

因此 prompt 必须引导模型把最终答案写成 `\boxed{...}`。如果模型只输出普通文本 `答案是 42`，`math` reward 会抽不到 boxed 答案，奖励就是 `0`。

可以先用一个极小脚本确认 reward 行为：

```bash
cd ${SLIME_ROOT}

PYTHONDONTWRITEBYTECODE=1 python - <<'PY'
import asyncio
from types import SimpleNamespace

from slime.rollout.rm_hub import async_rm
from slime.utils.types import Sample

args = SimpleNamespace(custom_rm_path=None, rm_type="math", rm_url=None)

right = Sample(response=r"推理略。最终答案为 \boxed{42}", label="42")
wrong = Sample(response=r"推理略。最终答案为 \boxed{41}", label="42")
missing_box = Sample(response="推理略。最终答案为 42", label="42")

print("boxed 正确答案 reward =", asyncio.run(async_rm(args, right)))
print("boxed 错误答案 reward =", asyncio.run(async_rm(args, wrong)))
print("没有 boxed 的答案 reward =", asyncio.run(async_rm(args, missing_box)))
PY
```

期望输出是：

```text
boxed 正确答案 reward = 1
boxed 错误答案 reward = 0
没有 boxed 的答案 reward = 0
```

如果你后续换成自己的数学数据，最重要的是保证每条数据有 `prompt` 和 `label`，并且 prompt 明确要求模型把最终答案写在 `\boxed{...}` 里。

## 第 5 步：把 Hugging Face 权重转换成 Megatron checkpoint

同步训练入口 `train.py` 使用 Megatron 训练后端。Megatron 不直接读取 Hugging Face checkpoint，因此需要先转换为 `torch_dist` 格式。

进入 slime 仓库，加载 Qwen2.5-0.5B 的模型结构参数：

```bash
cd ${SLIME_ROOT}
source scripts/models/qwen2.5-0.5B.sh
```

执行转换：

```bash
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint ${MODEL_DIR} \
  --save ${MODEL_DIR}_torch_dist
```

检查转换结果：

```bash
ls ${MODEL_DIR}_torch_dist
test -f ${MODEL_DIR}_torch_dist/latest_checkpointed_iteration.txt
```

本教程后面的 checkpoint 参数会这样写：

```bash
--hf-checkpoint ${MODEL_DIR}
--ref-load ${MODEL_DIR}_torch_dist
--load ${SAVE_DIR}
--save ${SAVE_DIR}
```

这几个路径不要混淆：

| 参数 | 作用 |
| --- | --- |
| `--hf-checkpoint` | 给 SGLang 和 tokenizer 用的 Hugging Face 模型目录。 |
| `--ref-load` | 初始 reference model 的 Megatron checkpoint。 |
| `--load` | actor 续训读取路径；第一次没有 checkpoint 时会从 `--ref-load` 初始化。 |
| `--save` | actor 训练后保存路径。 |

## 第 6 步：编写最小启动脚本

下面是最小同步 GRPO 脚本。它使用 2 张 GPU：1 张给 Megatron actor，1 张给 SGLang rollout。

```bash
cd ${SLIME_ROOT}

cat > /root/run_qwen25_math_grpo_minimal.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
set -x

# 这些路径可以在启动脚本前通过环境变量覆盖，方便在不同机器上复用同一份脚本。
SLIME_ROOT=${SLIME_ROOT:-/root/slime}
MODEL_NAME=${MODEL_NAME:-Qwen2.5-0.5B-Instruct}
MODEL_DIR=${MODEL_DIR:-/root/models/${MODEL_NAME}}
TRAIN_DATA=${TRAIN_DATA:-/root/datasets/dapo-math-17k/dapo-math-17k.tiny.jsonl}
EVAL_DATA=${EVAL_DATA:-/root/datasets/aime-2024/aime-2024.tiny.jsonl}
SAVE_DIR=${SAVE_DIR:-/root/outputs/qwen2.5-0.5b-math-grpo/slime_ckpt}
NUM_GPUS=${NUM_GPUS:-2}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

cd "${SLIME_ROOT}"

# Qwen2.5-0.5B 的 Megatron 模型结构参数；换模型规模时优先只改这一行和 MODEL_NAME。
source scripts/models/qwen2.5-0.5B.sh

# NVLink 开关会影响 NCCL 行为；没有 NVLink 时设为 0，避免错误启用。
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [ "${NVLINK_COUNT}" -gt 0 ]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi

mkdir -p "${SAVE_DIR}"

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${MODEL_DIR}_torch_dist"
  --load "${SAVE_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 1
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_DATA}"
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle

  --rm-type math

  --num-rollout 3
  --rollout-batch-size 4
  --n-samples-per-prompt 4
  --rollout-max-response-len 1024
  --rollout-temperature 0.8

  --global-batch-size 16
  --balance-data
)

EVAL_ARGS=(
  --eval-interval 1
  --eval-prompt-data aime "${EVAL_DATA}"
  --n-samples-per-eval-prompt 1
  --eval-max-response-len 1024
  --eval-top-k 1
)

PERF_ARGS=(
  --tensor-model-parallel-size 1
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --use-kl-loss
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.7
  --sglang-cuda-graph-max-bs 16
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

# 如果当前机器上已经有自己的 Ray 集群，不要重复启动；本教程默认是独占调试容器。
ray start --head \
  --node-ip-address "${MASTER_ADDR}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 1 \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"
BASH

chmod +x /root/run_qwen25_math_grpo_minimal.sh
```

脚本只跑 `--num-rollout 3`，每轮取 `4` 个 prompt，每个 prompt 采样 `4` 条 response，总共只做很小的 smoke run。它的目的不是得到好模型，而是确认整个工程链路可以跑通。

## 第 7 步：理解最关键的 batch 关系

GRPO 需要同一个 prompt 采样多条 response 做组内比较，因此这里设置：

```bash
--rollout-batch-size 4
--n-samples-per-prompt 4
```

这表示每轮 rollout 会产生：

```text
4 个 prompt * 每个 prompt 4 条 response = 16 条训练样本
```

训练侧设置：

```bash
--global-batch-size 16
```

因此每轮 rollout 产出的 16 条样本刚好被一个 optimizer step 消耗。最简单时保持这个等式即可：

```text
rollout-batch-size * n-samples-per-prompt = global-batch-size * num-steps-per-rollout
```

本脚本没有显式写 `--num-steps-per-rollout`，默认按 slime 当前逻辑使用 1 步。如果你显式设置它，就要重新检查上面的等式。

## 第 8 步：启动训练

启动前确认数据和 checkpoint 都存在：

```bash
test -f ${TRAIN_DATA}
test -f ${EVAL_DATA}
test -f ${MODEL_DIR}/config.json
test -f ${MODEL_DIR}_torch_dist/latest_checkpointed_iteration.txt
```

运行最小脚本：

```bash
bash /root/run_qwen25_math_grpo_minimal.sh
```

正常情况下，你会看到这些阶段陆续出现：

```text
Ray 集群启动
SGLang rollout engine 启动
Megatron actor 初始化
读取 prompt 数据
rollout 生成 response
math reward 打分
GRPO 计算 advantage
训练一步 actor
同步权重到 rollout engine
执行 eval rollout
保存 checkpoint
```

如果只想确认 rollout 数据生成和 reward 是否正常，可以先把脚本中的 `ray job submit` 后面的训练参数临时加上：

```bash
--debug-rollout-only
--save-debug-rollout-data /root/outputs/qwen2.5-0.5b-math-grpo/debug_rollout_{rollout_id}.pt
```

这样会只生成 rollout 数据，不训练。确认生成样本和 reward 正常后，再去掉这两个参数跑完整训练。

## 第 9 步：检查结果

训练结束后先看 checkpoint：

```bash
ls ${SAVE_DIR}
cat ${SAVE_DIR}/latest_checkpointed_iteration.txt
```

如果 `latest_checkpointed_iteration.txt` 存在，并且 `iter_...` 目录存在，说明 Megatron checkpoint 已经写出。

再看 Ray job 输出中的几个指标：

| 指标或日志 | 应该怎么看 |
| --- | --- |
| `reward` / `raw_reward` | smoke run 中可能很低，甚至经常为 0；只要 reward 能正常计算，先不用追求数值。 |
| `response_length` | 如果大量等于 `rollout-max-response-len`，说明输出被截断，需要增大长度或改善 prompt。 |
| `ppo_kl` / KL 类指标 | `kl-loss-coef=0` 时主要用于观察，不参与损失。 |
| `entropy` | 用于观察采样分布，第一次跑通不必调。 |
| eval reward | 样本很少时波动很大，只用于确认评估链路能跑。 |

最小 run 的判断标准是：

1. 能完成 3 轮 rollout。
2. 能完成至少 3 个训练 step。
3. 能保存 checkpoint。
4. reward 不是程序异常导致的全空值。
5. 评估阶段能正常读 `EVAL_DATA` 并输出结果。

## 第 10 步：把 smoke run 放大一点

确认最小脚本跑通后，可以按下面顺序逐步放大，不要一次改太多：

### 10.1 增加训练轮数

先改：

```bash
--num-rollout 20
--save-interval 10
```

如果 20 轮稳定，再考虑 100、500 或更多。

### 10.2 增加训练数据

把：

```bash
TRAIN_DATA=/root/datasets/dapo-math-17k/dapo-math-17k.tiny.jsonl
```

换成完整数据：

```bash
TRAIN_DATA=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
```

### 10.3 增大 response 长度

数学题需要推理时，`1024` 可能偏短。可以逐步改成：

```bash
--rollout-max-response-len 2048
--eval-max-response-len 2048
```

如果要跑更难数学题，再考虑 `4096`、`8192`。长度增大后显存和 rollout 时间都会明显增加。

### 10.4 增加每个 prompt 的采样数

GRPO 依赖组内 reward 差异。可以从：

```bash
--n-samples-per-prompt 4
```

逐步改到：

```bash
--n-samples-per-prompt 8
```

同时保持 batch 等式。例如：

```bash
--rollout-batch-size 4
--n-samples-per-prompt 8
--global-batch-size 32
```

### 10.5 开启动态采样过滤

如果很多 prompt 的 4 条或 8 条 response reward 完全一样，GRPO 的组内比较信息会变少。可以加入：

```bash
--over-sampling-batch-size 8
--dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
```

含义是多采一些 prompt，只保留组内 reward 标准差非零的 prompt 组。第一次最小 run 不强制打开，是为了让链路更容易定位问题。

## 第 11 步：换成更大的 Qwen2.5

换模型时按这张表改：

| 目标模型 | Hugging Face 模型名 | 模型参数脚本 |
| --- | --- | --- |
| 0.5B | `Qwen2.5-0.5B-Instruct` | `scripts/models/qwen2.5-0.5B.sh` |
| 1.5B | `Qwen2.5-1.5B-Instruct` | `scripts/models/qwen2.5-1.5B.sh` |
| 3B | `Qwen2.5-3B-Instruct` | `scripts/models/qwen2.5-3B.sh` |
| 7B | `Qwen2.5-7B-Instruct` | `scripts/models/qwen2.5-7B.sh` |

具体步骤：

1. 修改 `MODEL_NAME`。
2. 下载新的 Hugging Face 模型。
3. `source` 对应的 `scripts/models/qwen2.5-*.sh`。
4. 重新执行 `convert_hf_to_torch_dist.py`。
5. 增加 actor 和 rollout GPU。
6. 按显存情况调整 `--tensor-model-parallel-size`、`--context-parallel-size`、`--rollout-num-gpus-per-engine` 和 `--max-tokens-per-gpu`。

例如切到 7B 时，常见方向是：

```bash
export MODEL_NAME=Qwen2.5-7B-Instruct
export MODEL_DIR=/root/models/${MODEL_NAME}

# 脚本里改为：
source scripts/models/qwen2.5-7B.sh

# 根据机器显存改并行参数，例如先从 4 张 actor + 4 张 rollout 这类配置开始。
--actor-num-gpus-per-node 4
--rollout-num-gpus 4
--tensor-model-parallel-size 2
--context-parallel-size 2
--rollout-num-gpus-per-engine 2
```

具体并行并没有唯一答案，需要按 GPU 显存、上下文长度、batch size 和吞吐目标调。

## 第 12 步：常见问题排查

### reward 一直是 0

优先检查模型 response 是否包含 `\boxed{...}`。`--rm-type math` 不会从普通自然语言答案里猜最终答案。

可以临时降低任务难度，或者改 prompt，让最后一句固定为：

```text
最后请把最终答案写成 \boxed{答案}。
```

也可以抽几条 rollout response 人工看：

```bash
python - <<'PY'
import torch

path = "/root/outputs/qwen2.5-0.5b-math-grpo/debug_rollout_0.pt"
data = torch.load(path, map_location="cpu")
print(type(data))
print(data)
PY
```

### SGLang OOM

先降低这些参数：

```bash
--rollout-max-response-len 512
--sglang-mem-fraction-static 0.6
--sglang-cuda-graph-max-bs 8
```

如果还是 OOM，降低 `--rollout-batch-size`，并同步降低或重算 `--global-batch-size`。

### Megatron OOM

先降低：

```bash
--max-tokens-per-gpu 2048
```

如果模型更大，增加 `--tensor-model-parallel-size` 或 `--context-parallel-size`，同时确保 actor GPU 数能被这些并行维度整除。

### Ray 提示端口或集群已经存在

如果这是独占调试容器，可以先清理当前用户自己的 Ray：

```bash
ray stop --force
```

如果是共享服务器，不要随意杀别人的 Ray 或 Python 进程。改用已有 Ray 集群，或者换端口启动自己的 Ray。

### checkpoint 没有保存

检查：

1. `--save ${SAVE_DIR}` 是否有写权限。
2. `--save-interval` 是否大于实际训练 step 数。
3. 训练是否在第一次保存前报错退出。

最小 smoke run 建议先用：

```bash
--save-interval 1
--num-rollout 3
```

这样每轮训练后都会尝试保存，方便立刻确认 checkpoint 链路。正式训练时再把 `--save-interval` 调大，避免频繁写盘。

### 训练能跑但效果不涨

先不要急着改算法，按顺序检查：

1. prompt 是否稳定要求 boxed 输出。
2. reward 是否能区分正确和错误。
3. `n-samples-per-prompt` 是否足够产生组内差异。
4. response 是否大量截断。
5. 数据难度是否远高于当前 0.5B 模型能力。
6. 学习率是否过大或过小。
7. 是否需要动态采样过滤掉 reward 全同的 prompt 组。

0.5B 的目标主要是工程验证；如果要追求数学能力提升，通常需要更大的 Qwen2.5、更多 rollout、更长 response、更合适的数据难度和更完整的实验监控。

## 第 13 步：最小配置到正式配置的改动清单

从教程配置走向正式训练，可以按这张表逐步改：

| 项目 | smoke run | 正式训练方向 |
| --- | --- | --- |
| 模型 | `Qwen2.5-0.5B-Instruct` | `Qwen2.5-3B/7B` 或更大 |
| 数据 | `head -n 64` | 完整训练集，必要时混合多个数学源 |
| rollout 轮数 | `3` | 几百到几千轮 |
| 每题采样数 | `4` | `8` 或更多 |
| response 长度 | `1024` | `4096`、`8192` 或按任务需要 |
| reward | `math` | `math`、`dapo` 或自定义 verifier |
| 动态采样 | 关闭 | 打开 `check_reward_nonzero_std` |
| 评估 | 少量 AIME | 多套数学 eval config |
| 监控 | 看终端日志 | 接入 SwanLab 或 WandB |
| checkpoint | 少量保存 | 固定间隔保存并保留关键版本 |

## 第 14 步：当前仓库默认 skill 速查

slime 仓库当前默认技能放在 `.claude/skills/` 下。它们不是训练时自动执行的模块，而是给 Codex 或其他协作 agent 使用的工程指导：当你要生成某类代码、配置、测试或审查结论时，先看对应 skill，按里面的接口、参数、验证方式和常见错误来做。

当前仓库默认包含这些 skill：

```text
.claude/skills/add-dynamic-filter/SKILL.md
.claude/skills/add-eval-dataset-config/SKILL.md
.claude/skills/add-reward-function/SKILL.md
.claude/skills/add-rollout-function/SKILL.md
.claude/skills/add-tests-and-ci/SKILL.md
.claude/skills/slime-code-review-preferences/SKILL.md
```

| skill | 主要指导生成什么 | 什么时候用 | 关键产物或入口 |
| --- | --- | --- | --- |
| `add-dynamic-filter` | 生成 rollout 采样过滤、buffer 过滤、单样本移除或全样本后处理逻辑。 | 想筛掉 reward 全同的 prompt 组、训练前从 buffer 中挑样本、给某些 rollout sample 打 `remove_sample` 标记、或统计所有生成样本时使用。 | `--dynamic-sampling-filter-path`、`--buffer-filter-path`、`--rollout-sample-filter-path`、`--rollout-all-samples-process-path`。 |
| `add-eval-dataset-config` | 生成或整理评估数据集配置。 | 想添加周期性 eval、从 `--eval-prompt-data` 迁移到 `--eval-config`、或给不同评估集设置不同采样参数、key、`rm_type` 和 metadata 时使用。 | `--eval-config <yaml>`、`--eval-prompt-data`、`slime/utils/eval_config.py`。 |
| `add-reward-function` | 生成自定义 reward 函数和可选 reward 后处理。 | 内置 `math`、`dapo`、`deepscaler` 不够用，想接自己的数学 verifier、远程 reward 服务、任务特定 reward shaping 时使用。 | `slime/rollout/rm_hub/<your_rm>.py`、`--custom-rm-path`、`--custom-reward-post-process-path`。 |
| `add-rollout-function` | 生成完整自定义 rollout 函数。 | 默认 `slime.rollout.sglang_rollout.generate_rollout` 不够用，需要替换训练或评估的数据生成编排时使用。比如复杂多轮环境、非标准样本结构、特殊 train/eval 输出。 | `slime/rollout/<your_rollout>.py`、`--rollout-function-path`、`RolloutFnTrainOutput`、`RolloutFnEvalOutput`。 |
| `add-tests-and-ci` | 生成测试文件和 CI 注册改动。 | 新增行为需要测试、修改已有 `tests/`、新增 GPU/e2e 用例、更新 GitHub workflow matrix 或说明本地验证命令时使用。 | `tests/test_<feature>.py`、`NUM_GPUS`、`.github/workflows/pr-test.yml.j2`、生成后的 `.github/workflows/pr-test.yml`。 |
| `slime-code-review-preferences` | 生成代码审查意见或指导代码修改风格。 | 做 slime 代码 review 或重构时使用，特别是 helper API、分支选择、参数校验、减少薄 wrapper、让控制流更自解释这类问题。 | 审查结论、重构建议、删除无意义 wrapper、调整分支条件和断言位置。 |

和本教程最相关的是三个：

1. `add-reward-function`：当 `--rm-type math` 不能满足你的数学评分规则时，用它生成自己的 verifier reward。
2. `add-dynamic-filter`：当 GRPO 组内 reward 经常全 0 或全 1 时，用它生成动态采样过滤，保留有区分度的 prompt 组。
3. `add-eval-dataset-config`：当你不只评估 AIME，而是要同时评估 GSM8K、MATH、AIME、AMC 等多套数据时，用它生成结构化 eval config。

如果只是跑通最小 Qwen2.5 数学 GRPO，不需要新增任何 skill 代码；照前面的 `--rm-type math` 和最小启动脚本即可。只有在你要扩展 reward、rollout、eval、filter 或测试时，才需要按对应 skill 生成新模块或配置。

## 延伸阅读

- [快速使用](quick_start.md)：完整解释 slime 默认训练脚本的参数块。
- [使用文档](usage.md)：解释数据格式、GRPO、Megatron、SGLang 等核心概念。
- [自定义指南](customization.md)：当 `--rm-type math` 不够用时，使用 `--custom-rm-path` 接入自己的数学 verifier。
