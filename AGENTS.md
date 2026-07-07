# AGENTS.md

本文件只规定代码库协作规则和不可忽略的实现纪律。它不绑定具体实验任务、数据格式、模型配置或某一版方案文档。

## 1. 文档协作说明

### 文档边界

- `AGENTS.md` 不记录具体实验任务、数据格式、模型配置或某一版方案文档，只保留跨任务通用的协作纪律。
- `doc/` 代码库公共说明、工程协作说明、环境搭建、服务器操作和核心代码模块说明等与代码库维护直接相关的文档。
- `projects/` 具体实验、基准数据集、子想法或子项目文档保留在其所属目录。
- `codex/` 保存 Codex 协作时需要执行或追踪的计划、审核表和执行记录；内容应明确区分已完成项和未完成项。
- `doc/hjxa.doc/整体框架.md` 应尽可能简洁，只保留目录职责、模块边界和主调用关系；各部分具体说明应使用相对链接指向对应文档或源码入口，避免在总览中重复展开。
- 模块文档只描述当前代码实际支持的接口、路径、参数和流程；已经删除或不再支持的内容不做历史说明、迁移说明或对比说明。
- 所有说明类文档必须包含 `## 目录` 小节，并使用 `text` 代码块列出相关目录、文件和中文职责；目录只做定位，不替代后续必要说明。
- 修改代码、整理文档或做项目重构时，不要把 `.vllm/`、`.sglang/`、`.git/` 纳入业务模块说明、源码重构范围或手动清理范围，除非用户明确要求。

目录格式示例：

```text
hjxa/
  infer/
    inference_backends/
      base.py               # BaseModelBackend，只提供 load/get/shutdown
      transformer.py        # 加载 Hugging Face model/tokenizer
      vllm.py               # 加载 vLLM LLM，或启动 vLLM OpenAI-compatible 服务
      sglang.py             # 加载 SGLang Engine，或启动 SGLang OpenAI-compatible 服务
      remote_api.py         # 构造 Agno Agent
      utils/
        backend.py          # 后端名称、模式和资源读取辅助
        registry.py         # get_backend_class 与 build_backend
        local_service.py    # 本地服务端口探测和进程管理
        http_openai_like.py # 远程 API 的 OpenAI-like 适配
    generate/
      generate_batch.py     # 批量生成入口
      answer_ppl.py         # 答案 PPL 入口
      utils/                # chat template、保存、抽取和 scoring 辅助
  train/
    causal_lm_training.py  # run_causal_lm_training 主入口
    coe_trainer.py          # CoETrainer，扩展 HuggingFace Trainer 诊断和 token_acc
    coe_utils/              # hidden states 聚合和 CoE 指标计算
    tokenized_collator.py   # TokenizedCausalCollator，消费外部 token ids 和 labels
    utils/
      checkpoints.py        # 训练 checkpoint 与 LoRA final 保存
      dataset_contract.py   # 外部 dataset 固定列名检查和长度读取
      model_loading.py      # 模型加载参数、dtype 和 embedding/context 校验
      output_files.py       # 训练结果 JSON 写出
      save_tensor.py        # AsyncTensorSaver，后台 gzip 保存训练张量
      trainer_args.py       # Trainer 默认配置、Liger 校验和 TrainingArguments 构造
      training_config.py    # 训练 YAML 配置读取
      training_metrics.py   # token_acc 等训练指标开关读取
  utils/
    config.py             # Config、ModelConfig 与 load_config
    paths.py              # 项目 .env 与存储目录解析
    logging.py            # 统一日志、日志分流和行为开关解析
```

### 文档语言

- 所有新增或修改的文档说明使用中文。
- 代码标识符、配置键名、第三方 API 参数名和协议字段可按原接口保留英文。

## 2. 代码协作说明

### 基本规则

- 所有新增或修改的注释、异常消息和面向用户的运行输出使用中文。

### 注释规则

- 核心逻辑必须写中文注释，说明为什么这样做以及它保证什么约束。
- 每个函数均需要有函数作用说明，优先使用中文 docstring；若函数为极短 helper，也必须用紧邻注释说明其作用。
- `if`、`for`、`while`、`try` 等控制语句前需要中文注释，说明这段分支、循环或异常处理是在做什么。
- 重要语句前需要中文注释，说明该语句为什么关键、维护了什么约束或避免了什么风险。
- 字符串处理逻辑必须在注释中给出输入输出例子，说明从什么样的字符串转化成什么样的字符串。
- 仅为 dry-run 服务、正式运行不会使用的函数，必须在 `def` 的上一行写固定注释：`# 仅dry-run服务，正式运行不用`。

示例：

```python
# 只在主进程写 Parquet，避免多进程同时写同一文件造成分片损坏。
writer.write_batch(batch)

# 将完整样本文本 `A>B,B>C,A?:A>B>C` 截成模型实际输入 `A>B,B>C,A?:`。
prompt = text.rsplit(":", 1)[0] + ":"
```

### 日志规则

- 人类可读日志、命令行输出和评估摘要默认使用中文。
- 允许保留英文级别词：`ERROR`、`WARNING`、`INFO`、`DEBUG`。
- 日志级别只使用 `INFO`、`WARNING`、`ERROR`、`DEBUG` 四种。
- 项目入口必须调用 `hjxa.utils.logging.setup_logging()` 作为唯一日志初始化入口，并通过 `log_dir` 或 `log_path` 传入项目日志保存位置；日志系统不提供默认保存目录，缺少保存路径时应直接报错。
- 外部项目调用框架时，应把项目源码根目录通过 `project_roots` 传给 `setup_logging()`，确保项目日志和框架日志使用同一套格式并写入同一个 `all.log`。
- 同一进程内不要重复初始化日志；如子流程或框架入口再次调用 `setup_logging()`，应复用已有日志状态，不应额外添加 handler 或生成第二个日志目录。
- 日志文件、日志格式、日志分流和日志行为开关只维护在 [`doc/hjxa.doc/utils.md`](doc/hjxa.doc/utils.md)，其他文档只链接引用，不重复展开。

### 验证规则

- 注意在 MAC 机器做时不用实际运行 CUDA 相关的代码，只需审查即可。
- 环境在 `.xxx` 下，`xxx` 为相应的后端，例如 `vllm` 或 `sglang`。


### 隐藏目录处理规则
- `.vllm/` 与 `.sglang/` 等是后端运行环境目录，通常包含第三方包、后端源码或虚拟环境文件；它们服务于运行 vLLM 和 SGLang 后端，不属于项目源码模块，但是如果有涉及第三方包的操作需要查看官方原码，可以在这两个目录中只读寻找。
- `.git/` 是版本控制内部目录，只保存 Git 元数据，不参与业务逻辑或架构设计。

### git commit 格式规范
- 当用户明确要求需要 commit 时，commit message 必须使用格式：`【月份日期】【项目名称或是hjxa】【实现类别】简要说明`。
- `月份日期` 使用四位数字 `MMDD`，例如 6 月 20 日写作 `0620`。
- `项目名称或是hjxa` 用于标识变更归属：框架代码、公共文档和通用工具写 `hjxa`；具体实验或子项目写项目名，例如 `DOGAC`。
- `实现类别` 使用简短类别词，优先从 `feat`、`fix`、`docs`、`test`、`refactor`、`chore` 中选择；如用户或历史提交已有更贴切类别，可沿用。
- `简要说明` 使用中文短句，概括本次提交的主要结果，不写句号。
- 示例：`【0620】【hjxa】【fix】多卡多次创建swanlab修复`。

## 3. 实验项目构造逻辑
- 实验方案文档需要引用公共模块时，应使用相对链接指向 `doc/`、源码入口或配置示例，避免复制公共框架说明。
- 实验项目默认按“数据生成、训练、测评”三个阶段组织；如需要模型初始化、tokenizer 构建或结果汇总等附加阶段，应保持为独立脚本，不要塞进某个阶段内部。
- 每个阶段必须有单独脚本，例如 `01_generate_data.sh`、`10_train.sh`、`20_eval.sh`；同时必须有一个一键全流程脚本，例如 `90_run_all.sh`，能从头执行完整链路。
- 单独阶段脚本和一键全流程脚本都必须支持断点重运行；阶段只有在所有关键产物成功落盘后才能写 `_SUCCESS` 或等价完成标识，重跑时只能根据该标识跳过，不能因为目录存在就跳过。
- 断点标识必须按阶段和 seed 隔离，避免 seed A 的完成状态影响 seed B；失败、中断或半成品目录不得被误判为完成。
- 生成数据、测评阶段应支持断点续跑，已经成功生成和计分的样本不应在断电重跑时重复生成；同时输出需要同时保留机器可读结果和少量样例，便于快速审核。
- 训练阶段断电或中断重跑时，如果训练阶段尚未写完成标识但输出目录中已有 HuggingFace `checkpoint-*`，必须从最新 checkpoint 继续训练；不能从初始模型重新开始覆盖旧进度。
- 实验运行产生的 `logs/`、`runs/`、`data/`、`models/`、`swanlog/` 内容默认视为运行产物，不要提交或手动清理其中的任何文件，哪怕用户明确要求。

### 项目配置与环境

- 每个实验项目默认目录骨架为：

```text
configs/          # 阶段配置和项目总配置
data/             # 数据产物逻辑目录，可随项目 .env 的产物根目录迁移
doc/              # 项目方案、数据生成、运行说明和结果汇总
logs/             # 日志产物逻辑目录，可随项目 .env 的产物根目录迁移
models/           # 模型与 tokenizer 产物逻辑目录，可随项目 .env 的产物根目录迁移
runs/             # 训练、测评和中间运行产物逻辑目录，可随项目 .env 的产物根目录迁移
scripts/          # 阶段脚本和一键全流程脚本
src/              # 项目特有数据、训练适配、评测和工具代码
requirements.txt  # 项目额外依赖
```

- `models/`、`runs/`、`data/`、`logs/` 是逻辑目录名，实际物理根目录应由项目 `.env` 或项目总配置统一决定；不同服务器只改 `.env`，不要修改阶段配置中的相对路径语义。
- 每个实验项目应有项目内 `.env` 和 `.env.example`，用于声明不同服务器上的产物根目录和 `PATH` 前缀，其他的则放入配置文件中；不要把某台服务器的绝对路径硬编码进代码。
- 项目脚本应优先加载项目内 `.env`，同时允许 shell 中显式传入的环境变量覆盖 `.env`；这样服务器临时调度不需要修改仓库文件。
- 项目配置必须清晰拆分数据生成、模型或 tokenizer 初始化、训练、测评与项目总配置；阶段配置只描述该阶段参数，公共根目录和运行环境入口放在项目总配置或项目 `.env` 中统一管理。
- 项目代码应优先调用 `hjxa` 框架的通用训练、推理、日志、路径和配置能力；只有任务特有的数据构造、指标解析和样本格式留在 `projects/<项目名>/src/` 中。
- 项目主入口文件只保留主流程编排，例如解析参数、加载配置、构造核心对象和调用框架入口；checkpoint 恢复、路径解析、日志初始化、样本转换、指标细节等辅助逻辑应放入同级 `utils.py`、`utils/` 或语义明确的 helper 模块中。
- 正式配置必须明确写出 `seed`、输入输出路径、模型路径、训练 batch、日志上传、评测数据列表和最大生成长度等关键参数；不要依赖读代码才能知道实验设置。

### 脚本调度与多 seed

- 每个正式实验默认提供多 seed 重复实验入口；seed 列表必须可通过环境变量或命令行覆盖，例如 `PROJECT_SEEDS="42 43 44 45 46"`。
- 一键全流程脚本应支持“一张卡一个任务”的 GPU 队列，也应支持“一个任务占用多张卡”的顺序队列；建议提供 `--gpu-groups` 参数，组内用逗号分隔 GPU，组间用分号分隔 GPU 组。

```bash
# 一张卡一个任务：5 个 seed 分别占用 GPU 0..4。
PROJECT_SEEDS="42 43 44 45 46" bash scripts/90_run_all.sh --gpu-groups "0;1;2;3;4"

# 一个任务占用 4 张卡：两个 GPU 组并行，每组内部顺序取下一个 seed。
PROJECT_SEEDS="42 43 44 45 46" bash scripts/90_run_all.sh --gpu-groups "0,1,2,3;4,5,6,7"
```

- GPU 调度脚本必须显式写入 `CUDA_VISIBLE_DEVICES`，并在日志中记录当前 seed、GPU 绑定和阶段名称；不要让多个 seed 隐式争抢同一张卡。
- 如果数据生成不依赖本地部署的 GPU 模型，一键全流程脚本必须将数据生成与训练/测评解耦：数据生成应作为 CPU/IO 阶段后台预取，不占用 GPU 队列；GPU 队列只等待对应 seed 数据完成后执行模型初始化、训练和测评，以便当前 seed 训练时并行生成后续 seed 数据。
- CPU/IO 数据生成预取不能固定按 seed 串行；应根据单个数据生成任务需要的 CPU 数和当前机器 CPU 预算自动决定并发数。默认 CPU 预算应保守，例如不超过机器 CPU 数的一半；当 `seed 数 * 单任务 CPU 数 <= CPU 预算` 时，应同时启动全部 seed 的数据生成；否则使用后台 worker 队列按预算并发推进，并提供环境变量或命令行参数覆盖单任务 CPU 数、CPU 总预算和数据生成并发数。
- 多卡单任务训练应使用框架或训练入口支持的分布式方式，例如 DDP、tensor parallel 或后端自己的多卡机制；不要在外层脚本中手工拆 batch。
- 脚本需要允许“顺序安排 + 后台并行队列 + 一行命令启动”，但默认并发数应保守，例如用 `--tasks-per-gpu-group 1` 控制每个 GPU 组同时只跑一个正式任务。


### 日志与监控

- 每个命令只能有一个明确的日志归属方：项目入口调用框架日志系统并传入保存路径，后续项目代码、框架代码、第三方 logging 和子进程输出统一进入同一个 `all.log`；不要同时保存两套内容相同的人类可读日志。
- 阶段脚本可以保存配置快照、命令行和阶段状态，但不应重复捕获框架日志系统已经保存的完整 stdout/stderr；需要重复保存时必须说明用途，例如远程调度系统只读外层日志。
- 一键全流程脚本必须把外层调度输出额外 `tee` 到统一日志文件，路径放在当前项目产物根目录的 `logs/<run_name>_<timestamp>/all.log`；该日志用于保存 GPU 队列、CPU 预取、seed 调度和子命令终端输出，不替代各阶段入口自己通过框架日志系统写出的 `all.log`。
- 日志目录必须包含运行名称、阶段名、时间戳或唯一后缀，并能从日志反查使用的配置文件、seed、GPU 绑定和产物目录。
- 正式训练默认上传 SwanLab，测试时，要在项目名称后加上`_test`后缀。
- 正式训练默认开启 CoE 指标；默认不保存 hidden states，除非诊断配置显式打开，避免正式训练额外放大磁盘和显存消耗。


### 最小全流程

- 每个实验项目必须提供小规模全流程配置，能用极少数据量、极少训练步和极少评测样本跑完“数据生成、训练、测评、监控上传”链路；该配置只能用于工程验证，不能用于论文结论。但是要注意特别是训练和评测时只是训练步数和评测样本数减少，但应该训练、评测一步的batch_size和正式运行时一致，以免正式运行OOM，此处可以重复生成的小数据量来达到指定的batch_size。
