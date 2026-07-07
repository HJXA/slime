# Docker release rule

We will publish 2 kinds of docker images:
1. stable version, which based on official sglang release. We will store the patch on those versions.
2. latest version, which aligns to `lmsysorg/sglang:latest`.

current stable version is:
- sglang v0.5.12.post1 (5a15cde858ea09b77116212a39356f2fc51b8584), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087

history versions:
- sglang v0.5.10.post1 (7c35342c10e201899e22fe2972d40e60da19ff3e), megatron dev 1dcf0dafa884ad52ffb243625717a3471643e087
- sglang v0.5.9 (bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.7 nightly-dev-20260107-dce8b060 (dce8b0606c06d3a191a24c7b8cbe8e238ab316c9), megatron dev 3714d81d418c9f1bca4594fc35f9e8289f652862
- sglang v0.5.6 nightly-dev-20251208-5e2cda61 (5e2cda6158e670e64b926a9985d65826c537ac82), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.5.post1 (303cc957e62384044dfa8e52d7d8af8abe12f0ac), megatron v0.14.0 (23e00ed0963c35382dfe8a5a94fb3cda4d21e133)
- sglang v0.5.0rc0-cu126 (8ecf6b9d2480c3f600826c7d8fef6a16ed603c3f), megatron 48406695c4efcf1026a7ed70bb390793918dd97b

The command to build:

```bash
just release
```

## A100 / CUDA 12.6 可迁移镜像

如果要把 A100 + CUDA 12.6 / cu126 环境打成可迁移 Docker 镜像，可以使用专用 Dockerfile：

```bash
bash docker/build_a100_cu126.sh
```

默认镜像名是 `slime:a100-cu126`。可以通过环境变量覆盖：

```bash
IMAGE_TAG=my-slime:a100-cu126 MAX_JOBS=16 bash docker/build_a100_cu126.sh
```

裸机安装脚本支持断点续跑，完成标识默认写在 `${BASE_DIR}/.resume_a100_cu126`。网络中断或编译失败后重复运行同一条裸机安装命令即可；如需忽略断点重新构建或重新验证，可使用 `--force-build`、`--force-verify` 或 `--reset-resume`。Docker 构建仍遵循 Docker layer cache：已经成功提交的层会复用，但失败的长 `RUN` 层中的临时进度不保证保留。

如果需要把镜像迁移到另一台机器，可以导出和导入：

```bash
docker save slime:a100-cu126 -o slime_a100_cu126.tar
docker load -i slime_a100_cu126.tar
```

镜像构建阶段通常没有 GPU，因此不会执行运行时验证。把镜像迁移到 A100 机器后，用下面命令验证：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  slime:a100-cu126 verify-a100-cu126
```

验证会检查 PyTorch CUDA 版本是否为 12.6、是否检测到 A100 / SM80、关键环境变量是否正确，以及 `slime`、`sglang`、`sglang_router`、`transformer_engine`、`deep_ep`、`megatron` 是否能正常导入。

Before each update, we will test the following models with 64xH100:

- Qwen3-4B sync
- Qwen3-4B async
- Qwen3-30B-A3B sync
- Qwen3-30B-A3B fp8 sync
- GLM-4.5-355B-A32B sync
