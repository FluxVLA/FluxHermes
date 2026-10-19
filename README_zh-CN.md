# FluxHermes

FluxHermes 是 FluxVLA 的模型部署工具，用于将模型编译为 ONNX 和 TensorRT，
支持部署优化、训练后量化、源模型与引擎的一致性验证，以及机器人部署包生成。

> **当前状态：** 初始框架已经覆盖通用 FluxVLA 适配、TensorRT
> FP32/FP16 构建、INT8 Q/DQ 量化、TensorRT 10/11 运行时、部署包
> 校验、示例配置和无 GPU 单元测试。每个模型系列仍需要提供可导出的封装
> 和代表性样本；真正上机前必须在目标 NVIDIA 设备上完成构建与验证。

## 职责边界

- **FluxVLA**：模型、checkpoint、数据预处理、归一化/反归一化和动作语义。
- **FluxHermes**：ONNX 导出、图优化、量化、TensorRT 构建、性能测试、
  一致性校验和真机运行时。
- **FluxThemis**：仿真/真机 rollout、评测任务和结果管理。

FluxHermes 不复制 FluxVLA 的模型语义。预处理器、tokenizer、统计量和机器人
embodiment 资产会作为部署包的一部分被显式打包。

## 参考 MMDeploy 的架构调整

FluxHermes 借鉴 MMDeploy 的职责分层，但不依赖 MMDeploy 本身：

- `codebase + task processor`：负责 FluxVLA 模型语义和导出输入；
- `IR`：负责 ONNX 契约与导出期 rewrite；
- `backend manager + wrapper`：负责 TensorRT 转换和统一运行时；
- `function rewriter`：临时替换后端不支持或需要优化的算子/函数；
- `pipeline manager`：记录每个阶段的耗时、失败信息，并支持 hook。

在此基础上增加 VLA 真机部署要求：代表性校准数据、动作输出一致性、目标 GPU
匹配、部署包校验，以及默认拒绝运行未验证产物。详细对照见
[MMDeploy 架构对照](docs/mmdeploy_alignment.md)。

## 核心能力

1. 使用稳定、具名的输入输出契约导出 `predict_action`。
2. 支持静态形状和 TensorRT min/opt/max 动态形状 profile。
3. 支持 ONNX 简化、TensorRT timing cache 和 FP32/FP16 构建。
4. 使用真实机器人代表性数据进行 INT8 Q/DQ PTQ。
5. 自动比较 PyTorch 源模型和 TensorRT 输出。
6. 生成带 SHA-256 完整性校验的真机部署 bundle。
7. 提供基于 PyTorch CUDA buffer 的 TensorRT 10/11 Python 运行时。

## 安装

基础环境只用于查看配置、检查部署包和运行测试：

```bash
cd /root/projects/FluxHermes
python -m pip install -e .
fluxhermes doctor
```

导出和量化环境：

```bash
python -m pip install -e '.[export,quantization,dev]'
```

TensorRT 必须使用和目标 CUDA/JetPack 对应的 NVIDIA 发行版本，仓库不会自动
安装一个可能与目标驱动不兼容的 TensorRT wheel。

## 快速开始

先做 dry-run；此命令不会加载 checkpoint，也不会调用 GPU：

```bash
export FLUXVLA_ROOT=/root/projects/FluxVLA
fluxhermes compile configs/pi05/pi05_aloha_fp16.py
```

准备与配置中输入名称一致的代表性 NPZ，然后执行完整构建：

```bash
export PI05_CHECKPOINT=/path/to/model.safetensors
export PI05_DEPLOY_SAMPLE=/path/to/pi05_sample.npz

fluxhermes compile configs/pi05/pi05_aloha_fp16.py --execute
fluxhermes inspect artifacts/pi05-aloha-fp16
```

INT8 量化还需要一组代表性校准数据：

```bash
export PI05_CALIBRATION_NPZ_DIR=/path/to/calibration_npz
fluxhermes compile configs/pi05/pi05_aloha_int8.py --execute
```

## 真机运行

运行时默认拒绝加载未通过源模型/TensorRT 一致性校验的 bundle：

```python
from fluxhermes.runtime import RobotPolicy

policy = RobotPolicy.from_bundle(
    '/root/projects/FluxHermes/artifacts/pi05-aloha-fp16')

actions = policy.predict_action(
    images=images,
    lang_tokens=lang_tokens,
    states=states,
    lang_masks=lang_masks,
    noise=noise,
)
```

如果真机循环输入的是原始图像、关节状态和自然语言，需要实现一个 host
adapter，将它和 tokenizer、归一化统计、相机/机器人配置一起放进
`package.assets`，并设置 `package.runtime.host_adapter='module:ClassName'`。
详细说明见 [真机部署指南](docs/robot_runtime.md)。

## 目录结构

```text
configs/                 PI0.5/GR00T 部署配置模板
docs/                    架构、模型适配、量化和真机文档
fluxhermes/adapters/     FluxVLA 与模型系列适配层
fluxhermes/apis/         对外 task/backend 构建 API
fluxhermes/codebase/     codebase 与 task processor
fluxhermes/backend/      backend manager 与运行时 wrapper
fluxhermes/core/         registry、pipeline manager 和 rewriter
fluxhermes/exporters/    ONNX 导出
fluxhermes/optimization/ ONNX 图优化
fluxhermes/quantization/ INT8 Q/DQ 量化
fluxhermes/builders/     TensorRT 构建器
fluxhermes/runtime/      TensorRT 与真机 Policy 运行时
fluxhermes/validation/   源模型/引擎一致性校验
test/                    无 GPU 基础测试
artifacts/               部署产物（默认不纳入 Git）
work_dirs/               中间文件和构建缓存（默认不纳入 Git）
```

## 开发验证

```bash
python -m unittest discover -s test -v
pytest
flake8 fluxhermes test scripts
```

## License

Apache License 2.0。
