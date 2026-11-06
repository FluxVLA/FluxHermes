# FluxHermes

FluxHermes 是 FluxVLA 的模型部署工具，用于将模型编译为 ONNX 和 TensorRT，
支持部署优化、训练后量化、源模型与引擎的一致性验证，以及机器人部署包生成。

> **当前状态：** PI0.5 已接通独立 SDK、双图编译打包和多引擎运行时。
> 此前 RTX 3090 / TensorRT 10.13.3.9 验证通过了小模型数值对齐，以及真实业务权重的
> 原生 PyTorch 与 SDK 逐位对齐。但完整业务模型的 TensorRT 数值验证**失败**，
> 最大绝对误差为 0.05605；目前不宣称业务模型可部署或已获得加速。

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

## PI0.5 独立导出

构建机安装由 CI 固定源码提交构建的独立 SDK wheel；pip 只构建指定包子目录，
不安装 FluxVLA 根目录的训练项目：

```bash
SDK_REVISION=55a464ae753f16dd19f444f63e321cea5535a6dc
python -m pip install \
  "fluxvla-export[test-onnx] @ git+https://github.com/ZaneZhang-67/FluxVLA.git@${SDK_REVISION}#subdirectory=packages/fluxvla-export" \
  '.[export,onnxruntime,dev]'
```

`test-onnx` extra 固定已验证的导出依赖。仅在开发 SDK 时，才可选用相邻仓库的
`../FluxVLA/packages/fluxvla-export[test-onnx]` 安装方式；运行部署包不要求源码目录。

训练侧调用 SDK 的 `produce_pi05_artifacts`，提供部署 JSON、已预处理的六个
具名 tensor（包括固定 noise）以及原始 safetensors checkpoint。存储 dtype 默认为
FP32，也支持显式声明的 BF16 存储。SDK 校验后将权重载入 FP32 模型参数，与普通
PyTorch 对同一 checkpoint 的加载语义一致；BF16 存储与 BF16 autocast 计算是不同选择。
原始 checkpoint 文件不重写。预处理来源必须显式提供；Hermes 不复制模型实现，
不重建训练配置，也不实现 checkpoint 加载器。

```bash
export PI05_DEPLOYMENT_JSON=/path/to/export-case/export_config.json
export PI05_CHECKPOINT=/path/to/model.safetensors
export PI05_EXPORT_INPUTS=/path/to/export-case/export_inputs.safetensors

fluxhermes compile configs/pi05/pi05_export_fp32.py --execute
fluxhermes run artifacts/pi05-export-fp32 --inputs "$PI05_EXPORT_INPUTS" \
  --device cuda:0 --output /tmp/actions.npz
fluxhermes benchmark artifacts/pi05-export-fp32 --inputs "$PI05_EXPORT_INPUTS" \
  --device cuda:0 --warmup 10 --repeats 100 \
  --reference-config configs/pi05/pi05_export_fp32.py
```

配方自动选择 FP32 legacy/opset18 或 BF16 Dynamo/opset22（`optimize=True`）。标准
优化器将常量 reduction axes 固化为 TensorRT 要求的 initializer。这里的后端
`precision=fp32` 表示关闭全图降精，保留模型的显式 BF16 和 FP32 保护路径；
TensorRT 使用强类型并关闭 TF32。此配方暂不支持 FP16 转换、INT8、RTC 或动态 fixture。
双图与完整 Euler actions 都通过校验后，机器人运行时才接受 bundle。

CLI 支持 NPZ 和 safetensors。基准计时覆盖输入已在设备上的同步策略推理，不含读文件、
加载引擎和 H2D。加上 `--reference-config` 后，先核对同一配置、checkpoint 和输入，
计时 PyTorch 并释放模型，再加载引擎；数值对齐通过才输出实测比值，小于 1 就是变慢。
每次构建使用新的 work/bundle 路径，不会自动删除已有部署包或 staging 目录。
CPU `onnxruntime` 后端拒绝 BF16，不能代替 TensorRT 验证。

```bash
python -m unittest discover -s test -v
python -m unittest discover -s test_integration -v
# 目标机真实 GPU 检查；缺 CUDA/TensorRT 会报错，不会跳过后显示成功。
FLUXHERMES_CUDA_DEVICE=cuda:0 python -m unittest discover -s test_gpu -v
```

### 验证边界

以上 NVIDIA 结果来自 2026-11-05，并非本次发布重新运行。真实业务 BF16 存储权重的
原生 PyTorch 与 SDK 逐位一致；双图 ONNX 导出、TensorRT 构建也通过了。但完整
TensorRT 数值验证在固定 `atol=0.02`、`rtol=0.02`、余弦 `0.999` 门限下失败。
引擎能构建不等于策略通过验证，不应放宽门限或绕过验证来部署失败产物。

3.62B 参数模型使用分阶段 harness 完成实验，普通完整 `CompilePipeline` 的内存
可行性仍未证实。仓库的两个 GPU 测试文件检查 Add 图运行时、stream/profile 和
日志/JSON 行为，不等同于独立的小策略或业务模型 GPU 验证。

## 旧版训练环境适配入口

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

运行时默认拒绝加载未通过源模型/TensorRT 一致性校验的 bundle。独立 SDK 部署包
仍需安装 SDK executor 包和后端，但不需要训练目录、模型构造、原始权重或导出 JSON。
TensorRT 使用目标 CUDA 对应的 NVIDIA 发行版；运行时依赖可单独安装：

```bash
SDK_REVISION=55a464ae753f16dd19f444f63e321cea5535a6dc
python -m pip install \
  "fluxvla-export @ git+https://github.com/ZaneZhang-67/FluxVLA.git@${SDK_REVISION}#subdirectory=packages/fluxvla-export" \
  '.[runtime]' 'safetensors>=0.4,<1'
```

CLI `run`/`benchmark` 需要 Torch；safetensors 输入还需要上面的 `safetensors` 包，
NPZ 使用基础 NumPy 依赖。独立进程测试会检查未导入模型/训练模块；这与环境中是否
安装了这些包是两回事。

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
flake8 fluxhermes test scripts test_integration test_gpu
```

## License

Apache License 2.0。
