# VidGen 当前状态与本地部署执行计划
> 最后更新：2026-04-14

## 1. 当前结论
项目已经验证了两条不同层级的结果：

- 不带 `control_plan` 的最小真实视频生成已经打通。
- 带 `control_plan` 的真实生成仍未稳定通过，当前主阻断不再是 API 逻辑，而是本机环境容量和底层推理稳定性。

已经确认成功的能力：

- `VidGen` conda 环境可用。
- 本地 `SDXL`、`SVD`、`ControlNet(OpenPose/Depth)` 模型目录可识别。
- `scripts/test_local_sdxl_load.py --skip-infer` 和 `scripts/test_local_svd_load.py --skip-infer` 通过。
- OpenPose / Depth Anything 预处理器可离线初始化，并可对单张测试图执行最小预处理。
- 关闭 `control_plan` 后，API 路径可成功产出真实 `json/png/mp4`。

## 2. 这次确认过的问题

### 2.1 环境与依赖问题
- 初始环境缺少 `fastapi`、`pydantic_settings`、`ruff`，导致轻量测试无法跑。
- 基础环境中还出现过 `numpy` 与 `scikit-learn` ABI 不兼容，报 `numpy.dtype size changed`。
- 结论：开发依赖和 ML 依赖必须区分；模型相关检查必须在正确的 `VidGen` 环境中做。

### 2.2 默认运行策略不安全
- `Settings.device` 原先默认 `cuda`，普通开发或显存紧张时容易直接走高风险路径。
- 已修复：默认设备改为 `cpu`，需要 GPU 时显式设置 `VIDGEN_DEVICE=cuda`。

### 2.3 无 ML 栈时缓存清理会直接崩
- `clear_pipeline_cache()` 曾无条件导入 `torch`，导致无 ML 依赖环境下连轻量测试都失败。
- 已修复：改为受保护导入，未安装 `torch` 时只做 Python 级缓存清理。

### 2.4 OpenPose / Depth Anything 的离线与兼容性问题
- `mediapipe` 版本不兼容时，OpenPose 初始化失败，表现为 `mediapipe.solutions` 缺失。
- `depth-anything-small-hf` 本地缓存不完整时，Depth Anything 在 `local_files_only=True` 下无法初始化。
- 已修复：将 `mediapipe` 固定到可用版本，补齐 `lllyasviel/Annotators` 和 `LiheYoung/depth-anything-small-hf` 的离线缓存。

### 2.5 API 默认会强制走控制链路
- 标准 `JobService` / API 路径原本只要有 `control_plan` 就会推进控制链路，难以验证“最小真实生成”。
- 已修复：
  - 新增 `enable_control_plan` 请求字段。
  - `scripts/submit_demo.py` 新增 `--disable-control-plan`。
  - 只有 `generation_context.metadata` 中真的存在 `pose_asset_images` / `depth_asset_images` 时，才进入 ControlNet 分支。

### 2.6 控制链路调试时，空字符串不能覆盖 `.env`
- 通过环境变量把 `VIDGEN_SDXL_OPENPOSE_CONTROLNET_ID` 设为 `''` 时，并不会覆盖 `.env` 中已有值。
- 结果：本想做 `depth-only`，实际仍会把 `openpose` ControlNet 一起加载。
- 已修复：`none` / `disabled` / `off` / `false` / `0` 会被明确视为禁用。

### 2.7 带 control plan 时的底层异常
- 先后出现过：
  - `MemoryError`
  - `c10.dll` 访问冲突
  - `torch_cpu.dll` 访问冲突
  - 长时间卡在 `running`
- 为降低同时占用 SDXL 和 SVD 的风险，已经做过分阶段加载：
  - 首帧 `SDXL + ControlNet` 阶段单独加载
  - 视频 `SVD` 阶段单独加载
  - 首帧阶段强制切到 `cpu`
- 结果：快速崩溃有所缓解，但并未彻底解决。

## 3. 目前最关键的根因

### 3.1 Windows 分页文件 / 内存提交不足
已经通过最小复现确认，问题可以在不跑完整 job 的情况下复现：

```powershell
conda run -n VidGen python -c "from diffusers import StableDiffusionXLPipeline; print('before-sdxl'); pipe=StableDiffusionXLPipeline.from_pretrained(r'B:\agent\MyCode\VidGen\storage\models\stable-diffusion-xl-base-1.0', torch_dtype=None, cache_dir=r'B:\agent\MyCode\VidGen\storage\model-cache'); print(type(pipe).__name__)"
```

复现结果：

- `Loading pipeline components...` 后失败
- 先报 Windows `os error 1455`
- 随后触发 `MemoryError`

结论：

- 当前“带 control plan 的真实生成失败”首先是本机内存 / 分页文件不足问题。
- 在这种状态下，`torch` 原生层可能继续升级为 `c10.dll` / `torch_cpu.dll` 访问冲突，并拖坏 IDE 终端。

### 3.2 控制链路的真实稳定性尚未完成闭环
- 不带 `control_plan` 的最小真实生成已成功。
- 带 `control_plan` 的完整视频生成还未完成最终成功闭环。
- 当前阻断更偏环境容量和底层稳定性，而不是简单的业务逻辑 bug。

## 4. 已完成的修复
- 默认设备从 `cuda` 改为 `cpu`。
- `clear_pipeline_cache()` 去掉对 `torch` 的硬依赖。
- `diffusers_loader` 支持 `device_override`，允许图像阶段和视频阶段分开选设备。
- `video_pipeline` 对控制链路采用 staged loading，先首帧，再视频。
- 增加阶段日志，能判断卡在“图像管线加载 / 首帧 / 视频扩展”的哪一段。
- 增加 `enable_control_plan` 和 `--disable-control-plan`，可稳定验证最小真实生成。
- 规范化“显式禁用 ControlNet ID”的行为，避免被 `.env` 默认值误导。
- 新增和补强了配置、ControlNet 加载、预处理器失败路径等测试。

## 5. 当前建议的本地部署与验证顺序

### Phase A：环境准备
1. 使用 `conda activate VidGen`
2. 保持：
   - `VIDGEN_USE_MOCK_PIPELINE=false`
   - `VIDGEN_TASK_MODE=eager`
   - `HF_HUB_OFFLINE=1`
3. 确认本地模型和缓存存在：
   - `storage/models/stable-diffusion-xl-base-1.0`
   - `storage/models/stable-video-diffusion-img2vid-xt`
   - `storage/models/controlnet-openpose-sdxl-1.0`
   - `storage/models/controlnet-depth-sdxl-1.0`
   - `storage/model-cache`

### Phase B：轻量检查
1. `pytest` 跑轻量测试
2. `ruff check .`
3. `python scripts/test_local_sdxl_load.py --device cpu --skip-infer`
4. `python scripts/test_local_svd_load.py --device cpu --skip-infer`
5. 单张图 OpenPose / Depth Anything 最小执行

### Phase C：最小真实生成
1. 先跑不带 `control_plan` 的最小真实生成
2. 再通过 API 路径验证 `json/png/mp4` 落盘

### Phase D：控制链路验证
1. 先只做单一路控制，不叠加多路条件
2. 先验证首帧 `SDXL + ControlNet`，不直接进视频阶段
3. 稳定后再回到完整 `control_plan + SVD`

## 6. 下一步可行方案

### 方案 1：先修环境容量，再继续控制链路调试
这是当前最优先方案。

- 增大 Windows 分页文件，建议系统管理或至少 `24-48 GB`
- 关闭高内存程序后重启 IDE / 终端
- 再从“只加载 SDXL 图像管线”开始验证，不直接跑完整生成

### 方案 2：继续压低控制链路负载
- 只保留单一路控制，例如 `depth-only`
- 首帧保持 `cpu`
- 进一步降低首帧分辨率或步数

### 方案 3：继续增强可观测性
- 保留 staged loading 和阶段日志
- 如果后续仍卡住，进一步给：
  - ControlNet 模型加载
  - SDXL 图像管线组装
  - 首帧推理
  - SVD 视频扩展
  分别增加更细粒度日志

## 7. 当前状态判断

可以认为项目已达到：
- “最小真实视频生成可用”
- “控制链路的轻量检查可用”

但还没有达到：
- “带 control plan 的完整真实视频生成稳定可复现”

当前主阻断：
- Windows 分页文件不足导致 `SDXL` 加载失败
- 继发的 `torch` 原生层崩溃与终端异常

因此，下一轮工作不应直接重跑重型任务，而应先处理系统内存提交问题，再回到 `depth-only` 的首帧验证。
