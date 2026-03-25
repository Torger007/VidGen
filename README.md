# VidGen

VidGen 是一个文本生成视频的 MVP 脚手架，围绕开源模型编排、任务执行以及未来的微调工作流构建。

## 当前状态

仓库目前已经包含第 1 阶段的第一批实现内容：

- 用于提交任务和查询状态的 FastAPI 服务
- 使用 Celery worker 的异步生成集成
- 适合本地开发、无需 Redis 的 `eager` 任务模式
- Prompt 编排层
- 可替换的视频生成管线接口
- 便于早期迭代的本地文件型任务存储
- 基于 Diffusers 的开源管线接入，支持 `SDXL -> SVD`
- 覆盖完整路线图的架构文档

## 项目结构

- `app/`：API、核心配置、服务、worker 和任务代码
- `docs/architecture.md`：端到端架构与分阶段路线图
- `configs/`：预留给训练和推理配置
- `pipelines/`：预留给可复用的生成管线
- `training/`：预留给 LoRA / 时序微调代码
- `evaluation/`：预留给质量评估脚本
- `tests/`：基础 API 测试

## 快速开始

1. 使用 Python 3.11+ 创建虚拟环境。
2. 安装依赖：

```bash
pip install -e .[dev]
```

3. 复制环境变量文件：

```bash
copy .env.example .env
```

4. 如果只在本地启动，保持 `VIDGEN_TASK_MODE=eager`。
5. 启动 API：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

6. 可选：如果要切换到 Celery 模式，将 `VIDGEN_TASK_MODE=celery`，启动 Redis，然后运行 worker：

```bash
celery -A app.workers.celery_app worker --loglevel=INFO
```

7. 可选：安装机器学习依赖，并关闭 mock 模式：

```bash
pip install -e .[ml]
```

然后设置：

```bash
VIDGEN_USE_MOCK_PIPELINE=false
VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid
```

如果你希望在存在中间件产物时，让 SDXL 首帧生成接入真实的 pose/depth ControlNet 分支，还需要设置：

```bash
VIDGEN_SDXL_OPENPOSE_CONTROLNET_ID=<your-openpose-controlnet-id>
VIDGEN_SDXL_DEPTH_CONTROLNET_ID=<your-depth-controlnet-id>
```

如果你希望适配器执行阶段基于本地参考图真正生成 OpenPose / Depth Anything 预处理资源，还需要设置：

```bash
VIDGEN_OPENPOSE_DETECTOR_ID=lllyasviel/Annotators
VIDGEN_DEPTH_ANYTHING_MODEL_ID=LiheYoung/depth-anything-small-hf
```

或者把首帧图像主干模型切换为 `FLUX.1-dev`：

```bash
VIDGEN_DEFAULT_MODEL=stable-video-diffusion-flux
```

你也可以在 `.env` 中全局设置默认阈值：

```bash
VIDGEN_DEFAULT_MIN_TOTAL_SCORE=0.45
VIDGEN_DEFAULT_MIN_TEXT_ALIGNMENT=0.35
VIDGEN_DEFAULT_MIN_TEMPORAL_STABILITY=0.30
VIDGEN_DEFAULT_GENERATION_PROFILE=balanced
```

## 本地演示环境配置

为了获得可复现的本地演示环境，建议使用 `VidGen` conda 环境，并将模型目录保留在仓库内部。

推荐的激活流程：

```bash
conda activate VidGen
cd /d b:\agent\MyCode\VidGen
```

推荐的真实本地推理环境变量：

```bash
set VIDGEN_USE_MOCK_PIPELINE=false
set VIDGEN_TASK_MODE=eager
set VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid
set VIDGEN_DEVICE=cpu
```

项目内本地模型目录：

```text
storage/models/stable-diffusion-xl-base-1.0
storage/models/stable-video-diffusion-img2vid-xt
storage/models/FLUX.1-dev
```

当前模型注册表会优先使用这些本地目录，只有在本地目录不存在时才回退到 Hugging Face 的模型 ID。

最小化独立模型检查：

```bash
python scripts/test_local_sdxl_load.py --skip-infer
python scripts/test_local_svd_load.py --skip-infer
```

如果你想跑完整的 smoke test：

```bash
python scripts/test_local_sdxl_load.py
python scripts/test_local_svd_load.py
```

## 真实任务演示

如果你已经有本地参考图，建议统一放在 `storage/regression_inputs/` 下，或者其他有文档说明的本地固定路径。

示例：直接在本地运行任务

```bash
python - <<'PY'
from app.models.schemas import GenerateVideoRequest
from app.services.job_service import JobService

service = JobService()
job = service.create_job(
    GenerateVideoRequest(
        prompt="A robot walking forward in a city street at night",
        reference_image_path="storage/regression_inputs/verification-reference.png",
        generation_profile="balanced",
        parameters={
            "model": "stable-video-diffusion-img2vid",
            "fps": 4,
            "width": 256,
            "height": 256,
            "num_candidates": 1,
            "retry_attempts": 1,
        },
    )
)

print("job_id=", job.job_id)
print("status=", job.status)
print("error=", job.error_message)
print("output=", job.output_path)
print("preview=", job.output_preview_path)
PY
```

演示和回归输入资源建议统一放在：

```text
storage/regression_inputs/
```

输出文件会写入：

```text
storage/jobs/
storage/outputs/
storage/adapters/
```

对于已完成的任务，可以查看：

```text
storage/jobs/<job_id>.json
storage/outputs/<job_id>.json
storage/outputs/<job_id>.png
storage/outputs/<job_id>.mp4
```

## 本地 API 演示

本地启动 API：

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

常用演示接口：

- `GET /health`
- `POST /v1/videos:generate`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`

你也可以使用演示脚本：

```bash
python scripts/submit_demo.py --prompt "A robot walking through a rainy neon city street at night" --reference-image-path "storage/regression_inputs/verification-reference.png"
```

## 控制回归测试

仓库中包含一个最小化的控制回归脚本，用于比较 baseline 运行结果和 pose-depth 控制运行结果。

回归用例定义位于：

```text
tests/fixtures/regression_cases.json
```

脚本默认会将汇总结果写入：

```text
storage/regression/control-regression-summary.json
```

在项目根目录运行：

```bash
python -m scripts.run_control_regression
```

汇总结果中每个 case 都包含：

- `baseline`
- `pose_depth`
- `assertions`

最值得重点关注的字段有：

- `routing_differs_from_baseline`
- `controlled_has_pose`
- `controlled_has_depth`
- `controlled_video_conditioning_used`
- `score_delta`

如果某次运行失败，可以查看：

```text
storage/jobs/<job_id>.json
storage/outputs/<job_id>.json
terminal traceback
```

## API 接口

- `GET /health`
- `POST /v1/videos:generate`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`

## 请求示例

```json
{
  "prompt": "A woman in a red coat walking through a snowy city street at night",
  "style_hint": "cinematic realistic",
  "reference_image_path": "B:/agent/MyCode/VidGen/examples/reference.png",
  "generation_profile": "balanced",
  "parameters": {
    "model": "mock-svd",
    "duration_sec": 3,
    "fps": 8,
    "num_frames": 24,
    "width": 576,
    "height": 320,
    "num_candidates": 2,
    "retry_attempts": 2,
    "reference_strength": 0.75,
    "prompt_strength": 0.85,
    "min_total_score": 0.45,
    "min_text_alignment": 0.35,
    "min_temporal_stability": 0.30
  }
}
```

## 说明

- 默认管线为 mock 模式，会在 `storage/outputs/` 下生成结构化输出文件。
- 在 `eager` 模式下，任务会以内联方式执行，不依赖 Redis 或 Celery。
- 第一条真实模型路径位于 [app/services/video_pipeline.py](b:\agent\MyCode\VidGen\app\services\video_pipeline.py) 和 [app/services/diffusers_loader.py](b:\agent\MyCode\VidGen\app\services\diffusers_loader.py)。
- 当前开源路径使用 `SDXL` 生成首帧，再用 `Stable Video Diffusion XT` 扩展成短视频。
- 如果提供了 `reference_image_path`，VidGen 会跳过首帧生成，直接将本地图片作为视频种子帧。
- 如果提供了 `reference_image_path` 且 `reference_strength < 1.0`，VidGen 会把 prompt 生成的首帧与参考图进行融合。
- `num_candidates` 允许 VidGen 采样多个候选结果，并选出评分最高的结果。
- `retry_attempts` 会在候选生成失败时重试，超过次数后任务失败。
- 当前候选排序综合了文图对齐、时序稳定性和运动幅度。如果安装了 `transformers`，文图对齐会优先使用多帧 CLIP 平均；否则退回启发式评分。
- 时序稳定性在安装 OpenCV 时会优先使用光流一致性；否则退回帧差评分。
- 候选评估现在会在可用时包含诊断信息，例如采样帧索引、帧差均值以及光流统计。
- 候选选择会先应用硬阈值（`min_total_score`、`min_text_alignment`、`min_temporal_stability`），如果所有候选都不达标，再回退到最高分策略。
- `GET /v1/jobs` 现在返回轻量级摘要列表，而 `GET /v1/jobs/{job_id}` 保留完整候选详情。
- 任务响应现在还会暴露 `selection_mode` 和聚合后的 `rejection_reason_counts`。
- 内置了生成配置档位：`strict`、`balanced`、`creative`。它们现在同时影响采样阈值以及 prompt 编排中的风格/镜头模板。请求级 `parameters` 会覆盖所选档位。
- Prompt 编排现在还会输出 `scene_template` 和受档位影响的 `shot_plan`，方便下游运动/控制模块消费结构化镜头节奏。
- 任务现在还会携带派生出的 `control_plan`，用于把镜头节奏映射到帧范围、镜头路径、pose/depth 提示类型、转场类型和运动标签。当前管线会把它写入输出元数据，并据此推导 `duration_sec` 和 `num_frames`。
- 管线现在还会输出 `adapter_plan`，把控制信号翻译成面向不同 provider 的 camera、pose、depth 和 transition 模块载荷。
- 面向 provider 的载荷现在通过适配器 stub 执行，并在 `storage/adapters/` 下产出标准化的中间件资源，包括 `skeleton_sequence`、`depth_manifest`、`camera_manifest` 和 `transition_manifest`。
- 这些中间件资源随后会作为 `generation_context` 被重新注入管线，用于增强 prompt 文本并调整解码 / 帧调度。
- 当 `reference_image_path` 可用且安装了可选预处理器时，适配器执行阶段会先生成真实的 OpenPose pose map 和 Depth Anything 深度资源，再退回到 stub 载荷。
- 当前模型注册表同时支持 `stable-video-diffusion-img2vid` 和 `stable-video-diffusion-flux`。
- 进行真实推理时，模型权重必须已经能通过 Hugging Face 授权或本地缓存被 `Diffusers` 访问到。

## 演示脚本

你可以通过下面的方式提交并轮询一个任务：

```bash
python scripts/submit_demo.py --prompt "A robot walking through a rainy neon city street at night"
```

如果你已经有种子帧：

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --reference-image-path "B:/path/to/reference.png"
```

如果你想启用候选采样：

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --num-candidates 3 --retry-attempts 2
```

如果你想使用更严格的质量档位：

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --generation-profile strict
```
