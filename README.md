# VidGen

VidGen is a text-to-video MVP scaffold built around open-source model orchestration, task execution, and future fine-tuning workflows.

## Current status

The repository now includes the first implementation slice for phase 1:

- FastAPI service for job submission and status queries
- Celery worker integration for asynchronous generation
- Local `eager` task mode for development without Redis
- Prompt orchestration layer
- Swappable video pipeline interface
- Local file-based job store for early iteration
- Diffusers-based open-source pipeline hook for `SDXL -> SVD`
- Architecture document for the full roadmap

## Project layout

- `app/`: API, core config, services, worker, and task code
- `docs/architecture.md`: end-to-end architecture and phased roadmap
- `configs/`: reserved for training and inference configs
- `pipelines/`: reserved for reusable generation pipelines
- `training/`: reserved for LoRA / temporal fine-tuning code
- `evaluation/`: reserved for quality evaluation scripts
- `tests/`: basic API tests

## Quick start

1. Create a virtual environment with Python 3.11+.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Copy environment variables:

```bash
copy .env.example .env
```

4. Keep `VIDGEN_TASK_MODE=eager` for local-only startup.
5. Run the API:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

6. Optional: switch to Celery mode by setting `VIDGEN_TASK_MODE=celery`, start Redis, then run the worker:

```bash
celery -A app.workers.celery_app worker --loglevel=INFO
```

7. Optional: install the ML stack and switch off mock mode:

```bash
pip install -e .[ml]
```

Then set:

```bash
VIDGEN_USE_MOCK_PIPELINE=false
VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid
```

If you want SDXL first-frame generation to attach real pose/depth ControlNet branches when middleware artifacts are present, also set:

```bash
VIDGEN_SDXL_OPENPOSE_CONTROLNET_ID=<your-openpose-controlnet-id>
VIDGEN_SDXL_DEPTH_CONTROLNET_ID=<your-depth-controlnet-id>
```

If you want the adapter execution stage to produce real OpenPose / Depth Anything preprocessing assets from a local reference image, also set:

```bash
VIDGEN_OPENPOSE_DETECTOR_ID=lllyasviel/Annotators
VIDGEN_DEPTH_ANYTHING_MODEL_ID=LiheYoung/depth-anything-small-hf
```

Or switch the first-frame image backbone to `FLUX.1-dev`:

```bash
VIDGEN_DEFAULT_MODEL=stable-video-diffusion-flux
```

Threshold defaults can also be set globally in `.env`:

```bash
VIDGEN_DEFAULT_MIN_TOTAL_SCORE=0.45
VIDGEN_DEFAULT_MIN_TEXT_ALIGNMENT=0.35
VIDGEN_DEFAULT_MIN_TEMPORAL_STABILITY=0.30
VIDGEN_DEFAULT_GENERATION_PROFILE=balanced
```

## Local Demo Setup

For a reproducible local demo setup, use the `VidGen` conda environment and keep the model directories inside the repository.

Recommended activation flow:

```bash
conda activate VidGen
cd /d b:\agent\MyCode\VidGen
```

Recommended real local inference env vars:

```bash
set VIDGEN_USE_MOCK_PIPELINE=false
set VIDGEN_TASK_MODE=eager
set VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid
set VIDGEN_DEVICE=cpu
```

Project-local model directories:

```text
storage/models/stable-diffusion-xl-base-1.0
storage/models/stable-video-diffusion-img2vid-xt
storage/models/FLUX.1-dev
```

The current model registry prefers these local directories first and falls back to Hugging Face ids only if the local folders do not exist.

Minimal standalone model checks:

```bash
python scripts/test_local_sdxl_load.py --skip-infer
python scripts/test_local_svd_load.py --skip-infer
```

If you want a full smoke test:

```bash
python scripts/test_local_sdxl_load.py
python scripts/test_local_svd_load.py
```

## Real Job Demo

If you already have a local reference image, place it somewhere under `storage/` or another accessible local path.

Example direct local job run:

```bash
python - <<'PY'
from app.models.schemas import GenerateVideoRequest
from app.services.job_service import JobService

service = JobService()
job = service.create_job(
    GenerateVideoRequest(
        prompt="A robot walking forward in a city street at night",
        reference_image_path="storage/verification-reference.png",
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

Outputs are written under:

```text
storage/jobs/
storage/outputs/
storage/adapters/
```

For a finished job, inspect:

```text
storage/jobs/<job_id>.json
storage/outputs/<job_id>.json
storage/outputs/<job_id>.png
storage/outputs/<job_id>.mp4
```

## Local API Demo

Start the API locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Useful demo endpoints:

- `GET /health`
- `POST /v1/videos:generate`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`

You can also use the demo script:

```bash
python scripts/submit_demo.py --prompt "A robot walking through a rainy neon city street at night" --reference-image-path "storage/verification-reference.png"
```

## Control Regression

The repository includes a minimal control regression script for comparing a baseline run against a pose-depth controlled run.

Regression case definitions live in:

```text
tests/fixtures/regression_cases.json
```

By default the script writes its summary to:

```text
storage/regression/control-regression-summary.json
```

Run it from the project root:

```bash
python -m scripts.run_control_regression
```

The summary contains, for each case:

- `baseline`
- `pose_depth`
- `assertions`

The most useful fields to inspect are:

- `routing_differs_from_baseline`
- `controlled_has_pose`
- `controlled_has_depth`
- `controlled_video_conditioning_used`
- `score_delta`

If a run fails, inspect:

```text
storage/jobs/<job_id>.json
storage/outputs/<job_id>.json
terminal traceback
```

## API endpoints

- `GET /health`
- `POST /v1/videos:generate`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`

## Example request

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

## Notes

- The default pipeline is mock mode and writes a structured output file under `storage/outputs/`.
- In `eager` mode, jobs execute inline and do not require Redis or Celery.
- The first real-model path is isolated in [app/services/video_pipeline.py](b:\agent\MyCode\VidGen\app\services\video_pipeline.py) and [app/services/diffusers_loader.py](b:\agent\MyCode\VidGen\app\services\diffusers_loader.py).
- The current open-source path uses `SDXL` to create an initial frame, then `Stable Video Diffusion XT` to expand it into a short clip.
- If `reference_image_path` is provided, VidGen skips first-frame generation and uses the local image as the video seed frame.
- If `reference_image_path` is provided with `reference_strength < 1.0`, VidGen blends the prompt-generated first frame with the reference image.
- `num_candidates` lets VidGen sample multiple candidates and select the highest-scored result.
- `retry_attempts` retries failed candidate generation attempts before failing the job.
- Candidate ranking currently combines text-image alignment, temporal stability, and motion amplitude. If `transformers` is available, text alignment prefers multi-frame CLIP averaging; otherwise it falls back to heuristics.
- Temporal stability now prefers optical-flow consistency when OpenCV is installed; otherwise it falls back to frame-diff scoring.
- Candidate evaluations now include diagnostics such as sampled frame indices, frame-diff averages, and optical-flow statistics when available.
- Candidate selection uses hard thresholds first (`min_total_score`, `min_text_alignment`, `min_temporal_stability`), then falls back to best-score selection if all candidates fail.
- `GET /v1/jobs` now returns a lightweight summary list, while `GET /v1/jobs/{job_id}` keeps full candidate detail.
- Job responses now also expose `selection_mode` and aggregated `rejection_reason_counts`.
- Generation profiles are built in: `strict`, `balanced`, `creative`. They now affect both sampling thresholds and prompt orchestration style/camera templates. Request-level `parameters` override the chosen profile.
- Prompt orchestration now also emits `scene_template` and a profile-shaped `shot_plan`, so downstream motion/control modules can consume structured camera beats.
- Jobs now also carry a derived `control_plan`, which maps shot beats to frame ranges, camera paths, pose/depth hint types, transition types, and motion labels. The current pipeline stores it in output metadata and uses it to derive `duration_sec` and `num_frames`.
- The pipeline now also emits an `adapter_plan`, which translates control signals into provider-specific payloads for camera, pose, depth, and transition modules.
- Provider-specific payloads are now executed through adapter stubs, which emit standardized middleware artifacts under `storage/adapters/`, including `skeleton_sequence`, `depth_manifest`, `camera_manifest`, and `transition_manifest`.
- Middleware artifacts are now consumed back into the pipeline as a `generation_context`, which can augment prompt text and adjust decode/frame scheduling.
- When `reference_image_path` is available and the optional preprocessors are installed, adapter execution now emits real OpenPose pose maps and Depth Anything depth assets before falling back to stub payloads.
- The model registry currently supports both `stable-video-diffusion-img2vid` and `stable-video-diffusion-flux`.
- Model weights must already be accessible to `Diffusers` through Hugging Face auth/local cache for real inference.

## Demo script

You can submit and poll a job with:

```bash
python scripts/submit_demo.py --prompt "A robot walking through a rainy neon city street at night"
```

If you already have a seed frame:

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --reference-image-path "B:/path/to/reference.png"
```

If you want candidate sampling:

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --num-candidates 3 --retry-attempts 2
```

If you want a stricter quality profile:

```bash
python scripts/submit_demo.py --prompt "A cinematic robot scene" --generation-profile strict
```
