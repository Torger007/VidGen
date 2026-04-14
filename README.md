# VidGen

VidGen is a local-first video generation prototype built around FastAPI, task orchestration, open-source image/video pipelines, and optional pose/depth control signals.

## Current Status

The repository has already validated these paths:

- Local `SDXL` and `SVD` model directories can be discovered and loaded with `--skip-infer`.
- OpenPose and Depth Anything preprocessors can initialize offline and run on a single test image.
- A minimal real generation flow without `control_plan` succeeds and produces `json/png/mp4`.
- The API flow also succeeds when `control_plan` is explicitly disabled.

The main unfinished area is full real generation with `control_plan` enabled. The current blocker is not the API layer itself, but Windows memory / pagefile pressure during `SDXL` image-pipeline loading.

## Project Layout

- `app/`: FastAPI app, config, services, tasks, workers
- `tests/`: pytest suite
- `scripts/`: local smoke tests, demos, regression helpers
- `docs/`: architecture and deployment notes
- `storage/`: local models, caches, jobs, outputs, adapter artifacts

## Environment

Recommended local environment:

```powershell
conda activate VidGen
cd /d B:\agent\MyCode\VidGen
```

Install dependencies:

```bash
pip install -e .[dev]
pip install -e .[ml]
```

Recommended runtime settings:

```bash
VIDGEN_USE_MOCK_PIPELINE=false
VIDGEN_TASK_MODE=eager
VIDGEN_DEFAULT_MODEL=stable-video-diffusion-img2vid
HF_HUB_OFFLINE=1
```

Default device handling is intentionally conservative. If you want GPU inference, set:

```bash
VIDGEN_DEVICE=cuda
```

## Local Models

Expected local model directories:

```text
storage/models/stable-diffusion-xl-base-1.0
storage/models/stable-video-diffusion-img2vid-xt
storage/models/controlnet-openpose-sdxl-1.0
storage/models/controlnet-depth-sdxl-1.0
storage/models/FLUX.1-dev
```

Expected local cache directory:

```text
storage/model-cache
```

## Safe Validation Order

Run lightweight checks first:

```bash
ruff check .
pytest tests/test_diffusers_loader.py tests/test_api_models.py -q
python scripts/test_local_sdxl_load.py --device cpu --skip-infer
python scripts/test_local_svd_load.py --device cpu --skip-infer
```

If you only want to verify the real end-to-end path, prefer the minimal API demo without control signals:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/submit_demo.py --base-url http://127.0.0.1:8000 --disable-control-plan
```

Or use:

```powershell
.\scripts\run_demo.ps1
```

## Control Plan Notes

`GenerateVideoRequest` now supports:

- `enable_control_plan=true` by default
- `enable_control_plan=false` to force the minimal real generation path

ControlNet is only activated when real pose/depth image assets exist in `generation_context.metadata`.

For debugging, the following values are treated as disabled ControlNet IDs:

- `none`
- `disabled`
- `off`
- `false`
- `0`

This matters because an empty string does not reliably override values coming from `.env`.

## Outputs

Generated artifacts are written to:

```text
storage/jobs/
storage/outputs/
storage/adapters/
```

Typical files:

```text
storage/jobs/<job_id>.json
storage/outputs/<job_id>.json
storage/outputs/<job_id>.png
storage/outputs/<job_id>.mp4
```

## Known Limitation

On Windows, full `SDXL + ControlNet` image-pipeline loading may fail before video generation starts if system memory commit is too low. The observed failure pattern is:

- Windows `os error 1455`
- `MemoryError`
- sometimes `torch_cpu.dll` or `c10.dll` access violations

If this happens:

1. Increase the Windows pagefile size.
2. Close high-memory applications and restart the terminal / IDE.
3. Re-test `SDXL` loading before retrying full `control_plan` generation.

More detailed status and next-step planning is tracked in [docs/current-status-and-local-deploy-plan.md](docs/current-status-and-local-deploy-plan.md).
