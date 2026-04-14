import json
import logging
from typing import Any

from app.core.config import get_settings
from app.core.model_registry import get_model_spec, LOCAL_SDXL_OPENPOSE_CONTROLNET, LOCAL_SDXL_DEPTH_CONTROLNET
from app.models.schemas import GenerationContext

logger = logging.getLogger(__name__)


#负责真实模型加载，包含 SDXL、FLUX、SVD 和可选 ControlNet 支持。
class DiffusersUnavailableError(RuntimeError):
    pass


# Module-level cache for loaded pipelines.  Unlike lru_cache, we can
# explicitly clear it to free GPU VRAM before loading a new configuration.
_pipeline_cache: dict[str, dict[str, Any]] = {}


def _normalize_optional_model_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "null", "disabled", "false", "off", "0"}:
        return None
    return normalized


def _require_ml_stack() -> tuple[Any, dict[str, Any]]:
    """Lazy-import torch and diffusers pipeline classes."""
    import torch
    from diffusers import FluxPipeline, StableDiffusionXLPipeline, StableVideoDiffusionPipeline
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    pipeline_classes = {
        "FluxPipeline": FluxPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
        "StableVideoDiffusionPipeline": StableVideoDiffusionPipeline,
        "ControlNetModel": ControlNetModel,
        "StableDiffusionXLControlNetPipeline": StableDiffusionXLControlNetPipeline,
    }
    return torch, pipeline_classes


def _cache_key(
    model_name: str,
    generation_context_key: str | None,
    skip_image_pipeline: bool,
    skip_video_pipeline: bool,
    device_override: str | None,
) -> str:
    return f"{model_name}|{generation_context_key}|{skip_image_pipeline}|{skip_video_pipeline}|{device_override}"


def clear_pipeline_cache() -> None:
    """Unload all cached pipelines from GPU and clear the cache."""
    import gc

    for entry in _pipeline_cache.values():
        for pipe in (entry.get("image"), entry.get("video")):
            if pipe is not None:
                try:
                    # Move model components to CPU first, then delete
                    if hasattr(pipe, "to"):
                        pipe.to("cpu")
                except Exception:
                    pass
    _pipeline_cache.clear()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def load_pipelines(
    model_name: str,
    generation_context_key: str | None = None,
    skip_image_pipeline: bool = False,
    skip_video_pipeline: bool = False,
    device_override: str | None = None,
) -> dict[str, Any]:
    key = _cache_key(
        model_name,
        generation_context_key,
        skip_image_pipeline,
        skip_video_pipeline,
        device_override,
    )
    if key in _pipeline_cache:
        return _pipeline_cache[key]

    # Before loading a new pipeline configuration, evict existing entries to
    # free GPU VRAM.  On a 6 GB card we cannot keep two full pipelines alive.
    clear_pipeline_cache()

    spec = get_model_spec(model_name)
    settings = get_settings()
    torch, pipeline_classes = _require_ml_stack()
    FluxPipeline = pipeline_classes["FluxPipeline"]
    StableDiffusionXLPipeline = pipeline_classes["StableDiffusionXLPipeline"]
    StableVideoDiffusionPipeline = pipeline_classes["StableVideoDiffusionPipeline"]

    if spec.provider != "diffusers":
        raise DiffusersUnavailableError(f"No diffusers loader implemented for model '{spec.name}'.")

    target_device = device_override or settings.device
    dtype = torch.float16 if target_device == "cuda" else torch.float32
    generation_context = _decode_generation_context_key(generation_context_key)
    logger.info(
        "load_pipelines model=%s device=%s skip_image=%s skip_video=%s context=%s",
        model_name,
        target_device,
        skip_image_pipeline,
        skip_video_pipeline,
        generation_context is not None,
    )
    image_pipe = None
    if not skip_image_pipeline:
        image_pipe = _load_image_pipeline(
            spec.image_provider,
            spec.image_model_id,
            dtype=dtype,
            cache_dir=str(settings.model_cache_dir),
            flux_cls=FluxPipeline,
            sdxl_cls=StableDiffusionXLPipeline,
            pipeline_classes=pipeline_classes,
            settings=settings,
            generation_context=generation_context,
        )
    video_pipe = None
    if not skip_video_pipeline:
        video_pipe = StableVideoDiffusionPipeline.from_pretrained(
            spec.model_id,
            torch_dtype=dtype,
            cache_dir=str(settings.model_cache_dir),
        )
    if image_pipe is not None:
        if target_device == "cuda":
            image_pipe.enable_vae_slicing()
            try:
                image_pipe.enable_sequential_cpu_offload()
            except Exception:
                image_pipe = image_pipe.to(target_device)
        else:
            image_pipe = image_pipe.to(target_device)
    if video_pipe is not None:
        if target_device == "cuda":
            try:
                video_pipe.enable_sequential_cpu_offload()
            except Exception:
                video_pipe = video_pipe.to(target_device)
        else:
            video_pipe = video_pipe.to(target_device)

    result = {"image": image_pipe, "video": video_pipe}
    _pipeline_cache[key] = result
    return result


def _load_image_pipeline(
    provider: str | None,
    model_id: str | None,
    *,
    dtype: Any,
    cache_dir: str,
    flux_cls: Any,
    sdxl_cls: Any,
    pipeline_classes: dict[str, Any],
    settings: Any,
    generation_context: GenerationContext | None,
) -> Any:
    if provider is None or model_id is None:
        raise DiffusersUnavailableError("Image pipeline configuration is incomplete.")

    if provider == "diffusers-sdxl":
        controlnet_pipe = _load_sdxl_controlnet_pipeline(
            model_id=model_id,
            dtype=dtype,
            cache_dir=cache_dir,
            pipeline_classes=pipeline_classes,
            settings=settings,
            generation_context=generation_context,
        )
        if controlnet_pipe is not None:
            return controlnet_pipe
        return sdxl_cls.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    if provider == "diffusers-flux":
        return flux_cls.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
    raise DiffusersUnavailableError(f"Unsupported image provider '{provider}'.")


def build_generation_context_key(generation_context: GenerationContext | None) -> str | None:
    if generation_context is None:
        return None
    dump = getattr(generation_context, "model_dump", None)
    payload = dump() if callable(dump) else generation_context.dict()
    return json.dumps(payload, sort_keys=True)


def _decode_generation_context_key(generation_context_key: str | None) -> GenerationContext | None:
    if not generation_context_key:
        return None
    validate_json = getattr(GenerationContext, "model_validate_json", None)
    if callable(validate_json):
        return validate_json(generation_context_key)
    return GenerationContext.parse_raw(generation_context_key)


def _load_sdxl_controlnet_pipeline(
    *,
    model_id: str,
    dtype: Any,
    cache_dir: str,
    pipeline_classes: dict[str, Any],
    settings: Any,
    generation_context: GenerationContext | None,
) -> Any | None:
    if generation_context is None:
        return None

    metadata = generation_context.metadata
    requested_ids: list[str] = []
    pose_assets = metadata.get("pose_asset_images")
    depth_assets = metadata.get("depth_asset_images")
    openpose_controlnet_id = _normalize_optional_model_id(settings.sdxl_openpose_controlnet_id)
    depth_controlnet_id = _normalize_optional_model_id(settings.sdxl_depth_controlnet_id)
    if pose_assets and openpose_controlnet_id:
        controlnet_id = openpose_controlnet_id
        if LOCAL_SDXL_OPENPOSE_CONTROLNET.exists():
            controlnet_id = str(LOCAL_SDXL_OPENPOSE_CONTROLNET)
        requested_ids.append(controlnet_id)
    if depth_assets and depth_controlnet_id:
        controlnet_id = depth_controlnet_id
        if LOCAL_SDXL_DEPTH_CONTROLNET.exists():
            controlnet_id = str(LOCAL_SDXL_DEPTH_CONTROLNET)
        requested_ids.append(controlnet_id)
    logger.info(
        "sdxl_controlnet selection pose_assets=%s depth_assets=%s openpose_enabled=%s depth_enabled=%s requested=%s",
        bool(pose_assets),
        bool(depth_assets),
        bool(openpose_controlnet_id),
        bool(depth_controlnet_id),
        requested_ids,
    )
    if not requested_ids:
        return None

    controlnet_cls = pipeline_classes.get("ControlNetModel")
    controlnet_pipe_cls = pipeline_classes.get("StableDiffusionXLControlNetPipeline")
    if controlnet_cls is None or controlnet_pipe_cls is None:
        return None

    controlnets = []
    for controlnet_id in requested_ids:
        cn = _load_controlnet_model(controlnet_cls, controlnet_id, dtype, cache_dir)
        if cn is not None:
            controlnets.append(cn)
    if not controlnets:
        return None
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets
    return controlnet_pipe_cls.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )


def _load_controlnet_model(
    controlnet_cls: Any,
    controlnet_id: str,
    dtype: Any,
    cache_dir: str,
) -> Any | None:
    """Try loading a ControlNet model with multiple fallback strategies."""
    # Strategy 1: fp16 variant (e.g. depth model ships diffusion_pytorch_model.fp16.safetensors)
    try:
        return controlnet_cls.from_pretrained(
            controlnet_id, torch_dtype=dtype, cache_dir=cache_dir, variant="fp16"
        )
    except Exception:
        pass
    # Strategy 2: default (no variant) — works if diffusion_pytorch_model.safetensors exists
    try:
        return controlnet_cls.from_pretrained(
            controlnet_id, torch_dtype=dtype, cache_dir=cache_dir
        )
    except Exception:
        pass
    # Strategy 3: scan for non-standard .safetensors and load via state dict
    try:
        from pathlib import Path as _P
        model_dir = _P(controlnet_id)
        if model_dir.is_dir():
            safetensor_files = sorted(model_dir.glob("*.safetensors"))
            for sf in safetensor_files:
                if sf.name.startswith("diffusion_pytorch_model"):
                    continue  # already tried above
                try:
                    from safetensors.torch import load_file
                    state_dict = load_file(str(sf))
                    config = controlnet_cls.load_config(model_dir)
                    cn = controlnet_cls.from_config(config, torch_dtype=dtype)
                    cn.load_state_dict(state_dict, strict=False)
                    return cn
                except Exception:
                    continue
    except Exception:
        pass
    return None
