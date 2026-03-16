import json
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.core.model_registry import get_model_spec
from app.models.schemas import GenerationContext


class DiffusersUnavailableError(RuntimeError):
    pass


def _require_ml_stack() -> tuple[Any, dict[str, Any]]:
    try:
        import torch
        from diffusers import FluxPipeline, StableDiffusionXLPipeline, StableVideoDiffusionPipeline
    except ImportError as exc:  # pragma: no cover
        raise DiffusersUnavailableError(
            "Diffusers ML stack is not installed. Install with `pip install -e .[ml]`."
        ) from exc
    classes: dict[str, Any] = {
        "FluxPipeline": FluxPipeline,
        "StableDiffusionXLPipeline": StableDiffusionXLPipeline,
        "StableVideoDiffusionPipeline": StableVideoDiffusionPipeline,
    }
    try:  # pragma: no cover
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        classes["ControlNetModel"] = ControlNetModel
        classes["StableDiffusionXLControlNetPipeline"] = StableDiffusionXLControlNetPipeline
    except ImportError:
        pass
    return torch, classes


@lru_cache(maxsize=8)
def load_pipelines(
    model_name: str,
    generation_context_key: str | None = None,
    skip_image_pipeline: bool = False,
) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    settings = get_settings()
    torch, pipeline_classes = _require_ml_stack()
    FluxPipeline = pipeline_classes["FluxPipeline"]
    StableDiffusionXLPipeline = pipeline_classes["StableDiffusionXLPipeline"]
    StableVideoDiffusionPipeline = pipeline_classes["StableVideoDiffusionPipeline"]

    if spec.provider != "diffusers":
        raise DiffusersUnavailableError(f"No diffusers loader implemented for model '{spec.name}'.")

    dtype = torch.float16 if settings.device == "cuda" else torch.float32
    generation_context = _decode_generation_context_key(generation_context_key)
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
    video_pipe = StableVideoDiffusionPipeline.from_pretrained(
        spec.model_id,
        torch_dtype=dtype,
        cache_dir=str(settings.model_cache_dir),
    )
    if image_pipe is not None:
        image_pipe = image_pipe.to(settings.device)
    video_pipe = video_pipe.to(settings.device)
    return {"image": image_pipe, "video": video_pipe}


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
    if metadata.get("pose_artifact_paths") and settings.sdxl_openpose_controlnet_id:
        requested_ids.append(settings.sdxl_openpose_controlnet_id)
    if metadata.get("depth_artifact_paths") and settings.sdxl_depth_controlnet_id:
        requested_ids.append(settings.sdxl_depth_controlnet_id)
    if not requested_ids:
        return None

    controlnet_cls = pipeline_classes.get("ControlNetModel")
    controlnet_pipe_cls = pipeline_classes.get("StableDiffusionXLControlNetPipeline")
    if controlnet_cls is None or controlnet_pipe_cls is None:
        return None

    controlnets = [
        controlnet_cls.from_pretrained(controlnet_id, torch_dtype=dtype, cache_dir=cache_dir)
        for controlnet_id in requested_ids
    ]
    controlnet = controlnets[0] if len(controlnets) == 1 else controlnets
    return controlnet_pipe_cls.from_pretrained(
        model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )
