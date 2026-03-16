from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: str
    provider: str
    model_id: str
    image_model_id: str | None = None
    image_provider: str | None = None
    revision: str | None = None
    torch_dtype: str = "float16"
    default_num_inference_steps: int = 25
    default_image_steps: int = 30
    supports_image_to_video: bool = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MODEL_ROOT = PROJECT_ROOT / "storage" / "models"
LOCAL_SDXL_BASE = LOCAL_MODEL_ROOT / "stable-diffusion-xl-base-1.0"
LOCAL_SVD_IMG2VID_XT = LOCAL_MODEL_ROOT / "stable-video-diffusion-img2vid-xt"
LOCAL_FLUX_DEV = LOCAL_MODEL_ROOT / "FLUX.1-dev"


def _prefer_local_model(local_path: Path, remote_id: str) -> str:
    return str(local_path) if local_path.exists() else remote_id


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mock-svd": ModelSpec(
        name="mock-svd",
        task="image-to-video",
        provider="mock",
        model_id="mock/mock-svd",
        supports_image_to_video=True,
    ),
    "stable-video-diffusion-img2vid": ModelSpec(
        name="stable-video-diffusion-img2vid",
        task="image-to-video",
        provider="diffusers",
        model_id=_prefer_local_model(
            LOCAL_SVD_IMG2VID_XT,
            "stabilityai/stable-video-diffusion-img2vid-xt",
        ),
        image_model_id=_prefer_local_model(
            LOCAL_SDXL_BASE,
            "stabilityai/stable-diffusion-xl-base-1.0",
        ),
        image_provider="diffusers-sdxl",
        supports_image_to_video=True,
        default_num_inference_steps=25,
    ),
    "stable-video-diffusion-flux": ModelSpec(
        name="stable-video-diffusion-flux",
        task="image-to-video",
        provider="diffusers",
        model_id=_prefer_local_model(
            LOCAL_SVD_IMG2VID_XT,
            "stabilityai/stable-video-diffusion-img2vid-xt",
        ),
        image_model_id=_prefer_local_model(
            LOCAL_FLUX_DEV,
            "black-forest-labs/FLUX.1-dev",
        ),
        image_provider="diffusers-flux",
        supports_image_to_video=True,
        default_num_inference_steps=25,
        default_image_steps=28,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{name}'. Supported models: {supported}") from exc
