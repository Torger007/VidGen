from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="VIDGEN_", extra="ignore")

    app_name: str = "VidGen"
    environment: str = "local"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "sqlite:///./vidgen.db"
    storage_root: Path = Field(default=Path("./storage"))
    task_mode: str = "eager"
    use_mock_pipeline: bool = True
    default_model: str = "mock-svd"
    allow_local_reference_images: bool = True
    adapter_artifact_root: Path = Field(default=Path("./storage/adapters"))
    model_cache_dir: Path = Field(default=Path("./storage/model-cache"))
    sdxl_openpose_controlnet_id: str | None = None
    sdxl_depth_controlnet_id: str | None = None
    ffmpeg_binary: str = "ffmpeg"
    device: str = "cuda"
    output_fps: int = 8
    output_frames: int = 24
    output_width: int = 576
    output_height: int = 320
    default_min_total_score: float = 0.45
    default_min_text_alignment: float = 0.35
    default_min_temporal_stability: float = 0.30
    default_generation_profile: str = "balanced"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    (settings.storage_root / "jobs").mkdir(parents=True, exist_ok=True)
    (settings.storage_root / "outputs").mkdir(parents=True, exist_ok=True)
    settings.adapter_artifact_root.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
