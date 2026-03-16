import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

from app.models.schemas import GenerationContext
from app.core.model_registry import get_model_spec


with patch.dict(
    sys.modules,
    {
        "pydantic_settings": SimpleNamespace(
            BaseSettings=object,
            SettingsConfigDict=lambda **kwargs: kwargs,
        )
    },
):
    diffusers_loader = importlib.import_module("app.services.diffusers_loader")


class FakePipeline:
    def __init__(self, model_id: str, kwargs: dict) -> None:
        self.model_id = model_id
        self.kwargs = kwargs
        self.device = None

    def to(self, device: str) -> "FakePipeline":
        self.device = device
        return self


class FakeFluxPipeline:
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> FakePipeline:
        return FakePipeline(model_id, kwargs)


class FakeSDXLPipeline:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> FakePipeline:
        cls.calls.append((model_id, kwargs))
        return FakePipeline(model_id, kwargs)


class FakeSVDPipeline:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> FakePipeline:
        cls.calls.append((model_id, kwargs))
        return FakePipeline(model_id, kwargs)


class FakeControlNetModel:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> str:
        cls.calls.append((model_id, kwargs))
        return f"controlnet:{model_id}"


class FakeSDXLControlNetPipeline:
    calls: list[tuple[str, dict]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> FakePipeline:
        cls.calls.append((model_id, kwargs))
        return FakePipeline(model_id, kwargs)


def test_load_pipelines_uses_sdxl_controlnet_when_conditioning_configured(monkeypatch) -> None:
    diffusers_loader.load_pipelines.cache_clear()
    FakeSDXLPipeline.calls.clear()
    FakeSVDPipeline.calls.clear()
    FakeControlNetModel.calls.clear()
    FakeSDXLControlNetPipeline.calls.clear()

    monkeypatch.setattr(
        diffusers_loader,
        "get_settings",
        lambda: SimpleNamespace(
            device="cpu",
            model_cache_dir="cache",
            sdxl_openpose_controlnet_id="openpose-id",
            sdxl_depth_controlnet_id="depth-id",
        ),
    )
    monkeypatch.setattr(
        diffusers_loader,
        "_require_ml_stack",
        lambda: (
            SimpleNamespace(float16="fp16", float32="fp32"),
            {
                "FluxPipeline": FakeFluxPipeline,
                "StableDiffusionXLPipeline": FakeSDXLPipeline,
                "StableVideoDiffusionPipeline": FakeSVDPipeline,
                "ControlNetModel": FakeControlNetModel,
                "StableDiffusionXLControlNetPipeline": FakeSDXLControlNetPipeline,
            },
        ),
    )

    context = GenerationContext(
        metadata={
            "pose_artifact_paths": ["pose.json"],
            "depth_artifact_paths": ["depth.json"],
        }
    )
    pipelines = diffusers_loader.load_pipelines(
        "stable-video-diffusion-img2vid",
        diffusers_loader.build_generation_context_key(context),
    )

    assert len(FakeControlNetModel.calls) == 2
    assert FakeSDXLControlNetPipeline.calls
    assert pipelines["image"].kwargs["controlnet"] == ["controlnet:openpose-id", "controlnet:depth-id"]
    assert pipelines["video"].device == "cpu"


def test_load_pipelines_falls_back_without_controlnet_config(monkeypatch) -> None:
    diffusers_loader.load_pipelines.cache_clear()
    FakeSDXLPipeline.calls.clear()
    FakeSVDPipeline.calls.clear()
    FakeSDXLControlNetPipeline.calls.clear()

    monkeypatch.setattr(
        diffusers_loader,
        "get_settings",
        lambda: SimpleNamespace(
            device="cpu",
            model_cache_dir="cache",
            sdxl_openpose_controlnet_id=None,
            sdxl_depth_controlnet_id=None,
        ),
    )
    monkeypatch.setattr(
        diffusers_loader,
        "_require_ml_stack",
        lambda: (
            SimpleNamespace(float16="fp16", float32="fp32"),
            {
                "FluxPipeline": FakeFluxPipeline,
                "StableDiffusionXLPipeline": FakeSDXLPipeline,
                "StableVideoDiffusionPipeline": FakeSVDPipeline,
            },
        ),
    )

    context = GenerationContext(metadata={"pose_artifact_paths": ["pose.json"]})
    pipelines = diffusers_loader.load_pipelines(
        "stable-video-diffusion-img2vid",
        diffusers_loader.build_generation_context_key(context),
    )

    assert not FakeSDXLControlNetPipeline.calls
    assert pipelines["image"].model_id == get_model_spec("stable-video-diffusion-img2vid").image_model_id


def test_load_pipelines_skips_image_pipeline_when_reference_path_exists(monkeypatch) -> None:
    diffusers_loader.load_pipelines.cache_clear()
    FakeSDXLPipeline.calls.clear()
    FakeSVDPipeline.calls.clear()

    monkeypatch.setattr(
        diffusers_loader,
        "get_settings",
        lambda: SimpleNamespace(
            device="cpu",
            model_cache_dir="cache",
            sdxl_openpose_controlnet_id=None,
            sdxl_depth_controlnet_id=None,
        ),
    )
    monkeypatch.setattr(
        diffusers_loader,
        "_require_ml_stack",
        lambda: (
            SimpleNamespace(float16="fp16", float32="fp32"),
            {
                "FluxPipeline": FakeFluxPipeline,
                "StableDiffusionXLPipeline": FakeSDXLPipeline,
                "StableVideoDiffusionPipeline": FakeSVDPipeline,
            },
        ),
    )

    pipelines = diffusers_loader.load_pipelines(
        "stable-video-diffusion-img2vid",
        None,
        skip_image_pipeline=True,
    )

    assert pipelines["image"] is None
    assert not FakeSDXLPipeline.calls
    assert FakeSVDPipeline.calls
