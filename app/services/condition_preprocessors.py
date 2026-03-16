import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class OpenPoseUnavailableError(RuntimeError):
    pass


class DepthEstimatorUnavailableError(RuntimeError):
    pass


class OpenPosePreprocessor:
    def process(self, image: Image.Image) -> Image.Image:
        detector = _load_openpose_detector()
        if detector is None:
            raise OpenPoseUnavailableError("OpenPose detector is unavailable.")
        try:
            result = detector(image)
        except Exception as exc:
            raise OpenPoseUnavailableError(f"OpenPose detector failed: {exc}") from exc
        if isinstance(result, Image.Image):
            return result.convert("RGB")
        return Image.fromarray(np.asarray(result)).convert("RGB")


class DepthAnythingPreprocessor:
    def process(self, image: Image.Image) -> tuple[Image.Image, np.ndarray | None]:
        estimator = _load_depth_estimator()
        if estimator is None:
            raise DepthEstimatorUnavailableError("Depth Anything estimator is unavailable.")
        try:
            prediction = estimator(image)
        except Exception as exc:
            raise DepthEstimatorUnavailableError(f"Depth Anything estimator failed: {exc}") from exc
        depth_image = _resolve_depth_image(prediction)
        depth_array = _resolve_depth_array(prediction)
        return depth_image.convert("RGB"), depth_array


@lru_cache(maxsize=1)
def _load_openpose_detector() -> Any | None:
    try:
        from controlnet_aux import OpenposeDetector
    except ImportError:
        return None

    model_id = os.getenv("VIDGEN_OPENPOSE_DETECTOR_ID", "lllyasviel/Annotators")
    cache_dir = os.getenv("VIDGEN_MODEL_CACHE_DIR")
    kwargs: dict[str, Any] = {}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return OpenposeDetector.from_pretrained(model_id, **kwargs)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _load_depth_estimator() -> Any | None:
    try:
        from transformers import pipeline
    except ImportError:
        return None

    model_id = os.getenv("VIDGEN_DEPTH_ANYTHING_MODEL_ID", "LiheYoung/depth-anything-small-hf")
    device = 0 if os.getenv("VIDGEN_DEVICE", "cpu") == "cuda" else -1
    try:
        return pipeline("depth-estimation", model=model_id, device=device)
    except Exception:
        return None


def _resolve_depth_image(prediction: Any) -> Image.Image:
    if isinstance(prediction, dict):
        depth_image = prediction.get("depth")
        if isinstance(depth_image, Image.Image):
            return depth_image
        predicted_depth = prediction.get("predicted_depth")
        if predicted_depth is not None:
            return Image.fromarray(_normalize_depth_array(predicted_depth))
    if isinstance(prediction, Image.Image):
        return prediction
    return Image.fromarray(_normalize_depth_array(prediction))


def _resolve_depth_array(prediction: Any) -> np.ndarray | None:
    if isinstance(prediction, dict):
        predicted_depth = prediction.get("predicted_depth")
        if predicted_depth is not None:
            return _to_numpy(predicted_depth)
    if isinstance(prediction, Image.Image):
        return np.asarray(prediction)
    if prediction is None:
        return None
    return _to_numpy(prediction)


def _normalize_depth_array(value: Any) -> np.ndarray:
    array = _to_numpy(value).astype("float32")
    if array.ndim == 3:
        array = array.squeeze()
    array = array - float(array.min())
    max_value = float(array.max())
    if max_value > 0:
        array = array / max_value
    return (array * 255).clip(0, 255).astype("uint8")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def clear_preprocessor_caches() -> None:
    _load_openpose_detector.cache_clear()
    _load_depth_estimator.cache_clear()
