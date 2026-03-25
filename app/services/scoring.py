from typing import Any

import numpy as np
from PIL import Image

from app.models.schemas import GenerationParameters, PromptBundle


#对生成候选做评分，包括 text alignment、temporal stability、motion score。
class CandidateScorer:
    def __init__(self) -> None:
        self._clip_bundle: tuple[Any, Any] | None = None
        self._clip_available = True
        self._cv2_module: Any | None = None
        self._cv2_checked = False

    def evaluate(
        self,
        *,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        used_reference_image: bool,
        candidate_index: int,
        initial_frame: Image.Image | None = None,
        frames: list[Image.Image] | None = None,
    ) -> dict[str, Any]:
        text_alignment, alignment_method, sampled_frame_indices = self._text_alignment(
            prompt_bundle=prompt_bundle,
            parameters=parameters,
            initial_frame=initial_frame,
            frames=frames,
        )
        temporal_stability, temporal_method, temporal_diagnostics = self._temporal_stability(frames)
        motion_score, motion_diff_mean = self._motion_score(frames)
        reference_score = parameters.reference_strength if used_reference_image else 0.0
        prompt_score = parameters.prompt_strength
        candidate_bias = max(0.0, 0.1 - (candidate_index * 0.02))

        total_score = (
            text_alignment * 0.35
            + temporal_stability * 0.25
            + motion_score * 0.15
            + min(parameters.guidance_scale / 10.0, 1.0) * 0.1
            + min(prompt_score, 1.0) * 0.05
            + min(reference_score, 1.0) * 0.05
            + candidate_bias * 0.05
        )

        rejection_reasons = self._rejection_reasons(
            total_score=total_score,
            text_alignment=text_alignment,
            temporal_stability=temporal_stability,
            parameters=parameters,
        )

        return {
            "total_score": round(min(total_score, 0.99), 4),
            "metrics": {
                "text_alignment": round(text_alignment, 4),
                "temporal_stability": round(temporal_stability, 4),
                "motion_score": round(motion_score, 4),
                "prompt_score": round(min(prompt_score, 1.0), 4),
                "reference_score": round(min(reference_score, 1.0), 4),
                "candidate_bias": round(candidate_bias, 4),
            },
            "method": {
                "text_alignment": alignment_method,
                "temporal_stability": temporal_method,
                "motion_score": "frame-diff",
            },
            "diagnostics": {
                "optical_flow_mean": temporal_diagnostics.get("optical_flow_mean"),
                "optical_flow_std": temporal_diagnostics.get("optical_flow_std"),
                "frame_diff_mean": temporal_diagnostics.get("frame_diff_mean"),
                "motion_diff_mean": motion_diff_mean,
                "sampled_frame_indices": sampled_frame_indices,
            },
            "passed_thresholds": not rejection_reasons,
            "rejection_reasons": rejection_reasons,
        }

    def _text_alignment(
        self,
        *,
        prompt_bundle: PromptBundle,
        parameters: GenerationParameters,
        initial_frame: Image.Image | None,
        frames: list[Image.Image] | None,
    ) -> tuple[float, str, list[int]]:
        clip_frames, sampled_frame_indices = self._select_clip_frames(initial_frame, frames)
        if clip_frames:
            clip_score = self._clip_alignment(prompt_bundle, clip_frames)
            if clip_score is not None:
                return clip_score, "clip-avg", sampled_frame_indices

        heuristic = 0.0
        heuristic += min(parameters.guidance_scale / 10.0, 1.0) * 0.45
        heuristic += min(parameters.prompt_strength, 1.0) * 0.25
        heuristic += 0.15 if "cinematic" in prompt_bundle.style.lower() else 0.05
        heuristic += 0.1 if "slow" in prompt_bundle.camera.lower() else 0.04
        return min(heuristic, 0.95), "heuristic", sampled_frame_indices

    def _temporal_stability(self, frames: list[Image.Image] | None) -> tuple[float, str, dict[str, float | None]]:
        if not frames or len(frames) < 2:
            return 0.6, "frame-diff", {
                "optical_flow_mean": None,
                "optical_flow_std": None,
                "frame_diff_mean": None,
            }
        flow_score, flow_diagnostics = self._optical_flow_stability(frames)
        if flow_score is not None:
            return flow_score, "optical-flow", flow_diagnostics
        diffs = []
        for previous, current in zip(frames, frames[1:]):
            prev_arr = np.asarray(previous.convert("RGB"), dtype=np.float32)
            cur_arr = np.asarray(current.convert("RGB"), dtype=np.float32)
            diffs.append(np.mean(np.abs(cur_arr - prev_arr)) / 255.0)
        mean_diff = float(np.mean(diffs))
        return max(0.0, min(1.0, 1.0 - (mean_diff * 1.8))), "frame-diff", {
            "optical_flow_mean": None,
            "optical_flow_std": None,
            "frame_diff_mean": mean_diff,
        }

    def _motion_score(self, frames: list[Image.Image] | None) -> tuple[float, float | None]:
        if not frames or len(frames) < 2:
            return 0.5, None
        diffs = []
        for previous, current in zip(frames, frames[1:]):
            prev_arr = np.asarray(previous.convert("L"), dtype=np.float32)
            cur_arr = np.asarray(current.convert("L"), dtype=np.float32)
            diffs.append(np.mean(np.abs(cur_arr - prev_arr)) / 255.0)
        mean_diff = float(np.mean(diffs))
        target = 0.08
        score = 1.0 - min(abs(mean_diff - target) / target, 1.0)
        return max(0.0, min(1.0, score)), mean_diff

    def _optical_flow_stability(self, frames: list[Image.Image]) -> tuple[float | None, dict[str, float | None]]:
        cv2 = self._load_cv2()
        if cv2 is None:
            return None, {
                "optical_flow_mean": None,
                "optical_flow_std": None,
                "frame_diff_mean": None,
            }
        magnitudes = []
        for previous, current in zip(frames, frames[1:]):
            prev_arr = np.asarray(previous.convert("L"), dtype=np.uint8)
            cur_arr = np.asarray(current.convert("L"), dtype=np.uint8)
            flow = cv2.calcOpticalFlowFarneback(
                prev_arr,
                cur_arr,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(float(np.mean(magnitude)))
        if not magnitudes:
            return None, {
                "optical_flow_mean": None,
                "optical_flow_std": None,
                "frame_diff_mean": None,
            }
        mean_magnitude = float(np.mean(magnitudes))
        std_magnitude = float(np.std(magnitudes))
        if mean_magnitude <= 1e-6:
            return 0.75, {
                "optical_flow_mean": mean_magnitude,
                "optical_flow_std": std_magnitude,
                "frame_diff_mean": None,
            }
        normalized = std_magnitude / (mean_magnitude + 1e-6)
        return max(0.0, min(1.0, 1.0 - min(normalized, 1.0))), {
            "optical_flow_mean": mean_magnitude,
            "optical_flow_std": std_magnitude,
            "frame_diff_mean": None,
        }

    def _clip_alignment(self, prompt_bundle: PromptBundle, images: list[Image.Image]) -> float | None:
        bundle = self._load_clip_bundle()
        if bundle is None:
            return None
        processor, model = bundle
        prompt = ", ".join(
            [
                prompt_bundle.subject,
                prompt_bundle.scene,
                prompt_bundle.style,
                prompt_bundle.action,
                prompt_bundle.camera,
            ]
        )
        inputs = processor(text=[prompt] * len(images), images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        value = logits.diag().float().mean().item()
        normalized = 1.0 / (1.0 + np.exp(-(value / 10.0)))
        return float(max(0.0, min(1.0, normalized)))

    def _select_clip_frames(
        self,
        initial_frame: Image.Image | None,
        frames: list[Image.Image] | None,
    ) -> tuple[list[Image.Image], list[int]]:
        if frames:
            if len(frames) <= 3:
                return [frame.convert("RGB") for frame in frames], list(range(len(frames)))
            middle = len(frames) // 2
            indices = [0, middle, len(frames) - 1]
            return [frames[index].convert("RGB") for index in indices], indices
        if initial_frame is not None:
            return [initial_frame.convert("RGB")], [0]
        return [], []

    def _rejection_reasons(
        self,
        *,
        total_score: float,
        text_alignment: float,
        temporal_stability: float,
        parameters: GenerationParameters,
    ) -> list[str]:
        reasons: list[str] = []
        if total_score < parameters.min_total_score:
            reasons.append("total_score_below_threshold")
        if text_alignment < parameters.min_text_alignment:
            reasons.append("text_alignment_below_threshold")
        if temporal_stability < parameters.min_temporal_stability:
            reasons.append("temporal_stability_below_threshold")
        return reasons

    def _load_clip_bundle(self) -> tuple[Any, Any] | None:
        if self._clip_bundle is not None:
            return self._clip_bundle
        if not self._clip_available:
            return None
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            self._clip_available = False
            return None

        self._clip_bundle = (
            CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
        )
        return self._clip_bundle

    def _load_cv2(self) -> Any | None:
        if self._cv2_checked:
            return self._cv2_module
        self._cv2_checked = True
        try:
            import cv2
        except ImportError:
            self._cv2_module = None
            return None
        self._cv2_module = cv2
        return self._cv2_module
