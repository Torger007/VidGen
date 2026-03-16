from PIL import Image

from app.models.schemas import GenerationParameters, PromptBundle
from app.services.scoring import CandidateScorer


def test_candidate_scorer_returns_metrics() -> None:
    scorer = CandidateScorer()
    bundle = PromptBundle(
        subject="a robot",
        scene="rainy neon city",
        style="cinematic realistic",
        action="walking forward",
        camera="slow dolly in",
        negative_prompt="blurry",
    )
    frames = [
        Image.new("RGB", (64, 64), color=(10, 10, 10)),
        Image.new("RGB", (64, 64), color=(20, 20, 20)),
    ]
    result = scorer.evaluate(
        prompt_bundle=bundle,
        parameters=GenerationParameters(),
        used_reference_image=False,
        candidate_index=0,
        initial_frame=frames[0],
        frames=frames,
    )

    assert "total_score" in result
    assert "metrics" in result
    assert "text_alignment" in result["metrics"]
    assert result["method"]["text_alignment"] in {"clip-avg", "heuristic"}
    assert result["method"]["temporal_stability"] in {"optical-flow", "frame-diff"}
    assert "diagnostics" in result
    assert "sampled_frame_indices" in result["diagnostics"]
    assert "passed_thresholds" in result
    assert "rejection_reasons" in result
