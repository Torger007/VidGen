from typing import Any


GENERATION_PROFILES: dict[str, dict[str, Any]] = {
    "balanced": {
        "num_candidates": 2,
        "retry_attempts": 2,
        "guidance_scale": 7.5,
        "prompt_strength": 0.85,
        "reference_strength": 0.75,
        "min_total_score": 0.45,
        "min_text_alignment": 0.35,
        "min_temporal_stability": 0.30,
    },
    "strict": {
        "num_candidates": 3,
        "retry_attempts": 2,
        "guidance_scale": 8.0,
        "prompt_strength": 0.9,
        "reference_strength": 0.8,
        "min_total_score": 0.60,
        "min_text_alignment": 0.50,
        "min_temporal_stability": 0.45,
    },
    "creative": {
        "num_candidates": 4,
        "retry_attempts": 1,
        "guidance_scale": 6.5,
        "prompt_strength": 0.7,
        "reference_strength": 0.55,
        "min_total_score": 0.30,
        "min_text_alignment": 0.25,
        "min_temporal_stability": 0.20,
    },
}


def get_generation_profile(name: str | None) -> tuple[str, dict[str, Any]]:
    profile_name = name or "balanced"
    if profile_name not in GENERATION_PROFILES:
        supported = ", ".join(sorted(GENERATION_PROFILES))
        raise ValueError(f"Unsupported generation profile '{profile_name}'. Supported profiles: {supported}")
    return profile_name, GENERATION_PROFILES[profile_name].copy()
