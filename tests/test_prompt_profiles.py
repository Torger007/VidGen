from app.services.prompting import PromptOrchestrator


def test_strict_profile_changes_prompt_bundle() -> None:
    orchestrator = PromptOrchestrator()
    bundle = orchestrator.build_bundle(
        "A robot walking through a city street at night",
        generation_profile="strict",
    )

    assert "stable" in bundle.style.lower()
    assert "controlled" in bundle.action.lower() or "restrained" in bundle.action.lower()
    assert "stable" in bundle.camera.lower() or "locked" in bundle.camera.lower()
    assert bundle.scene_template == "city"
    assert len(bundle.shot_plan) == 3


def test_creative_profile_prefers_dynamic_language() -> None:
    orchestrator = PromptOrchestrator()
    bundle = orchestrator.build_bundle(
        "A runner moving through a forest trail",
        generation_profile="creative",
    )

    assert "dynamic" in bundle.style.lower() or "stylized" in bundle.style.lower()
    assert "energetic" in bundle.action.lower() or "expressive" in bundle.action.lower()
    assert any("motion" in step.emphasis.lower() or "action focus" in step.emphasis.lower() for step in bundle.shot_plan)
