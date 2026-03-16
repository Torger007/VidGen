from app.services.control_plan import ControlPlanBuilder
from app.services.prompting import PromptOrchestrator


def test_control_plan_builder_derives_frames_from_shot_plan() -> None:
    orchestrator = PromptOrchestrator()
    bundle = orchestrator.build_bundle(
        "A robot walking through a city street at night",
        generation_profile="balanced",
    )
    builder = ControlPlanBuilder()
    plan = builder.build(bundle, "balanced", fps=8)

    assert plan.profile == "balanced"
    assert plan.total_frames > 0
    assert len(plan.steps) == len(bundle.shot_plan)
    assert plan.steps[0].start_frame == 0
    assert plan.steps[-1].end_frame >= plan.steps[-1].start_frame
    assert plan.steps[0].camera_path
    assert plan.steps[0].pose_hint_type
    assert plan.steps[0].depth_hint_type
    assert plan.steps[0].transition_type
    assert plan.dominant_camera_path is not None
