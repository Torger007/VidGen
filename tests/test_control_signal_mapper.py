from app.services.control_plan import ControlPlanBuilder
from app.services.control_signal_mapper import ControlSignalMapper
from app.services.prompting import PromptOrchestrator


def test_control_signal_mapper_builds_adapter_steps() -> None:
    orchestrator = PromptOrchestrator()
    bundle = orchestrator.build_bundle(
        "A robot walking through a city street at night",
        generation_profile="strict",
    )
    control_plan = ControlPlanBuilder().build(bundle, "strict", fps=8)
    adapter_plan = ControlSignalMapper().build(control_plan, "mock-svd", use_mock=True)

    assert adapter_plan.execution_mode == "mock"
    assert len(adapter_plan.steps) == len(control_plan.steps)
    first_step = adapter_plan.steps[0]
    signal_types = {signal.adapter_type for signal in first_step.signals}
    assert {"camera", "pose", "depth", "transition"} <= signal_types
    assert all(signal.provider is not None for signal in first_step.signals)
    assert all(signal.provider_payload for signal in first_step.signals)
