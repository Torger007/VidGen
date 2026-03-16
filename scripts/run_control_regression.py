from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.models.schemas import GenerationParameters, JobRecord, utc_now
from app.services.control_plan import ControlPlanBuilder
from app.services.prompting import PromptOrchestrator
from app.services.video_pipeline import VideoPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal control regression cases.")
    parser.add_argument(
        "--cases",
        default="tests/fixtures/regression_cases.json",
        help="Path to regression case definitions.",
    )
    parser.add_argument(
        "--output",
        default="storage/regression/control-regression-summary.json",
        help="Where to write the aggregated regression summary.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    case_path = Path(args.cases)
    output_path = Path(args.output)
    report_path = output_path.with_suffix(".md")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cases = json.loads(case_path.read_text(encoding="utf-8"))
    pipeline = VideoPipeline()
    prompting = PromptOrchestrator()
    control_builder = ControlPlanBuilder()

    summary: dict[str, Any] = {
        "status": "running",
        "case_count": len(cases),
        "cases": [],
    }
    _write_summary(output_path, report_path, summary)
    for case in cases:
        case_summary: dict[str, Any] = {
            "case_id": case["case_id"],
            "status": "running",
        }
        summary["cases"].append(case_summary)
        _write_summary(output_path, report_path, summary)

        baseline = run_case(
            case=case,
            mode="baseline",
            pipeline=pipeline,
            prompting=prompting,
            control_builder=control_builder,
            include_control=False,
        )
        case_summary["baseline"] = baseline
        _write_summary(output_path, report_path, summary)

        controlled = run_case(
            case=case,
            mode="pose_depth",
            pipeline=pipeline,
            prompting=prompting,
            control_builder=control_builder,
            include_control=True,
        )
        case_summary["pose_depth"] = controlled
        case_summary["assertions"] = compare_runs(baseline, controlled)
        case_summary["status"] = "completed"
        _write_summary(output_path, report_path, summary)

    summary["status"] = "completed"
    _write_summary(output_path, report_path, summary)
    print(output_path)
    print(report_path)
    return 0


def run_case(
    *,
    case: dict[str, Any],
    mode: str,
    pipeline: VideoPipeline,
    prompting: PromptOrchestrator,
    control_builder: ControlPlanBuilder,
    include_control: bool,
) -> dict[str, Any]:
    job_id = f"{case['case_id']}-{mode}"
    try:
        parameters = GenerationParameters.model_validate(case["parameters"])
        prompt_bundle = prompting.build_bundle(
            case["prompt"],
            generation_profile=case["generation_profile"],
        )
        control_plan = (
            control_builder.build(prompt_bundle, case["generation_profile"], parameters.fps) if include_control else None
        )
        if control_plan is not None:
            parameters.duration_sec = max(1, round(control_plan.total_duration_sec))
            parameters.num_frames = control_plan.total_frames
        job = JobRecord(
            job_id=job_id,
            prompt=case["prompt"],
            status="running",
            reference_image_path=case["reference_image_path"],
            generation_profile=case["generation_profile"],
            prompt_bundle=prompt_bundle,
            control_plan=control_plan,
            parameters=parameters,
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        output_path = pipeline.render(job)
        metadata = json.loads(Path(output_path).read_text(encoding="utf-8"))
        selected = metadata.get("selected_candidate") or {}
        routing_summary = selected.get("routing_summary") or {}
        return {
            "mode": mode,
            "job_id": job.job_id,
            "status": "succeeded",
            "error_message": None,
            "output_path": output_path,
            "preview_path": metadata.get("preview_path"),
            "video_path": metadata.get("video_path"),
            "provider_artifacts": metadata.get("provider_execution", {}).get("artifact_index", {}),
            "generation_context": metadata.get("generation_context", {}),
            "routing_summary": routing_summary,
            "score_total": selected.get("score"),
            "selected_candidate": selected,
        }
    except Exception as exc:
        return {
            "mode": mode,
            "job_id": job_id,
            "status": "failed",
            "error_message": str(exc),
            "output_path": None,
            "preview_path": None,
            "video_path": None,
            "provider_artifacts": {},
            "generation_context": {},
            "routing_summary": {},
            "score_total": None,
            "selected_candidate": {},
        }


def compare_runs(baseline: dict[str, Any], controlled: dict[str, Any]) -> dict[str, Any]:
    baseline_routing = baseline.get("routing_summary") or {}
    controlled_routing = controlled.get("routing_summary") or {}
    baseline_score = baseline.get("score_total")
    controlled_score = controlled.get("score_total")
    score_delta = None
    if isinstance(baseline_score, (int, float)) and isinstance(controlled_score, (int, float)):
        score_delta = round(controlled_score - baseline_score, 6)
    return {
        "baseline_succeeded": baseline.get("status") == "succeeded",
        "controlled_succeeded": controlled.get("status") == "succeeded",
        "controlled_has_pose": bool(controlled_routing.get("pose_control")),
        "controlled_has_depth": bool(controlled_routing.get("depth_control")),
        "controlled_video_conditioning_used": bool(controlled_routing.get("used_video_conditioning")),
        "routing_differs_from_baseline": baseline_routing != controlled_routing,
        "score_delta": score_delta,
    }


def _write_summary(json_path: Path, report_path: Path, payload: dict[str, Any]) -> None:
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(_render_markdown_report(payload), encoding="utf-8")


def _render_markdown_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Control Regression Report")
    lines.append("")
    lines.append(f"- Status: `{payload.get('status', 'unknown')}`")
    lines.append(f"- Cases: `{payload.get('case_count', 0)}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Case | Status | Baseline | Pose+Depth | Routing Differs | Score Delta |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for case in payload.get("cases", []):
        assertions = case.get("assertions") or {}
        baseline = case.get("baseline") or {}
        controlled = case.get("pose_depth") or {}
        lines.append(
            "| {case_id} | {status} | {baseline_status} | {controlled_status} | {routing_differs} | {score_delta} |".format(
                case_id=case.get("case_id", "<unknown>"),
                status=case.get("status", "running"),
                baseline_status=baseline.get("status", "-"),
                controlled_status=controlled.get("status", "-"),
                routing_differs=assertions.get("routing_differs_from_baseline", "-"),
                score_delta=assertions.get("score_delta", "-"),
            )
        )
    lines.append("")

    for case in payload.get("cases", []):
        lines.extend(_render_case_section(case))

    return "\n".join(lines) + "\n"


def _render_case_section(case: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    case_id = case.get("case_id", "<unknown>")
    baseline = case.get("baseline") or {}
    controlled = case.get("pose_depth") or {}
    assertions = case.get("assertions") or {}

    lines.append(f"## {case_id}")
    lines.append("")
    lines.append(f"- Case status: `{case.get('status', 'running')}`")
    lines.append(f"- Baseline status: `{baseline.get('status', '-')}`")
    lines.append(f"- Pose+Depth status: `{controlled.get('status', '-')}`")
    lines.append(f"- Routing differs: `{assertions.get('routing_differs_from_baseline', '-')}`")
    lines.append(f"- Score delta: `{assertions.get('score_delta', '-')}`")
    lines.append("")
    lines.append("### Baseline")
    lines.append("")
    lines.extend(_render_run_block(baseline))
    lines.append("")
    lines.append("### Pose+Depth")
    lines.append("")
    lines.extend(_render_run_block(controlled))
    lines.append("")
    lines.append("### Assertions")
    lines.append("")
    for key in (
        "baseline_succeeded",
        "controlled_succeeded",
        "controlled_has_pose",
        "controlled_has_depth",
        "controlled_video_conditioning_used",
        "routing_differs_from_baseline",
        "score_delta",
    ):
        lines.append(f"- {key}: `{assertions.get(key, '-')}`")
    lines.append("")
    return lines


def _render_run_block(run: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    routing = run.get("routing_summary") or {}
    provider_artifacts = run.get("provider_artifacts") or {}
    lines.append(f"- Job id: `{run.get('job_id', '-')}`")
    lines.append(f"- Status: `{run.get('status', '-')}`")
    lines.append(f"- Error: `{run.get('error_message', '-')}`")
    lines.append(f"- Output: `{run.get('output_path', '-')}`")
    lines.append(f"- Preview: `{run.get('preview_path', '-')}`")
    lines.append(f"- Video: `{run.get('video_path', '-')}`")
    lines.append(f"- Score: `{run.get('score_total', '-')}`")
    lines.append(f"- Provider artifacts: `{', '.join(sorted(provider_artifacts)) if provider_artifacts else '-'}`")
    if routing:
        lines.append(f"- Pose control: `{routing.get('pose_control', False)}`")
        lines.append(f"- Depth control: `{routing.get('depth_control', False)}`")
        lines.append(f"- Used image conditioning: `{routing.get('used_image_conditioning', False)}`")
        lines.append(f"- Used video conditioning: `{routing.get('used_video_conditioning', False)}`")
        lines.append(
            f"- Video call summary: `{json.dumps(routing.get('video_call_summary', {}), ensure_ascii=True)}`"
        )
    return lines


if __name__ == "__main__":
    raise SystemExit(main())
