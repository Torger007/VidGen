import json
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import CandidateSummary, utc_now
from app.services.job_store import JobStore
from app.services.video_pipeline import VideoPipeline
from app.workers.celery_app import celery_app


def enqueue_generate_video(job_id: str) -> None:
    settings = get_settings()
    if settings.task_mode == "eager":
        run_generate_video(job_id)
        return
    celery_app.send_task("app.tasks.generate_video.run_generate_video", args=[job_id])


@celery_app.task(name="app.tasks.generate_video.run_generate_video")
def run_generate_video(job_id: str) -> None:
    store = JobStore()
    pipeline = VideoPipeline()
    job = store.load(job_id)
    if job is None:
        return

    job.status = "running"
    job.updated_at = utc_now()
    store.save(job)

    try:
        output_path = pipeline.render(job)
        job.status = "succeeded"
        job.output_path = output_path
        if output_path.endswith(".json"):
            job.output_preview_path = output_path.replace(".json", ".png")
            metadata = _load_metadata(output_path)
            job.candidates = [CandidateSummary.model_validate(item) for item in metadata.get("candidates", [])]
            job.selected_candidate = (
                CandidateSummary.model_validate(metadata["selected_candidate"])
                if metadata.get("selected_candidate")
                else None
            )
            job.selection_mode = metadata.get("selection_mode")
            job.candidate_paths = [item.path for item in job.candidates if item.path]
            if not job.candidate_paths:
                job.candidate_paths = [item.video_path for item in job.candidates if item.video_path]
            job.rejection_reason_counts = _collect_rejection_reason_counts(metadata.get("candidates", []))
            if job.selected_candidate and job.selected_candidate.preview_path:
                job.output_preview_path = job.selected_candidate.preview_path
        job.error_message = None
    except Exception as exc:  # pragma: no cover
        job.status = "failed"
        job.error_message = str(exc)
    finally:
        job.updated_at = utc_now()
        store.save(job)


def _load_metadata(output_path: str) -> dict:
    path = Path(output_path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_rejection_reason_counts(candidates: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        evaluation = candidate.get("evaluation") or {}
        for reason in evaluation.get("rejection_reasons", []):
            counts[reason] = counts.get(reason, 0) + 1
    return counts
