import json
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import JobRecord


class JobStore:
    def __init__(self) -> None:
        self._root = get_settings().storage_root / "jobs"

    def save(self, job: JobRecord) -> None:
        path = self._job_path(job.job_id)
        path.write_text(job.model_dump_json(indent=2), encoding="utf-8")

    def load(self, job_id: str) -> JobRecord | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def list(self, limit: int = 20) -> list[JobRecord]:
        files = sorted(self._root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        items: list[JobRecord] = []
        for path in files[:limit]:
            items.append(JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return items

    def _job_path(self, job_id: str) -> Path:
        return self._root / f"{job_id}.json"
