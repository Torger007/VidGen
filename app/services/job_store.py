import json
import time
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.models.schemas import JobRecord


class JobStore:
    _read_retry_attempts = 3
    _read_retry_delay_sec = 0.05

    def __init__(self) -> None:
        self._root = get_settings().storage_root / "jobs"

    def save(self, job: JobRecord) -> None:
        path = self._job_path(job.job_id)
        tmp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(job.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)

    def load(self, job_id: str) -> JobRecord | None:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return self._read_job_file(path)

    def list(self, limit: int = 20) -> list[JobRecord]:
        files = sorted(self._root.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        items: list[JobRecord] = []
        for path in files[:limit]:
            job = self._read_job_file(path)
            if job is not None:
                items.append(job)
        return items

    def _job_path(self, job_id: str) -> Path:
        return self._root / f"{job_id}.json"

    def _read_job_file(self, path: Path) -> JobRecord | None:
        last_error: json.JSONDecodeError | None = None
        for attempt in range(self._read_retry_attempts):
            try:
                return JobRecord.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError as exc:
                last_error = exc
                if attempt == self._read_retry_attempts - 1:
                    raise
                time.sleep(self._read_retry_delay_sec)
        if last_error is not None:
            raise last_error
        return None
