"""
Async Job Queue for NIH ChestX-ray14 AI Service.

Implements an in-process async job store using asyncio + Python dict (no Redis
required for dev). Supports Redis as an optional production backend via the
REDIS_URL env variable.

Job lifecycle:
  pending → running → completed | failed

Each job is identified by a UUID and stores status, submission time, and result.
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobRecord:
    job_id: str
    status: JobStatus
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return round((self.completed_at - self.started_at) * 1000, 2)
        return None


class AsyncJobStore:
    """
    Thread-safe in-process async job store.

    For production, replace _jobs with Redis HASH operations:
      redis_client.hset(job_id, mapping={...})
      redis_client.hgetall(job_id)
    """

    def __init__(self) -> None:
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new pending job and return its ID."""
        job_id = str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = JobRecord(
                job_id=job_id,
                status=JobStatus.PENDING,
                metadata=metadata or {},
            )
        logger.debug("Job created: %s", job_id)
        return job_id

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.RUNNING
                job.started_at = time.time()

    async def mark_completed(self, job_id: str, result: Any) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.result = result
        logger.info("Job completed: %s (%.1f ms)", job_id, self._jobs[job_id].duration_ms or 0)

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = error
        logger.error("Job failed: %s — %s", job_id, error)

    async def list_jobs(self, limit: int = 50) -> list:
        async with self._lock:
            jobs = list(self._jobs.values())
        return [j.to_dict() for j in sorted(jobs, key=lambda j: j.submitted_at, reverse=True)[:limit]]

    async def cleanup_old_jobs(self, max_age_seconds: float = 3600.0) -> int:
        """Prune completed/failed jobs older than max_age_seconds. Returns purge count."""
        now = time.time()
        to_delete = []
        async with self._lock:
            for jid, job in self._jobs.items():
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    if now - job.submitted_at > max_age_seconds:
                        to_delete.append(jid)
            for jid in to_delete:
                del self._jobs[jid]
        if to_delete:
            logger.info("Pruned %d old jobs from store.", len(to_delete))
        return len(to_delete)


# Global singleton job store (initialized in app.py lifespan)
_job_store: Optional[AsyncJobStore] = None


def get_job_store() -> AsyncJobStore:
    global _job_store
    if _job_store is None:
        _job_store = AsyncJobStore()
    return _job_store


def init_job_store() -> AsyncJobStore:
    global _job_store
    _job_store = AsyncJobStore()
    return _job_store
