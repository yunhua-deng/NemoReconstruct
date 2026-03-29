from __future__ import annotations

import queue
import threading

from app.core.database import SessionLocal
from app.models import Reconstruction, ReconstructionStatus
from app.services.pipeline import process_reconstruction_job


class ReconstructionRunner:
    def __init__(self) -> None:
        self._queue: queue.Queue[str] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="nemo-reconstruct-runner", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def enqueue(self, reconstruction_id: str) -> None:
        self._queue.put(reconstruction_id)

    def mark_inflight_jobs_failed(self) -> None:
        with SessionLocal() as session:
            jobs = session.query(Reconstruction).filter(
                Reconstruction.status.in_(
                    [
                        ReconstructionStatus.uploading.value,
                        ReconstructionStatus.queued.value,
                        ReconstructionStatus.extracting_frames.value,
                        ReconstructionStatus.feature_extraction.value,
                        ReconstructionStatus.feature_matching.value,
                        ReconstructionStatus.sparse_reconstruction.value,
                        ReconstructionStatus.fvdb_reconstruction.value,
                        ReconstructionStatus.grut_reconstruction.value,
                        ReconstructionStatus.exporting.value,
                        ReconstructionStatus.generating_collision_mesh.value,
                    ]
                )
            )
            for job in jobs:
                job.status = ReconstructionStatus.failed.value
                job.processing_step = "startup_recovery"
                job.error_message = "Server restarted before the reconstruction finished. Re-run the job."
                session.add(job)
            session.commit()

    def _run(self) -> None:
        while True:
            reconstruction_id = self._queue.get()
            try:
                with SessionLocal() as session:
                    process_reconstruction_job(session, reconstruction_id)
            finally:
                self._queue.task_done()


runner = ReconstructionRunner()
