from __future__ import annotations

import json
import shutil
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models import Reconstruction, ReconstructionStatus
from app.schemas import DeleteResponse, PipelineInfo, ReconstructionArtifacts, ReconstructionDetail, ReconstructionParams, ReconstructionStatusResponse, RetryRequest, RetryResponse, UploadResponse
from app.services.pipeline import PIPELINE_INFO
from app.services.storage import ensure_parent, remove_workspace
from app.workers.runner import runner


router = APIRouter(prefix=settings.api_prefix, tags=["reconstructions"])

ALLOWED_EXTENSIONS = {".mov", ".mp4", ".m4v"}
ARTIFACT_FIELDS = {
    "source_video": "source_video_path",
    "splat_ply": "artifact_ply_path",
    "scene_usdz": "artifact_usdz_path",
    "sim_bundle": "artifact_bundle_path",
    "run_log": "artifact_log_path",
    "metadata": "artifact_metadata_path",
}


def build_download_url(reconstruction_id: str, artifact: str) -> str:
    return f"{settings.api_prefix}/reconstructions/{reconstruction_id}/download/{artifact}"


def serialize_processing_params(reconstruction: Reconstruction) -> ReconstructionParams:
    if not reconstruction.processing_params_json:
        return ReconstructionParams()
    try:
        payload = json.loads(reconstruction.processing_params_json)
    except json.JSONDecodeError:
        return ReconstructionParams()
    if not isinstance(payload, dict):
        return ReconstructionParams()
    try:
        return ReconstructionParams(**payload)
    except Exception:
        return ReconstructionParams()


def build_processing_params(
    frame_rate: float | None,
    sequential_matcher_overlap: int | None,
    fvdb_max_epochs: int | None,
    fvdb_sh_degree: int | None,
    fvdb_image_downsample_factor: int | None,
    splat_only_mode: bool | None,
) -> ReconstructionParams:
    return ReconstructionParams(
        frame_rate=frame_rate,
        sequential_matcher_overlap=sequential_matcher_overlap,
        fvdb_max_epochs=fvdb_max_epochs,
        fvdb_sh_degree=fvdb_sh_degree,
        fvdb_image_downsample_factor=fvdb_image_downsample_factor,
        splat_only_mode=splat_only_mode,
    )


def serialize_reconstruction(reconstruction: Reconstruction) -> ReconstructionDetail:
    return ReconstructionDetail(
        id=reconstruction.id,
        name=reconstruction.name,
        description=reconstruction.description,
        status=reconstruction.status,
        pipeline_slug=reconstruction.pipeline_slug,
        processing_step=reconstruction.processing_step,
        processing_pct=reconstruction.processing_pct,
        error_message=reconstruction.error_message,
        source_video_filename=reconstruction.source_video_filename,
        frame_count=reconstruction.frame_count,
        created_at=reconstruction.created_at,
        updated_at=reconstruction.updated_at,
        started_at=reconstruction.started_at,
        completed_at=reconstruction.completed_at,
        processing_params=serialize_processing_params(reconstruction),
        artifact_ply_url=build_download_url(reconstruction.id, "splat_ply") if reconstruction.artifact_ply_path else None,
        artifact_usdz_url=build_download_url(reconstruction.id, "scene_usdz") if reconstruction.artifact_usdz_path else None,
        artifact_bundle_url=build_download_url(reconstruction.id, "sim_bundle") if reconstruction.artifact_bundle_path else None,
        artifact_log_url=build_download_url(reconstruction.id, "run_log") if reconstruction.artifact_log_path else None,
        artifact_metadata_url=build_download_url(reconstruction.id, "metadata") if reconstruction.artifact_metadata_path else None,
    )


def get_reconstruction_or_404(db: Session, reconstruction_id: str) -> Reconstruction:
    reconstruction = db.get(Reconstruction, reconstruction_id)
    if reconstruction is None:
        raise HTTPException(status_code=404, detail="Reconstruction not found")
    return reconstruction


@router.get("/pipelines", response_model=list[PipelineInfo], tags=["pipelines"])
def list_pipelines() -> list[PipelineInfo]:
    return [PipelineInfo(**PIPELINE_INFO)]


@router.post("/reconstructions/upload", response_model=UploadResponse, status_code=201)
def upload_reconstruction(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str | None = Form(None),
    frame_rate: float | None = Form(None),
    sequential_matcher_overlap: int | None = Form(None),
    fvdb_max_epochs: int | None = Form(None),
    fvdb_sh_degree: int | None = Form(None),
    fvdb_image_downsample_factor: int | None = Form(None),
    splat_only_mode: bool | None = Form(None),
    db: Session = Depends(get_db),
) -> UploadResponse:
    suffix = Path(file.filename or "upload.mov").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only MOV, MP4, and M4V uploads are supported")

    processing_params = build_processing_params(
        frame_rate,
        sequential_matcher_overlap,
        fvdb_max_epochs,
        fvdb_sh_degree,
        fvdb_image_downsample_factor,
        splat_only_mode,
    )

    reconstruction = Reconstruction(
        name=name,
        description=description,
        status=ReconstructionStatus.uploading.value,
        source_video_filename=file.filename or f"upload{suffix}",
        source_video_path="",
        workspace_dir="",
        processing_params_json=processing_params.model_dump_json(exclude_none=True),
    )
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    workspace_dir = settings.storage_dir / reconstruction.id
    source_video_path = workspace_dir / f"source{suffix}"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(source_video_path)

    with source_video_path.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    reconstruction.workspace_dir = str(workspace_dir)
    reconstruction.source_video_path = str(source_video_path)
    reconstruction.status = ReconstructionStatus.queued.value
    reconstruction.processing_step = "queued"
    reconstruction.processing_pct = 1
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    runner.enqueue(reconstruction.id)
    return UploadResponse(**serialize_reconstruction(reconstruction).model_dump())


@router.get("/reconstructions", response_model=list[ReconstructionDetail])
def list_reconstructions(db: Session = Depends(get_db)) -> list[ReconstructionDetail]:
    reconstructions = db.query(Reconstruction).order_by(Reconstruction.created_at.desc()).all()
    return [serialize_reconstruction(item) for item in reconstructions]


@router.get("/reconstructions/{reconstruction_id}", response_model=ReconstructionDetail)
def get_reconstruction(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionDetail:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return serialize_reconstruction(reconstruction)


@router.get("/reconstructions/{reconstruction_id}/status", response_model=ReconstructionStatusResponse)
def get_reconstruction_status(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionStatusResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return ReconstructionStatusResponse(
        id=reconstruction.id,
        status=reconstruction.status,
        processing_step=reconstruction.processing_step,
        processing_pct=reconstruction.processing_pct,
        error_message=reconstruction.error_message,
        updated_at=reconstruction.updated_at,
    )


@router.get("/reconstructions/{reconstruction_id}/artifacts", response_model=ReconstructionArtifacts)
def get_reconstruction_artifacts(reconstruction_id: str, db: Session = Depends(get_db)) -> ReconstructionArtifacts:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    return ReconstructionArtifacts(
        source_video_url=build_download_url(reconstruction.id, "source_video"),
        splat_ply_url=build_download_url(reconstruction.id, "splat_ply") if reconstruction.artifact_ply_path else None,
        scene_usdz_url=build_download_url(reconstruction.id, "scene_usdz") if reconstruction.artifact_usdz_path else None,
        sim_bundle_url=build_download_url(reconstruction.id, "sim_bundle") if reconstruction.artifact_bundle_path else None,
        run_log_url=build_download_url(reconstruction.id, "run_log") if reconstruction.artifact_log_path else None,
        metadata_url=build_download_url(reconstruction.id, "metadata") if reconstruction.artifact_metadata_path else None,
    )


@router.get("/reconstructions/{reconstruction_id}/download/{artifact}")
def download_artifact(reconstruction_id: str, artifact: str, db: Session = Depends(get_db)) -> FileResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    field_name = ARTIFACT_FIELDS.get(artifact)
    if field_name is None:
        raise HTTPException(status_code=404, detail="Unknown artifact")

    artifact_path = getattr(reconstruction, field_name)
    if not artifact_path:
        raise HTTPException(status_code=404, detail="Artifact not available")

    path = Path(artifact_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing on disk")

    media_type = None
    if artifact == "scene_usdz":
        media_type = "model/vnd.usdz+zip"
    elif artifact == "sim_bundle":
        media_type = "application/zip"
    elif artifact == "run_log":
        media_type = "text/plain"
    elif artifact == "metadata":
        media_type = "application/json"

    return FileResponse(path, media_type=media_type, filename=path.name)


@router.post("/reconstructions/{reconstruction_id}/retry", response_model=RetryResponse)
def retry_reconstruction(
    reconstruction_id: str,
    request: RetryRequest | None = Body(default=None),
    db: Session = Depends(get_db),
) -> RetryResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    if reconstruction.status not in {ReconstructionStatus.failed.value, ReconstructionStatus.completed.value}:
        raise HTTPException(status_code=409, detail="Only completed or failed reconstructions can be retried")

    reconstruction.status = ReconstructionStatus.queued.value
    reconstruction.processing_step = "queued"
    reconstruction.processing_pct = 1
    reconstruction.error_message = None
    reconstruction.started_at = None
    reconstruction.completed_at = None
    reconstruction.frame_count = None
    reconstruction.artifact_ply_path = None
    reconstruction.artifact_usdz_path = None
    reconstruction.artifact_bundle_path = None
    reconstruction.artifact_log_path = None
    reconstruction.artifact_metadata_path = None
    if request and request.params is not None:
        reconstruction.processing_params_json = request.params.model_dump_json(exclude_none=True)
    db.add(reconstruction)
    db.commit()
    db.refresh(reconstruction)

    runner.enqueue(reconstruction.id)
    return RetryResponse(**serialize_reconstruction(reconstruction).model_dump())


@router.delete("/reconstructions/{reconstruction_id}", response_model=DeleteResponse)
def delete_reconstruction(reconstruction_id: str, db: Session = Depends(get_db)) -> DeleteResponse:
    reconstruction = get_reconstruction_or_404(db, reconstruction_id)
    workspace = Path(reconstruction.workspace_dir)
    db.delete(reconstruction)
    db.commit()
    remove_workspace(workspace)
    return DeleteResponse(id=reconstruction_id)
