# NemoReconstruct

NemoReconstruct is a minimal end-to-end 3D reconstruction MVP for hackathon use.
It accepts a `.MOV` upload, runs a simplified video-to-fVDB workflow, exposes a clean FastAPI backend with OpenAPI docs at `/docs`, ships a lightweight Next.js frontend, and includes Python and TypeScript SDKs for agents.

## What It Does

- Upload a `.MOV` file through the backend or frontend
- Extract frames with `ffmpeg`
- Recover camera poses with `COLMAP`
- Run `frgs reconstruct` from NVIDIA fVDB Reality Capture
- Produce a demo-ready PLY splat output quickly (splat-only mode by default)
- Expose a reproducible API surface with OpenAPI and SDKs

## Architecture

- `backend/`: FastAPI API, SQLite job store, single-worker background runner
- `frontend/`: Next.js upload dashboard for uploading and monitoring reconstructions
- `sdk/python/`: Python client for agents and automation
- `sdk/typescript/`: TypeScript client matching the backend endpoints
- `docs/openapi.json`: exported OpenAPI schema

## Backend Endpoints

- `GET /health`
- `GET /api/v1/pipelines`
- `POST /api/v1/reconstructions/upload`
- `GET /api/v1/reconstructions`
- `GET /api/v1/reconstructions/{id}`
- `GET /api/v1/reconstructions/{id}/status`
- `GET /api/v1/reconstructions/{id}/artifacts`
- `GET /api/v1/reconstructions/{id}/download/{artifact}`
- `POST /api/v1/reconstructions/{id}/retry`
- `DELETE /api/v1/reconstructions/{id}`
- `GET /docs`
- `GET /openapi.json`

## Demo URLs (Current Setup)

- Frontend dashboard: `http://127.0.0.1:3000`
- Backend API: `http://127.0.0.1:8010`
- Backend docs: `http://127.0.0.1:8010/docs`
- OpenAPI JSON: `http://127.0.0.1:8010/openapi.json`

## How To Download Results

After a run completes, get artifact URLs:

```bash
curl -s http://127.0.0.1:8010/api/v1/reconstructions/<id>/artifacts
```

Download the splat PLY:

```bash
curl -L -o output.ply \
	http://127.0.0.1:8010/api/v1/reconstructions/<id>/download/splat_ply
```

Download run log and metadata:

```bash
curl -L -o run.log \
	http://127.0.0.1:8010/api/v1/reconstructions/<id>/download/run_log

curl -L -o metadata.json \
	http://127.0.0.1:8010/api/v1/reconstructions/<id>/download/metadata
```

The latest demo run in this session was:

- Reconstruction ID: `2c04ca28-3d20-4222-b91a-f75e0a8f3519`
- PLY URL: `http://127.0.0.1:8010/api/v1/reconstructions/2c04ca28-3d20-4222-b91a-f75e0a8f3519/download/splat_ply`

## System Requirements

These tools are expected to be installed on the machine running the backend:

- `ffmpeg`
- `colmap`
- `frgs` from NVIDIA fVDB Reality Capture

If any dependency is missing, the backend will fail the job with a clear error message in the reconstruction status.

## Quick Start

See [Quickstart](#quickstart-1) at the bottom for the exact commands.

## Agent-Tunable Reconstruction Params

Your agent can tune each run by passing these optional fields to `POST /api/v1/reconstructions/upload`:

- `frame_rate` (float, 0.25-12.0)
- `sequential_matcher_overlap` (int, 2-50)
- `fvdb_max_epochs` (int, 5-500)
- `fvdb_sh_degree` (int, 0-4)
- `fvdb_image_downsample_factor` (int, 1-12)
- `splat_only_mode` (bool)

These params are persisted per reconstruction and returned in API responses as `processing_params`.

### Upload With Params (curl)

```bash
curl -s -X POST http://127.0.0.1:8010/api/v1/reconstructions/upload \
	-F "file=@/path/to/video.MOV" \
	-F "name=iter-01" \
	-F "description=agent tuning run" \
	-F "frame_rate=2.0" \
	-F "sequential_matcher_overlap=12" \
	-F "fvdb_max_epochs=40" \
	-F "fvdb_sh_degree=3" \
	-F "fvdb_image_downsample_factor=6" \
	-F "splat_only_mode=true"
```

### Retry Existing Run With New Params

```bash
curl -s -X POST http://127.0.0.1:8010/api/v1/reconstructions/<id>/retry \
	-H "Content-Type: application/json" \
	-d '{
		"params": {
			"fvdb_max_epochs": 60,
			"fvdb_image_downsample_factor": 4,
			"frame_rate": 3.0
		}
	}'
```

### 3. Export OpenAPI

```bash
cd backend
python export_openapi.py
```

This writes `../docs/openapi.json`.

## Agent Workflow Example

```python
from nemo_reconstruct_client import NemoReconstructClient

client = NemoReconstructClient("http://127.0.0.1:8010")
job = client.upload_video(
	"/tmp/scene.mov",
	name="warehouse-walkthrough",
	params={
		"fvdb_max_epochs": 40,
		"fvdb_image_downsample_factor": 6,
		"splat_only_mode": True,
	},
)
result = client.wait_for_completion(job.id)
print(result.status)
print(client.get_artifacts(job.id))
```

## Notes

- This is intentionally an MVP: one hardcoded pipeline, one local runner, local disk storage, and SQLite.
- The pipeline shape mirrors Magic Mirror, but strips out Celery, Redis, MinIO, and Postgres to keep setup small.
- The workflow is suitable for hackathon iteration and agent orchestration, not high-availability production use.

## Quickstart

Run these commands in order. Open two terminals — one for the backend, one for the frontend.

### Terminal 1 — Backend

```bash
cd /home/clayton_littlejohn/devl/github/NemoReconstruct/backend
source /home/clayton_littlejohn/devl/github/magic-mirror/.venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload
```

Backend API: `http://127.0.0.1:8010`  
Interactive docs: `http://127.0.0.1:8010/docs`

### Terminal 2 — Frontend

```bash
cd /home/clayton_littlejohn/devl/github/NemoReconstruct/frontend
cp .env.example .env.local
npm install
npm run dev
```

Frontend dashboard: `http://127.0.0.1:3000`

### Upload a video

```bash
curl -s -X POST http://127.0.0.1:8010/api/v1/reconstructions/upload \
  -F "file=@/path/to/video.MOV" \
  -F "name=my-first-run"
# returns {"id": "<job-id>", ...}
```

Poll until `status` is `completed`:

```bash
curl -s http://127.0.0.1:8010/api/v1/reconstructions/<id>/status
```

### Download the PLY splat

```bash
curl -L -o output.ply \
  http://127.0.0.1:8010/api/v1/reconstructions/<id>/download/splat_ply
```

Open `output.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor) or [antimatter15/splat](https://antimatter15.com/splat/) by dragging the file onto the page.

