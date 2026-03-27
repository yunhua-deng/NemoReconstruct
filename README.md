# NemoReconstruct

A minimal 3D reconstruction pipeline designed as a **NemoClaw + OpenShell demo** — upload a video, get a Gaussian Splat PLY, all controlled by an AI agent running inside an isolated sandbox with local inference.

> **This repo is a template.** The [setup guide](docs/NEMOCLAW_SETUP.md) shows how to pair NemoClaw + OpenShell with **any** repo or API — NemoReconstruct is just the example project.

```
Video → ffmpeg → COLMAP → fVDB/frgs → PLY
                    ↑ controlled by ↑
              NemoClaw Agent (OpenClaw)
              inside OpenShell Sandbox
              powered by Ollama (local LLM)
```

---

## Quick Start

> **Prerequisites:** Linux + NVIDIA GPU, Docker, Python 3.10+, Node.js 18+.
> See [the full tutorial](docs/NEMOCLAW_SETUP.md) for step-by-step install of every prerequisite.

```bash
# 1. Start the backend (leave running)
cd ~/NemoReconstruct && make backend-dev

# 2. One-time OpenShell + Ollama setup (see full guide for details)
openshell gateway start --gpu
openshell provider create --name ollama --type openai \
  --credential OPENAI_API_KEY=empty \
  --config OPENAI_BASE_URL=http://host.openshell.internal:11434/v1
openshell inference set --provider ollama --model glm-4.7-flash

# 3. Run the agent in a sandbox
openshell sandbox create \
  --from openclaw \
  --policy nemoclaw/sandbox-policy.yaml \
  --upload "$PWD:/sandbox/NemoReconstruct" \
  -- bash -c '
mkdir -p /sandbox/.openclaw
cp /sandbox/NemoReconstruct/nemoclaw/sandbox-openclaw.json /sandbox/.openclaw/openclaw.json
export OPENAI_API_KEY=unused
cd /sandbox/NemoReconstruct
openclaw agent --local --session-id demo \
  --message "List all reconstruction jobs. Use curl to call the API at http://172.20.0.1:8010" \
  --json --timeout 120
'
```

**Full tutorial:** [docs/NEMOCLAW_SETUP.md](docs/NEMOCLAW_SETUP.md) — covers every step from a fresh machine, and shows how to adapt this for your own project.

---

## What the Agent Can Do

- Upload videos and start reconstruction jobs
- Tune parameters (frame rate, epochs, downsample factor, quality presets)
- Monitor progress in real time
- Download PLY splat outputs
- Retry failed jobs with adjusted settings
- Inspect logs and system state via shell

---

## Project Structure

```
backend/          FastAPI API, SQLite job store, background runner
frontend/         Next.js upload dashboard
sdk/python/       Python client for agents and automation
sdk/typescript/   TypeScript client
nemoclaw/         Agent config, system prompt, sandbox policy, example scripts
docs/             OpenAPI schema, setup guide
```

### NemoClaw Config Files

| File | Purpose |
|------|---------|
| `nemoclaw/sandbox-policy.yaml` | OpenShell sandbox network policy (NemoReconstruct) |
| `nemoclaw/sandbox-policy-template.yaml` | Generic sandbox policy — copy and customize for your project |
| `nemoclaw/sandbox-openclaw.json` | OpenClaw config (NemoReconstruct → `inference.local`) |
| `nemoclaw/sandbox-openclaw-template.json` | Generic OpenClaw config — copy and customize for your project |
| `nemoclaw/nemoclaw_config.yaml` | Agent tools, model, guardrails |
| `nemoclaw/system_prompt.md` | Agent persona and workflow rules |
| `nemoclaw/example_session.py` | SDK script to test the pipeline without an agent |

---

## Backend API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/pipelines` | List available pipelines |
| POST | `/api/v1/reconstructions/upload` | Upload video + start reconstruction |
| GET | `/api/v1/reconstructions` | List all jobs |
| GET | `/api/v1/reconstructions/{id}` | Job details |
| GET | `/api/v1/reconstructions/{id}/status` | Poll status + progress |
| GET | `/api/v1/reconstructions/{id}/artifacts` | List downloadable artifacts |
| GET | `/api/v1/reconstructions/{id}/download/{artifact}` | Download artifact |
| POST | `/api/v1/reconstructions/{id}/retry` | Retry with new params |
| DELETE | `/api/v1/reconstructions/{id}` | Delete a job |
| GET | `/docs` | Interactive API docs |
| GET | `/openapi.json` | OpenAPI schema |

### Tunable Parameters

Pass these to `POST /api/v1/reconstructions/upload`:

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `frame_rate` | 0.25 – 12.0 | 2.0 | Frames/sec extracted by ffmpeg |
| `sequential_matcher_overlap` | 2 – 50 | 12 | COLMAP matcher overlap |
| `fvdb_max_epochs` | 5 – 500 | 40 | fVDB training epochs |
| `fvdb_sh_degree` | 0 – 4 | 3 | Spherical harmonics degree |
| `fvdb_image_downsample_factor` | 1 – 12 | 6 | Input image downsampling |
| `splat_only_mode` | true/false | true | Skip USDZ, produce PLY only |

---

## Manual Usage (Without Agent)

### Upload a video
```bash
curl -s -X POST http://localhost:8010/api/v1/reconstructions/upload \
  -F "file=@/path/to/video.MOV" \
  -F "name=my-scene" \
  -F "frame_rate=2.0" \
  -F "fvdb_max_epochs=40"
```

### Poll status
```bash
curl -s http://localhost:8010/api/v1/reconstructions/<id>/status
```

### Download the PLY
```bash
curl -L -o output.ply \
  http://localhost:8010/api/v1/reconstructions/<id>/download/splat_ply
```

### Python SDK
```python
from nemo_reconstruct_client import NemoReconstructClient

client = NemoReconstructClient("http://localhost:8010")
job = client.upload_video("/tmp/scene.mov", "my-scene",
    params={"fvdb_max_epochs": 40, "splat_only_mode": True})
result = client.wait_for_completion(job.id)
print(client.get_artifacts(job.id))
```

---

## Development

```bash
# Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
make backend-dev                # Starts on 0.0.0.0:8010

# Frontend (optional)
cd frontend && npm install
make frontend-dev               # Starts on localhost:3000

# Export OpenAPI schema
make openapi

# Install Python SDK
pip install -e sdk/python
```

### System Requirements

Pipeline binaries (must be on PATH or configured):
- `ffmpeg`
- `colmap`
- `frgs` from NVIDIA fVDB Reality Capture (typically at `~/miniconda3/envs/fvdb/bin/frgs`)

---

## How It All Fits Together

```
┌─────────────────────────────────────────────────────────────────┐
│  DGX Spark                                                      │
│                                                                  │
│  ┌──────────────────────── OpenShell Sandbox ──────────────────┐ │
│  │                                                              │ │
│  │  OpenClaw Agent ──► inference.local ──► Gateway ──► Ollama  │ │
│  │       │                                           :11434    │ │
│  │       │ curl / OpenAPI tools                                 │ │
│  │       ▼                                                      │ │
│  │  NemoReconstruct API (:8010)                                │ │
│  │       │                                                      │ │
│  │       ▼                                                      │ │
│  │  ffmpeg → COLMAP → fVDB/frgs → PLY splat                   │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                    │                             │
│                                    ▼                             │
│                             NVIDIA GPU (Blackwell)               │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Port | Purpose |
|-----------|------|---------|
| Ollama | 11434 | Local LLM (glm-4.7-flash) |
| OpenShell Gateway | 8080 | Sandbox management + inference routing |
| NemoReconstruct Backend | 8010 | FastAPI pipeline server |
| NemoReconstruct Frontend | 3000 | Next.js dashboard (optional) |
| `inference.local` | — | In-sandbox LLM endpoint → Ollama |

---

## Using This as a Template for Your Own Project

The [full tutorial](docs/NEMOCLAW_SETUP.md) has a dedicated section (Part 2) that walks through connecting NemoClaw to **any** project:

1. **Start your service** on `0.0.0.0` (not `127.0.0.1`)
2. **Create a sandbox policy** — allow the sandbox to reach your service's port
3. **Create an OpenClaw config** — point the agent at your workspace
4. **Run the agent** — same `openshell sandbox create` pattern, just swap the paths

The NemoClaw + OpenShell infrastructure (Ollama, gateway, provider, inference routing) is set up once and reused across all projects.

---

## Notes

- Intentionally an MVP: one pipeline, one worker, SQLite, local disk.
- Designed for agent demos and livestream walkthroughs — not HA production.
- The pattern (API + OpenShell sandbox + local inference) generalizes to any tool or SDK.

