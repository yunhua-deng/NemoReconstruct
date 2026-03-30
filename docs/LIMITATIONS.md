# Known Limitations

## DGX Spark / GB10 (aarch64) Platform

### TSDF Fusion — CUDA Crash

**Affects:** Both fVDB and 3DGRUT workflows

The fVDB precompiled CUDA kernels (`fvdb.grid.integrate_tsdf_with_features()`) are not
compiled for compute capability 12.1 (NVIDIA GB10 / Blackwell). Calling TSDF fusion
results in a `cudaErrorIllegalAddress` crash.

- **fVDB `frgs mesh-dlnr`:** DLNR depth estimation completes successfully, but the
  subsequent TSDF fusion step crashes with `cudaErrorIllegalAddress`. The process
  runs for ~15 minutes before crashing, wasting pipeline time. Same crash affects
  `frgs mesh-basic`.
- **No timeout:** The `subprocess.run()` call for mesh extraction has no timeout,
  so the pipeline blocks for the full DLNR duration before the fallback triggers.
- **Legacy TSDF path (`run_tsdf_mesh_generation`):** Runs under the 3DGRUT conda env
  but still uses fVDB's TSDF kernel — same crash.
- **Mitigation:** The pipeline falls back to alpha-shape collision mesh generation
  (using `trimesh`), which works on all hardware. Jobs complete with a usable collision
  mesh (~50K faces) but without the higher-quality DLNR/TSDF mesh.
- **Resolution:** Will resolve when NVIDIA ships fVDB binaries compiled for CC 12.1.

### USDZ Conversion — Stub `pxr` Package in fVDB Env

**Affects:** fVDB workflows only (fixed)

The fVDB conda environment (Python 3.12, aarch64) ships a stub `pxr` package —
`usd-core` is not available for this platform/Python combination. `Usd.Stage.CreateInMemory()`
does not exist in the stub, causing `frgs convert` (PLY → USDZ) to crash with
`AttributeError`.

- **Fix applied:** A standalone PLY → USDZ converter (`backend/app/services/convert_ply_to_usdz.py`)
  runs under the 3DGRUT conda environment (Python 3.11), which has a working `usd-core 25.11`.
- **3DGRUT unaffected:** 3DGRUT produces USDZ natively during training (`export_usdz.enabled=true`).

### PyTorch Compute Capability Warning

PyTorch warns that CC 12.1 exceeds its maximum supported CC 12.0. JIT-compiled CUDA
extensions (e.g., 3DGRUT's ray-tracing kernels) still build and run correctly despite
the warning. The fVDB precompiled kernels are a separate issue (see TSDF above).

---

## Mesh Extraction by Backend

### fVDB

| Method | Status on GB10 | Notes |
|--------|----------------|-------|
| `frgs mesh-dlnr` (DLNR + TSDF) | **Broken** — TSDF crash | Depth estimation works; TSDF fusion crashes |
| `frgs mesh-basic` (TSDF only) | **Broken** — same TSDF crash | Same `fvdb.grid.integrate_tsdf_with_features` kernel |
| Alpha-shape (fallback) | **Works** | ~50K faces from Gaussian centroids via `trimesh` |

### 3DGRUT

| Method | Status on GB10 | Notes |
|--------|----------------|-------|
| DLNR (`frgs mesh-dlnr`) | **Not available** | DLNR requires an fVDB checkpoint; 3DGRUT does not produce one |
| Alpha-shape | **Works** | Primary mesh path for 3DGRUT |
| Legacy TSDF | **Broken** — same TSDF crash | Optional; guarded by `tsdf_mesh_enabled` flag |

3DGRUT has **no neural mesh extraction path** (no DLNR). Its collision mesh comes
exclusively from alpha-shape generation on Gaussian centroids, which is a coarser
approximation than DLNR-derived meshes.

---

## Pipeline Fallback Chain (fVDB)

When mesh extraction is requested for an fVDB workflow, the pipeline attempts methods
in this order:

1. **DLNR** (`frgs mesh-dlnr`) — neural stereo depth → TSDF fusion → marching cubes
2. **Basic** (`frgs mesh-basic`) — direct TSDF fusion → marching cubes
3. **Alpha-shape** — Gaussian centroids → alpha-shape via `trimesh`
4. **Legacy TSDF** — standalone TSDF script (last resort)

On GB10, steps 1, 2, and 4 all fail due to the TSDF kernel crash. Step 3 succeeds and
produces a usable collision mesh.

---

## Ollama on DGX Spark

`nemotron-3-nano` crashes during model load with a GGML CUDA assertion
(`ggml_nbytes(src0) <= INT_MAX`). Use `glm-4.7-flash` instead for local inference.
