#!/usr/bin/env python3
"""TSDF Fusion mesh extraction from 3D Gaussian Splat / COLMAP data.

Renders depth maps from the Gaussian PLY centroids using the COLMAP training
cameras, fuses them into a Truncated Signed Distance Field (TSDF) voxel grid,
then extracts a triangle mesh via marching cubes.

This script runs inside the 3dgrut conda environment (which has torch, kaolin,
trimesh, scipy, numpy). It is invoked as a subprocess by the NemoReconstruct
pipeline.

Algorithm (KinectFusion-style volumetric TSDF integration):
    1. Load Gaussian centroids + opacities from PLY
    2. Read COLMAP camera intrinsics and extrinsics
    3. For each camera: render a depth map by GPU-accelerated point splatting
    4. Integrate each depth map into a volumetric TSDF grid
    5. Extract the zero-crossing iso-surface via marching cubes (kaolin)
    6. Post-process: clean small components, simplify to target face count

Usage:
    python tsdf_fusion.py <ply_path> <colmap_sparse_dir> <output_obj> \
        [--voxel-size 0.02] \
        [--truncation-distance 0.06] \
        [--depth-image-size 512] \
        [--splat-radius 3] \
        [--target-faces 100000] \
        [--downsample 1]
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
import trimesh


# ---------------------------------------------------------------------------
# COLMAP binary readers (self-contained — no dependency on threedgrut at import
# time so the script can also be tested standalone)
# ---------------------------------------------------------------------------

CameraInfo = namedtuple("CameraInfo", ["id", "model", "width", "height", "params"])
ImageInfo = namedtuple("ImageInfo", ["id", "qvec", "tvec", "camera_id", "name"])


def _read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)


def read_cameras_binary(path: str) -> dict[int, CameraInfo]:
    cameras: dict[int, CameraInfo] = {}
    with open(path, "rb") as f:
        num_cameras = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_props = _read_next_bytes(f, 24, "iiQQ")
            cam_id = cam_props[0]
            model_id = cam_props[1]
            width = cam_props[2]
            height = cam_props[3]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5, 6: 8, 7: 12, 8: 4, 9: 5, 10: 5}.get(model_id, 4)
            params = np.array(_read_next_bytes(f, 8 * num_params, "d" * num_params))
            cameras[cam_id] = CameraInfo(id=cam_id, model=model_id, width=width, height=height, params=params)
    return cameras


def read_images_binary(path: str) -> dict[int, ImageInfo]:
    images: dict[int, ImageInfo] = {}
    with open(path, "rb") as f:
        num_images = _read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            props = _read_next_bytes(f, 64, "idddddddi")
            image_id = props[0]
            qvec = np.array(props[1:5])
            tvec = np.array(props[5:8])
            camera_id = props[8]
            name = b""
            ch = f.read(1)
            while ch != b"\x00":
                name += ch
                ch = f.read(1)
            num_points2D = _read_next_bytes(f, 8, "Q")[0]
            # Skip 2D point data (x, y, point3D_id) for each point
            f.read(num_points2D * 24)
            images[image_id] = ImageInfo(
                id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id,
                name=name.decode("utf-8"),
            )
    return images


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec / np.linalg.norm(qvec)
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


# ---------------------------------------------------------------------------
# PLY loading
# ---------------------------------------------------------------------------

def load_gaussian_ply(ply_path: str, downsample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Load positions and opacities from a 3DGRUT/fVDB Gaussian PLY.

    Returns (positions [N,3], opacities [N]).
    """
    mesh = trimesh.load(ply_path, process=False)
    pts = np.asarray(mesh.vertices, dtype=np.float32)

    # Try to read opacity from PLY metadata
    opacities: np.ndarray
    if hasattr(mesh, "metadata") and "ply_raw" in mesh.metadata:
        ply_raw = mesh.metadata["ply_raw"]
        if "vertex" in ply_raw and "data" in ply_raw["vertex"]:
            vdata = ply_raw["vertex"]["data"]
            if "opacity" in vdata.dtype.names:
                raw_op = np.asarray(vdata["opacity"], dtype=np.float32).ravel()
                # 3DGRUT stores pre-activation opacity; apply sigmoid
                opacities = 1.0 / (1.0 + np.exp(-raw_op))
            else:
                opacities = np.ones(len(pts), dtype=np.float32)
        else:
            opacities = np.ones(len(pts), dtype=np.float32)
    else:
        opacities = np.ones(len(pts), dtype=np.float32)

    if downsample > 1 and len(pts) > downsample * 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=len(pts) // downsample, replace=False)
        pts = pts[idx]
        opacities = opacities[idx]

    return pts, opacities


# ---------------------------------------------------------------------------
# GPU depth rendering via point splatting
# ---------------------------------------------------------------------------

@torch.no_grad()
def render_depth_map(
    points: torch.Tensor,        # [N, 3] world-space positions
    opacities: torch.Tensor,     # [N]
    K: torch.Tensor,             # [3, 3] intrinsic matrix
    R: torch.Tensor,             # [3, 3] rotation (world→camera)
    t: torch.Tensor,             # [3] translation
    height: int,
    width: int,
    splat_radius: int = 3,
    opacity_threshold: float = 0.3,
) -> torch.Tensor:
    """Render a depth map by splatting 3D Gaussian centroids onto the image plane.

    Uses a soft z-buffer: each point writes its depth to a small disc of
    *splat_radius* pixels, keeping the nearest depth.  Points with opacity
    below *opacity_threshold* are culled.

    Returns a [H, W] depth tensor (0 = no observation).
    """
    device = points.device

    # Filter low-opacity points
    mask = opacities >= opacity_threshold
    pts = points[mask]
    if pts.shape[0] == 0:
        return torch.zeros(height, width, device=device)

    # Transform to camera coordinates: p_cam = R @ p_world + t
    p_cam = (R @ pts.T).T + t  # [M, 3]

    # Cull behind camera
    valid = p_cam[:, 2] > 0.01
    p_cam = p_cam[valid]
    if p_cam.shape[0] == 0:
        return torch.zeros(height, width, device=device)

    # Project to pixel coordinates
    depths = p_cam[:, 2]
    u = (K[0, 0] * p_cam[:, 0] / depths + K[0, 2]).long()
    v = (K[1, 1] * p_cam[:, 1] / depths + K[1, 2]).long()

    # Initialise z-buffer with inf
    depth_map = torch.full((height, width), float("inf"), device=device)

    # Splat each point onto a small disc
    for du in range(-splat_radius, splat_radius + 1):
        for dv in range(-splat_radius, splat_radius + 1):
            if du * du + dv * dv > splat_radius * splat_radius:
                continue
            uu = u + du
            vv = v + dv
            in_bounds = (uu >= 0) & (uu < width) & (vv >= 0) & (vv < height)
            uu_valid = uu[in_bounds]
            vv_valid = vv[in_bounds]
            d_valid = depths[in_bounds]
            # scatter_reduce min is not available everywhere; use a loop-free
            # approach: flatten, then scatter with min
            flat_idx = vv_valid * width + uu_valid
            depth_map.view(-1).scatter_reduce_(0, flat_idx, d_valid, reduce="amin", include_self=True)

    # Replace inf with 0 (no observation)
    depth_map[depth_map == float("inf")] = 0.0
    return depth_map


# ---------------------------------------------------------------------------
# TSDF volume
# ---------------------------------------------------------------------------

class TSDFVolume:
    """GPU-resident voxel grid for Truncated Signed Distance Field integration."""

    def __init__(
        self,
        origin: torch.Tensor,
        voxel_size: float,
        dims: tuple[int, int, int],
        truncation: float,
        device: torch.device,
    ) -> None:
        self.origin = origin.to(device)
        self.voxel_size = voxel_size
        self.dims = dims  # (X, Y, Z)
        self.truncation = truncation
        self.device = device

        self.tsdf = torch.ones(*dims, device=device)
        self.weights = torch.zeros(*dims, device=device)

    def _voxel_centers(self) -> torch.Tensor:
        """Return world-space coordinates of every voxel centre [X*Y*Z, 3]."""
        xi = torch.arange(self.dims[0], device=self.device, dtype=torch.float32)
        yi = torch.arange(self.dims[1], device=self.device, dtype=torch.float32)
        zi = torch.arange(self.dims[2], device=self.device, dtype=torch.float32)
        gx, gy, gz = torch.meshgrid(xi, yi, zi, indexing="ij")
        coords = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        return coords * self.voxel_size + self.origin

    @torch.no_grad()
    def integrate(
        self,
        depth_map: torch.Tensor,  # [H, W]
        K: torch.Tensor,          # [3, 3]
        R: torch.Tensor,          # [3, 3] world→camera
        t: torch.Tensor,          # [3]
    ) -> None:
        """Integrate a single depth frame into the TSDF volume."""
        H, W = depth_map.shape
        centers = self._voxel_centers()  # [V, 3]

        # Transform voxel centres to camera coords
        cam_pts = (R @ centers.T).T + t  # [V, 3]

        # Project to pixel space
        cam_z = cam_pts[:, 2]
        valid_z = cam_z > 0.01
        px = (K[0, 0] * cam_pts[:, 0] / cam_z + K[0, 2])
        py = (K[1, 1] * cam_pts[:, 1] / cam_z + K[1, 2])

        iu = px.long()
        iv = py.long()

        # Bounds check
        in_bounds = valid_z & (iu >= 0) & (iu < W) & (iv >= 0) & (iv < H)

        # Observed depth at each projected pixel
        depth_obs = torch.zeros_like(cam_z)
        depth_obs[in_bounds] = depth_map[iv[in_bounds], iu[in_bounds]]

        # SDF = observed depth − voxel depth (positive = in front of surface)
        sdf = depth_obs - cam_z

        # Only update voxels that project inside the image, have a valid
        # depth observation, and fall within the truncation band
        valid = in_bounds & (depth_obs > 0) & (sdf >= -self.truncation)
        tsdf_val = torch.clamp(sdf / self.truncation, -1.0, 1.0)

        # Weighted running average
        flat = torch.arange(centers.shape[0], device=self.device)
        idx = flat[valid]
        w_old = self.weights.view(-1)[idx]
        w_new = 1.0
        updated_tsdf = (w_old * self.tsdf.view(-1)[idx] + w_new * tsdf_val[valid]) / (w_old + w_new)
        self.tsdf.view(-1)[idx] = updated_tsdf
        self.weights.view(-1)[idx] = w_old + w_new

    def extract_mesh(self, min_weight: float = 1.0) -> trimesh.Trimesh | None:
        """Run marching cubes and return a trimesh in world coordinates."""
        from kaolin.ops.conversions import voxelgrids_to_trianglemeshes

        # Convert TSDF to an occupancy grid for marching cubes.
        # Voxels with weight < min_weight are treated as empty (value 0).
        occ = torch.zeros_like(self.tsdf)
        observed = self.weights >= min_weight
        # Map TSDF [-1,1] → occupancy [0,1].  Inside surface (negative TSDF) → 1.
        occ[observed] = (1.0 - self.tsdf[observed]) * 0.5

        # kaolin expects (batch, X, Y, Z) of type uint8 or float
        grid = occ.unsqueeze(0).float()
        verts_list, faces_list = voxelgrids_to_trianglemeshes(grid.cuda(), iso_value=0.5)

        if len(verts_list) == 0 or len(verts_list[0]) == 0:
            return None

        verts = verts_list[0].cpu().numpy()
        faces = faces_list[0].cpu().numpy()

        # Scale verts from grid indices to world coordinates
        verts = verts * self.voxel_size + self.origin.cpu().numpy()

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        return mesh


# ---------------------------------------------------------------------------
# Scene-bounds estimation
# ---------------------------------------------------------------------------

def compute_scene_bounds(points: np.ndarray, pad_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Compute padded axis-aligned bounding box of a point cloud."""
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = hi - lo
    pad = extent * pad_fraction
    return lo - pad, hi + pad


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(
    ply_path: str,
    colmap_sparse_dir: str,
    output_path: str,
    *,
    voxel_size: float = 0.02,
    truncation_distance: float = 0.06,
    depth_image_size: int = 512,
    splat_radius: int = 3,
    target_faces: int = 100000,
    downsample: int = 1,
) -> dict:
    """Run the full TSDF fusion pipeline and return metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load Gaussian centroids ───────────────────────────────────
    pts_np, opacities_np = load_gaussian_ply(ply_path, downsample=downsample)
    points = torch.from_numpy(pts_np).to(device)
    opacities = torch.from_numpy(opacities_np).to(device)

    # ── 2. Read COLMAP cameras ───────────────────────────────────────
    sparse = Path(colmap_sparse_dir)
    cameras_bin = sparse / "cameras.bin"
    images_bin = sparse / "images.bin"
    if not cameras_bin.exists() or not images_bin.exists():
        raise RuntimeError(f"COLMAP binary model not found in {colmap_sparse_dir}")

    cameras = read_cameras_binary(str(cameras_bin))
    images = read_images_binary(str(images_bin))

    if not cameras or not images:
        raise RuntimeError("No cameras/images found in COLMAP model")

    # ── 3. Build intrinsic matrices ──────────────────────────────────
    intrinsics: dict[int, tuple[torch.Tensor, int, int]] = {}
    for cam_id, cam in cameras.items():
        p = cam.params
        # Models: SIMPLE_PINHOLE(0)=[f,cx,cy], PINHOLE(1)=[fx,fy,cx,cy],
        # SIMPLE_RADIAL(2)=[f,cx,cy,k1], RADIAL(3)=[f,cx,cy,k1,k2], ...
        if cam.model in (0, 2):
            fx = fy = p[0]
            cx, cy = p[1], p[2]
        elif cam.model in (1, 4):
            fx, fy = p[0], p[1]
            cx, cy = p[2], p[3]
        else:
            # Fallback: treat first param as focal length
            fx = fy = p[0]
            cx, cy = cam.width / 2.0, cam.height / 2.0

        # Scale intrinsics if we render at a different resolution
        scale_w = depth_image_size / cam.width
        scale_h = depth_image_size / cam.height
        K = torch.tensor([
            [fx * scale_w, 0,            cx * scale_w],
            [0,            fy * scale_h, cy * scale_h],
            [0,            0,            1],
        ], dtype=torch.float32, device=device)
        intrinsics[cam_id] = (K, depth_image_size, depth_image_size)

    # ── 4. Compute TSDF grid dimensions ──────────────────────────────
    lo, hi = compute_scene_bounds(pts_np, pad_fraction=0.1)
    extent = hi - lo
    dims = tuple(int(np.ceil(e / voxel_size)) for e in extent)
    # Cap to avoid OOM on huge scenes
    MAX_DIM = 512
    if max(dims) > MAX_DIM:
        scale = MAX_DIM / max(dims)
        voxel_size = voxel_size / scale
        dims = tuple(int(np.ceil(e / voxel_size)) for e in extent)
        dims = tuple(min(d, MAX_DIM) for d in dims)
        # Scale truncation proportionally so it spans enough voxels
        truncation_distance = truncation_distance / scale

    # Ensure truncation is at least 3x voxel_size for meaningful integration
    truncation_distance = max(truncation_distance, voxel_size * 3.0)

    origin = torch.from_numpy(lo.astype(np.float32)).to(device)
    tsdf_vol = TSDFVolume(origin, voxel_size, dims, truncation_distance, device)

    # ── 5. Render depth & integrate ──────────────────────────────────
    n_integrated = 0
    for img in images.values():
        cam_info = intrinsics.get(img.camera_id)
        if cam_info is None:
            continue
        K, h, w = cam_info

        R = torch.from_numpy(qvec_to_rotmat(img.qvec).astype(np.float32)).to(device)
        t = torch.from_numpy(img.tvec.astype(np.float32)).to(device)

        depth = render_depth_map(points, opacities, K, R, t, h, w, splat_radius=splat_radius)
        if depth.sum() == 0:
            continue

        tsdf_vol.integrate(depth, K, R, t)
        n_integrated += 1

    if n_integrated == 0:
        raise RuntimeError("No depth maps were successfully rendered")

    # ── 6. Extract mesh ──────────────────────────────────────────────
    mesh = tsdf_vol.extract_mesh(min_weight=2.0)
    if mesh is None:
        raise RuntimeError("Marching cubes extracted no geometry from TSDF volume")

    # Remove small disconnected components (keep largest)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # Simplify to target face count
    original_faces = len(mesh.faces)
    if len(mesh.faces) > target_faces:
        mesh = mesh.simplify_quadric_decimation(target_faces)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)

    metrics = {
        "method": "tsdf_fusion",
        "input_points": int(len(pts_np)),
        "downsample": downsample,
        "voxel_size": voxel_size,
        "truncation_distance": truncation_distance,
        "grid_dims": list(dims),
        "depth_image_size": depth_image_size,
        "splat_radius": splat_radius,
        "cameras_used": n_integrated,
        "cameras_total": len(images),
        "original_faces": original_faces,
        "final_faces": int(len(mesh.faces)),
        "final_vertices": int(len(mesh.vertices)),
        "target_faces": target_faces,
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area": float(mesh.area),
        "output_path": output_path,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="TSDF fusion mesh from Gaussian PLY + COLMAP cameras")
    parser.add_argument("ply_path", help="Path to the 3D Gaussian PLY file")
    parser.add_argument("colmap_sparse_dir", help="Path to COLMAP sparse/0 directory")
    parser.add_argument("output_path", help="Output OBJ file path")
    parser.add_argument("--voxel-size", type=float, default=0.02,
                        help="TSDF voxel edge length in world units (default: 0.02)")
    parser.add_argument("--truncation-distance", type=float, default=0.06,
                        help="TSDF truncation distance (default: 0.06, typically 3x voxel_size)")
    parser.add_argument("--depth-image-size", type=int, default=512,
                        help="Render depth maps at this resolution (default: 512)")
    parser.add_argument("--splat-radius", type=int, default=3,
                        help="Pixel radius for point splatting (default: 3)")
    parser.add_argument("--target-faces", type=int, default=100000,
                        help="Target face count after decimation (default: 100000)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Point cloud downsampling factor (default: 1 = no downsampling)")
    args = parser.parse_args()

    metrics = generate(
        ply_path=args.ply_path,
        colmap_sparse_dir=args.colmap_sparse_dir,
        output_path=args.output_path,
        voxel_size=args.voxel_size,
        truncation_distance=args.truncation_distance,
        depth_image_size=args.depth_image_size,
        splat_radius=args.splat_radius,
        target_faces=args.target_faces,
        downsample=args.downsample,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
