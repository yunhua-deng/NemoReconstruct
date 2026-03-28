#!/usr/bin/env python3
"""Generate a collision mesh from a 3D Gaussian PLY point cloud.

This script runs inside the 3dgrut conda environment (which has trimesh,
scipy, numpy, and fast-simplification). It is invoked as a subprocess by
the NemoReconstruct pipeline.

Usage:
    python generate_collision_mesh.py <ply_path> <output_obj> \
        [--method alpha|convex_hull] \
        [--target-faces N] \
        [--alpha VALUE] \
        [--downsample N]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, Delaunay


def load_gaussian_centroids(ply_path: str, downsample: int = 1) -> np.ndarray:
    """Load vertex positions (Gaussian centroids) from a PLY file."""
    cloud = trimesh.load(ply_path, process=False)
    if hasattr(cloud, "vertices"):
        pts = np.asarray(cloud.vertices)
    else:
        raise RuntimeError(f"Cannot extract vertices from {ply_path}")

    if downsample > 1 and len(pts) > downsample * 100:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=len(pts) // downsample, replace=False)
        pts = pts[idx]
    return pts


def alpha_shape_3d(points: np.ndarray, alpha: float) -> trimesh.Trimesh | None:
    """Compute the 3D alpha shape of a point cloud.

    Uses Delaunay tetrahedralization, filters tetrahedra by max edge length
    (< 1/alpha), then extracts boundary faces.
    """
    tri = Delaunay(points)
    simplices = tri.simplices
    tetra_pts = points[simplices]

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge_lens = np.stack(
        [np.linalg.norm(tetra_pts[:, i] - tetra_pts[:, j], axis=1) for i, j in pairs]
    )
    max_edge = edge_lens.max(axis=0)

    mask = max_edge < (1.0 / alpha)
    filtered = simplices[mask]

    if len(filtered) == 0:
        return None

    face_tuples: list[tuple[int, ...]] = []
    for tet in filtered:
        face_tuples.extend(
            [
                tuple(sorted([tet[0], tet[1], tet[2]])),
                tuple(sorted([tet[0], tet[1], tet[3]])),
                tuple(sorted([tet[0], tet[2], tet[3]])),
                tuple(sorted([tet[1], tet[2], tet[3]])),
            ]
        )
    face_counts = Counter(face_tuples)
    boundary = np.array([list(f) for f, c in face_counts.items() if c == 1])

    if len(boundary) == 0:
        return None

    return trimesh.Trimesh(vertices=points, faces=boundary, process=True)


def auto_alpha(points: np.ndarray) -> float:
    """Estimate a reasonable alpha from mean nearest-neighbor distance."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points[:5000] if len(points) > 5000 else points)
    dists, _ = tree.query(tree.data, k=2)
    mean_nn = np.mean(dists[:, 1])
    # alpha = 1 / (3 * mean_nn) gives a good default for most scenes
    return 1.0 / (3.0 * mean_nn)


def convex_hull_mesh(points: np.ndarray) -> trimesh.Trimesh:
    """Compute the convex hull of a point cloud."""
    hull = ConvexHull(points)
    return trimesh.Trimesh(vertices=points, faces=hull.simplices, process=True)


def simplify_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate a mesh to the target face count."""
    if len(mesh.faces) <= target_faces:
        return mesh
    ratio = 1.0 - (target_faces / len(mesh.faces))
    return mesh.simplify_quadric_decimation(ratio)


def generate(
    ply_path: str,
    output_path: str,
    method: str = "alpha",
    target_faces: int = 50000,
    alpha: float | None = None,
    downsample: int = 4,
) -> dict:
    """Generate a collision mesh and return metrics."""
    pts = load_gaussian_centroids(ply_path, downsample=downsample)

    if method == "convex_hull":
        mesh = convex_hull_mesh(pts)
    else:
        if alpha is None:
            alpha = auto_alpha(pts)
        mesh = alpha_shape_3d(pts, alpha)
        if mesh is None:
            # Fallback to convex hull
            mesh = convex_hull_mesh(pts)
            method = "convex_hull_fallback"

    original_faces = len(mesh.faces)
    mesh = simplify_mesh(mesh, target_faces)

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)

    metrics = {
        "method": method,
        "input_points": len(pts),
        "original_faces": original_faces,
        "final_faces": len(mesh.faces),
        "final_vertices": len(mesh.vertices),
        "target_faces": target_faces,
        "alpha": alpha,
        "downsample": downsample,
        "watertight": bool(mesh.is_watertight),
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area": float(mesh.area),
        "output_path": output_path,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate collision mesh from Gaussian PLY")
    parser.add_argument("ply_path", help="Path to the 3D Gaussian PLY file")
    parser.add_argument("output_path", help="Output OBJ file path")
    parser.add_argument("--method", choices=["alpha", "convex_hull"], default="alpha")
    parser.add_argument("--target-faces", type=int, default=50000)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()

    metrics = generate(
        ply_path=args.ply_path,
        output_path=args.output_path,
        method=args.method,
        target_faces=args.target_faces,
        alpha=args.alpha,
        downsample=args.downsample,
    )
    # Output metrics as JSON to stdout for the pipeline to capture
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
