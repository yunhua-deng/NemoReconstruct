"""Standalone PLY-to-USDZ converter for fVDB Gaussian Splat models.

This script runs under the 3DGRUT conda env (which has a working usd-core) and
converts fVDB-format PLY files to NuRec-compatible USDZ without needing the fvdb
Python package.  It recreates the same serialisation logic used by
``fvdb_reality_capture.tools._export_splats_to_usdz`` but reads raw PLY fields
via plyfile + numpy.

Usage::

    /path/to/3dgrut/python convert_ply_to_usdz.py INPUT.ply OUTPUT.usdz

Exit codes: 0 = success, 1 = error.
"""

from __future__ import annotations

import gzip
import io
import logging
import os
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import msgpack
import numpy as np
import plyfile
from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils, UsdVol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiny helpers ported from fvdb_reality_capture
# ---------------------------------------------------------------------------

@dataclass
class NamedSerialized:
    filename: str
    serialized: bytes

    def save_to_zip(self, zf: zipfile.ZipFile) -> None:
        zf.writestr(self.filename, self.serialized)


@dataclass
class NamedUSDStage:
    filename: str
    stage: Usd.Stage

    def save_to_zip(self, zf: zipfile.ZipFile) -> None:
        with tempfile.NamedTemporaryFile(suffix=self.filename, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            self.stage.GetRootLayer().Export(tmp_path)
            with open(tmp_path, "rb") as fh:
                zf.writestr(self.filename, fh.read())
        finally:
            os.unlink(tmp_path)


def _init_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    stage.SetMetadata("metersPerUnit", 1)
    stage.SetMetadata("upAxis", "Z")
    world = "/World"
    UsdGeom.Xform.Define(stage, world)
    stage.SetMetadata("defaultPrim", world[1:])
    return stage


# ---------------------------------------------------------------------------
# Template construction (matches fill_3dgut_template)
# ---------------------------------------------------------------------------

def _build_template(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    densities: np.ndarray,
    features_albedo: np.ndarray,
    features_specular: np.ndarray,
    n_active_features: int,
) -> dict:
    dtype = np.float16

    template: dict = {
        "nre_data": {
            "version": "0.2.576",
            "model": "nre",
            "config": {
                "layers": {
                    "gaussians": {
                        "name": "sh-gaussians",
                        "device": "cuda",
                        "density_activation": "sigmoid",
                        "scale_activation": "exp",
                        "rotation_activation": "normalize",
                        "precision": 16,
                        "particle": {
                            "density_kernel_planar": False,
                            "density_kernel_degree": 2,
                            "density_kernel_density_clamping": True,
                            "density_kernel_min_response": 0.0113,
                            "radiance_sph_degree": 3,
                        },
                        "transmittance_threshold": 0.0001,
                    }
                },
                "renderer": {
                    "name": "3dgut-nrend",
                    "log_level": 3,
                    "force_update": False,
                    "update_step_train_batch_end": False,
                    "per_ray_features": False,
                    "global_z_order": True,
                    "projection": {
                        "n_rolling_shutter_iterations": 5,
                        "ut_dim": 3,
                        "ut_alpha": 1.0,
                        "ut_beta": 2.0,
                        "ut_kappa": 0.0,
                        "ut_require_all_sigma_points": False,
                        "image_margin_factor": 0.1,
                        "min_projected_ray_radius": 0.5477225575051661,
                    },
                    "culling": {
                        "rect_bounding": True,
                        "tight_opacity_bounding": True,
                        "tile_based": True,
                        "near_clip_distance": 0.2,
                        "far_clip_distance": 3.402823466e38,
                    },
                    "render": {"mode": "kbuffer", "k_buffer_size": 0},
                },
                "name": "gaussians_primitive",
                "appearance_embedding": {
                    "name": "skip-appearance",
                    "embedding_dim": 0,
                    "device": "cuda",
                },
                "background": {
                    "name": "skip-background",
                    "device": "cuda",
                    "composite_in_linear_space": False,
                },
            },
            "state_dict": {
                "._extra_state": {"obj_track_ids": {"gaussians": []}},
            },
        }
    }

    sd = template["nre_data"]["state_dict"]
    prefix = ".gaussians_nodes.gaussians."

    def _store(name: str, arr: np.ndarray) -> None:
        sd[prefix + name] = arr.astype(dtype).tobytes()
        sd[prefix + name + ".shape"] = list(arr.shape)

    _store("positions", positions)
    _store("rotations", rotations)
    _store("scales", scales)
    _store("densities", densities)
    _store("features_albedo", features_albedo)
    _store("features_specular", features_specular)

    extra = np.zeros((positions.shape[0], 0), dtype=dtype)
    _store("extra_signal", extra)

    sd[prefix + "n_active_features"] = np.array(
        [n_active_features], dtype=np.int64
    ).tobytes()
    sd[prefix + "n_active_features.shape"] = []

    return template


# ---------------------------------------------------------------------------
# USD stage creation (matches _serialize_nurec_usd + serialize_usd_default_layer)
# ---------------------------------------------------------------------------

def _build_gauss_stage(
    model_filename: str, positions: np.ndarray
) -> NamedUSDStage:
    stage = _init_stage()

    mn = positions.min(axis=0)
    mx = positions.max(axis=0)

    render_settings = {
        "rtx:rendermode": "RaytracedLighting",
        "rtx:directLighting:sampledLighting:samplesPerPixel": 8,
        "rtx:post:histogram:enabled": False,
        "rtx:post:registeredCompositing:invertToneMap": True,
        "rtx:post:registeredCompositing:invertColorCorrection": True,
        "rtx:material:enableRefraction": False,
        "rtx:post:tonemap:op": 2,
        "rtx:raytracing:fractionalCutoutOpacity": False,
        "rtx:matteObject:visibility:secondaryRays": True,
    }
    stage.SetMetadataByDictKey("customLayerData", "renderSettings", render_settings)

    gauss_path = "/World/gauss"
    vol = UsdVol.Volume.Define(stage, gauss_path)
    prim = vol.GetPrim()

    conv = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
        dtype=float,
    )
    vol.AddTransformOp().Set(Gf.Matrix4d(*conv.flatten()))

    prim.CreateAttribute("omni:nurec:isNuRecVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateAttribute("omni:nurec:useProxyTransform", Sdf.ValueTypeNames.Bool).Set(False)

    nurec_rel = "./" + model_filename

    for field_name, field_type, role in [
        ("density", "float", "density"),
        ("emissiveColor", "float3", "emissiveColor"),
    ]:
        fp = f"{gauss_path}/{field_name}_field"
        fd = stage.DefinePrim(fp, "OmniNuRecFieldAsset")
        vol.CreateFieldRelationship(field_name, fp)
        fd.CreateAttribute("filePath", Sdf.ValueTypeNames.Asset).Set(nurec_rel)
        fd.CreateAttribute("fieldName", Sdf.ValueTypeNames.Token).Set(field_name)
        fd.CreateAttribute("fieldDataType", Sdf.ValueTypeNames.Token).Set(field_type)
        fd.CreateAttribute("fieldRole", Sdf.ValueTypeNames.Token).Set(role)

    ecf = stage.GetPrimAtPath(f"{gauss_path}/emissiveColor_field")
    ecf.CreateAttribute("omni:nurec:ccmR", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1, 0, 0, 0))
    ecf.CreateAttribute("omni:nurec:ccmG", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 1, 0, 0))
    ecf.CreateAttribute("omni:nurec:ccmB", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(0, 0, 1, 0))

    prim.GetAttribute("extent").Set(
        [[float(mn[0]), float(mn[1]), float(mn[2])],
         [float(mx[0]), float(mx[1]), float(mx[2])]]
    )
    prim.CreateAttribute("omni:nurec:offset", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3d(0, 0, 0)
    )
    prim.CreateAttribute("omni:nurec:crop:minBounds", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3d(float(mn[0]), float(mn[1]), float(mn[2]))
    )
    prim.CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(
        Gf.Vec3d(float(mx[0]), float(mx[1]), float(mx[2]))
    )
    prim.CreateRelationship("proxy")

    return NamedUSDStage(filename="gauss.usda", stage=stage)


def _build_default_stage(gauss: NamedUSDStage) -> NamedUSDStage:
    stage = _init_stage()
    _ = UsdUtils.CoalescingDiagnosticDelegate()  # silence dangling-ref warnings
    stem = Path(gauss.filename).stem
    prim = stage.OverridePrim(f"/World/{stem}")
    prim.GetReferences().AddReference(gauss.filename)

    gl = gauss.stage.GetRootLayer()
    if "renderSettings" in gl.customLayerData:
        stage.SetMetadataByDictKey(
            "customLayerData", "renderSettings", gl.customLayerData["renderSettings"]
        )
    return NamedUSDStage(filename="default.usda", stage=stage)


# ---------------------------------------------------------------------------
# PLY reading
# ---------------------------------------------------------------------------

def _read_ply(path: str) -> dict[str, np.ndarray]:
    """Extract Gaussian Splat parameters from an fVDB-format PLY."""
    data = plyfile.PlyData.read(path)
    v = data["vertex"]

    positions = np.column_stack([v["x"], v["y"], v["z"]])
    rotations = np.column_stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]])
    scales = np.column_stack([v["scale_0"], v["scale_1"], v["scale_2"]])
    densities = np.array(v["opacity"]).reshape(-1, 1)

    sh0 = np.column_stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]])

    # Collect f_rest_* (higher-order SH)
    rest_names = sorted(
        [p.name for p in v.properties if p.name.startswith("f_rest_")],
        key=lambda n: int(n.split("_")[-1]),
    )
    if rest_names:
        shN = np.column_stack([v[n] for n in rest_names])
    else:
        shN = np.zeros((len(positions), 0), dtype=np.float32)

    n_sh = (sh0.shape[1] + shN.shape[1]) // 3  # total SH bases per channel

    return {
        "positions": positions.astype(np.float32),
        "rotations": rotations.astype(np.float32),
        "scales": scales.astype(np.float32),
        "densities": densities.astype(np.float32),
        "features_albedo": sh0.astype(np.float32),
        "features_specular": shN.astype(np.float32),
        "n_active_features": int(n_sh),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(input_ply: str, output_usdz: str) -> None:
    out = Path(output_usdz).with_suffix(".usdz")

    logger.info("Reading PLY %s …", input_ply)
    fields = _read_ply(input_ply)
    logger.info(
        "Loaded %d Gaussians, SH bases=%d",
        fields["positions"].shape[0],
        fields["n_active_features"],
    )

    template = _build_template(**fields)

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=0) as gz:
        gz.write(msgpack.packb(template))

    nurec_name = out.stem + ".nurec"
    model_file = NamedSerialized(filename=nurec_name, serialized=buf.getvalue())

    gauss_usd = _build_gauss_stage(nurec_name, fields["positions"])
    default_usd = _build_default_stage(gauss_usd)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        default_usd.save_to_zip(zf)
        model_file.save_to_zip(zf)
        gauss_usd.save_to_zip(zf)

    logger.info("USDZ written to %s (%d bytes)", out, out.stat().st_size)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} INPUT.ply OUTPUT.usdz", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
