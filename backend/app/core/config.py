from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "NemoReconstruct"
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    redoc_url: str = "/redoc"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    base_dir: Path = Path(__file__).resolve().parents[3]
    storage_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "storage")
    database_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "nemo_reconstruct.db")
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[3] / "data")

    ffmpeg_bin: str = "ffmpeg"
    colmap_bin: str = "colmap"
    frgs_bin: str = "frgs"
    fvdb_conda_env: str = "fvdb"
    fvdb_conda_root: Path = Field(default_factory=lambda: Path.home() / "miniconda3" / "envs")

    grut_install_dir: Path = Field(default_factory=lambda: Path("/opt/3dgrut"))
    grut_conda_env: str = "3dgrut"

    frame_rate: float = 2.0
    max_upload_size_mb: int = 4096
    sequential_matcher_overlap: int = 12
    colmap_mapper_type: str = "incremental"
    colmap_max_num_features: int = 8192
    reconstruction_backend: str = "fvdb"
    fvdb_max_epochs: int = 40
    fvdb_sh_degree: int = 3
    fvdb_image_downsample_factor: int = 6
    grut_n_iterations: int = 30000
    grut_render_method: str = "3dgrt"
    grut_strategy: str = "gs"
    grut_downsample_factor: int = 2
    splat_only_mode: bool = False
    collision_mesh_enabled: bool = True
    collision_mesh_method: str = "alpha"
    collision_mesh_target_faces: int = 50000
    collision_mesh_alpha: float = 0.0  # 0 = auto
    collision_mesh_downsample: int = 4
    tsdf_mesh_enabled: bool = True
    tsdf_voxel_size: float = 0.01
    tsdf_truncation_distance: float = 0.03
    tsdf_depth_image_size: int = 1024
    tsdf_splat_radius: int = 5
    tsdf_target_faces: int = 200000
    tsdf_downsample: int = 1
    # DLNR / frgs mesh extraction (fVDB workflows)
    mesh_extraction_method: str = "dlnr"  # "dlnr", "basic", or "tsdf" (legacy)
    mesh_truncation_margin: float = 0.05
    mesh_grid_shell_thickness: float = 3.0
    mesh_dlnr_backbone: str = "middleburry"
    mesh_image_downsample_factor: int = 1
    mesh_embed_in_usdz: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="NEMO_RECONSTRUCT_",
        extra="ignore",
    )


settings = Settings()
