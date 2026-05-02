import sys
import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class ExportDataConfig(BaseModel) :
    enabled: bool = True

class ExportVisualsConfig(BaseModel) :
    enabled: bool = True

    clear_existing: bool = False
    overwrite_existing: bool = True
    exported_frame: Literal["brightfield", "fluorescent", "mixed"] = "brightfield"

    cells: bool = True
    channel_walls: bool = True
    channel_walls_inset: bool = True
    channel_interface: bool = True
    roi: bool = True

class OutputConfig(BaseModel) :
    export_visuals: ExportVisualsConfig = Field(default_factory=ExportVisualsConfig)
    export_data: ExportDataConfig = Field(default_factory=ExportDataConfig)

class DetectionConfig(BaseModel) :
    channel_wall_base_inset: int = 10
    channel_wall_max_inset_fraction: float = 0.175
    channel_wall_degree: int = 2

    expected_channel_width_um: float = 600.0      # prior used to anchor the left wall to the right wall
    channel_width_search_tolerance: float = 0.20  # ± fraction of expected width to search for the right-wall anchor
    wall_parallelism_search_fraction: float = 0.05  # ± fraction of expected width for per-row left-wall refinement
    wall_strip_half_width: int = 30               # px around the right-wall anchor for polynomial fit

    channel_interface_smoothing_factor: float = 32.0
    interface_ridge_sigma_px: float = 2.0         # gaussian sigma for the 1D ridge filter (matches line thickness)
    interface_dp_jump_penalty: float = 1.0        # cost per pixel of column jump between adjacent rows in the DP
    interface_dp_max_jump_px: int = 3             # hard cap on per-row column jump in the DP

class AnalysisConfig(BaseModel):
    left_fluid:           Literal["peg", "dex"] = "peg"   # which fluid sits on the left of the interface
    interface_band_um:    float = 50.0                    # |distance| ≤ this → category "int"
    transition_band_um:   float = 150.0                   # interface_band_um < |distance| ≤ this → "int_peg" / "int_dex"

class FlaggingConfig(BaseModel):
    wall_anchor_strength_min:      float = 2.5    # min ratio of profile_pos at wall-x to its image-wide median
    interface_signal_ratio_min:    float = 1.5    # min ratio of median ridge response along the spline path to the median in the inset strip
    interface_residual_mad_max_px: float = 5.0    # max robust std of (DP path points − spline) residuals (px)

class PreprocessConfig(BaseModel) :
    clear_existing: bool = True

    expected_channel_width_um: float = 600.0    # used to set the smoothing scale for center detection
    snr_threshold: float = 1.5                  # smoothed-profile peak must exceed this × median

class ExperimentConfig(BaseModel):
    dir: Path
    pixel_size_um: float

class IlastikConfig(BaseModel):
    model: Path
    exe: Path

class Config(BaseModel):
    #required, has no default
    ilastik: IlastikConfig
    experiment: ExperimentConfig

    preprocessing: PreprocessConfig = Field(default_factory=PreprocessConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    flagging: FlaggingConfig = Field(default_factory=FlaggingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    #optional groups, has a default
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path:str) -> "Config" :
        try :
            with open(path) as f :
                raw = yaml.safe_load(f) or {}
        except FileNotFoundError :
            sys.exit(f"[cytoscan] config file not found: {path}")
        except yaml.YAMLError as e :
            sys.exit(f"[cytoscan] invalid YAML in {path}:\n{e}")

        try :
            cfg = cls(**raw)
        except ValidationError as e :
            sys.exit(f"[cytoscan] config error in {path}:\n{e}")

        cfg._validate_paths(path)
        return cfg

    def _validate_paths(self, src: str) -> None :
        """ Custom checks
            
            TODO:
            wall_inset >= 0
            min_area < max_area
            ...
            etc
        """
        if not self.ilastik.exe.exists() :
            sys.exit(f"[cytoscan] ilastik_exe does not exist: {self.ilastik_exe}")
        if not self.ilastik.model.exists() :
            sys.exit(f"[cytoscan] ilastik_model does not exist: {self.ilastik_model}")
        if not self.experiment.dir.is_dir() :
            sys.exit(f"[cytoscan] experiment is not a directory: {self.experiment}")
        
