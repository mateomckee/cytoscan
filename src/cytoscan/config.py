import sys
import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ValidationError

class ResearchConfig(BaseModel) :
    #required, no defaults
    pixel_size_um: float
    cell_diameter_um: float
    channel_width_um: float
    left_fluid: Literal["peg", "dex"]

class PreprocessConfig(BaseModel) :
    clear_existing: bool = True
    snr_threshold: float = 4.0                  # smoothed-profile peak must exceed this × median

class CellDetectionConfig(BaseModel):
    threshold: int = 100     # intensity 0–255 for binary mask of fluorescent frame

class ChannelDetectionConfig(BaseModel) :
    channel_wall_degree: int = 2
    channel_wall_base_inset: int = 15
    channel_wall_max_inset_fraction: float = 0.2

    channel_width_search_tolerance: float = 0.20  # +/- fraction of expected width to search for the right-wall anchor
    wall_parallelism_search_fraction: float = 0.05  # +/- fraction of expected width for per-row left-wall refinement
    wall_strip_half_width: int = 30               # px around the right-wall anchor for polynomial fit

    channel_interface_smoothing_factor: float = 100.0
    interface_ridge_sigma_px: float = 2.0         # gaussian sigma for the 1D ridge filter (matches line thickness)
    interface_dp_jump_penalty: float = 1.0        # cost per pixel of column jump between adjacent rows in the DP
    interface_dp_max_jump_px: int = 3             # hard cap on per-row column jump in the DP

class FlaggingConfig(BaseModel):
    wall_anchor_strength_min:      float = 1.8    # min ratio of profile_pos at wall-x to its image-wide median
    interface_signal_ratio_min:    float = 3.0    # min ratio of median ridge response along the spline path to the median in the inset strip
    interface_residual_mad_max_px: float = 4.0    # max robust std of (DP path points, spline) residuals (px)

class AnalysisConfig(BaseModel):
    interface_band_um:           float = 1.0                   # |distance| <= this → category "int"
    transition_band_um:          float = 50.0                   # interface_band_um < |distance| <= this → "int_peg" / "int_dex"
    interface_sample_step_px:    int = 10                        # sample step (in y) for the long-format interface.csv

class ExportVisualsConfig(BaseModel) :
    enabled: bool = True

    clear_existing: bool = True
    overwrite_existing: bool = True
    exported_frame: Literal["brightfield", "fluorescent", "mixed"] = "brightfield"

    print_logo: bool = True
    cells: bool = True
    channel_walls: bool = True
    channel_walls_inset: bool = True
    channel_interface: bool = True

class ExportDataConfig(BaseModel) :
    enabled: bool = True

class Config(BaseModel):
    research: ResearchConfig

    #optional, has defaults
    preprocessing: PreprocessConfig = Field(default_factory=PreprocessConfig)
    cell_detection: CellDetectionConfig = Field(default_factory=CellDetectionConfig)
    channel_detection: ChannelDetectionConfig = Field(default_factory=ChannelDetectionConfig)
    flagging: FlaggingConfig = Field(default_factory=FlaggingConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    export_visuals: ExportVisualsConfig = Field(default_factory=ExportVisualsConfig)
    export_data: ExportDataConfig = Field(default_factory=ExportDataConfig)

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
        # custom checks
        if self.research.pixel_size_um <= 0.0 :
            sys.exit("[cytoscan] pixel_size_um must be a positive decimal value");
        if self.research.cell_diameter_um <= 0.0 :
            sys.exit("[cytoscan] cell_diameter_um must be a positive decimal value")
        if self.research.channel_width_um <= 0.0 :
            sys.exit("[cytoscan] channel_width_um must be a positive decimal value");

