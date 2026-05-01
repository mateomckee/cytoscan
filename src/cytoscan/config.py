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

    channel_interface_smoothing_factor: float = 32.0

class FlaggingConfig(BaseModel):
    interface_curve_amplitude_max: float = 30.0   # max detrended max-min of the interface spline (px)
    expected_channel_width_um: float = 600.0
    channel_width_tolerance: float = 0.20

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

    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    flagging: FlaggingConfig = Field(default_factory=FlaggingConfig)

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
        
