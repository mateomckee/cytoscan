from dataclasses import dataclass

@dataclass
class CellFindings:
    distance_signed: float
    distance_abs: float
    side: str
    category: str

@dataclass
class FrameFindings:
    frame_index: int
    cells: list[CellFindings]   # per-cell computed values
    n_dex: int
    n_int: int
    n_peg: int
    interface_amplitude_px: float
    # or any other per-frame aggregates

@dataclass
class ExperimentFindings:
    frames: dict[int, FrameFindings]
