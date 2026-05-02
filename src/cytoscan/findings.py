from dataclasses import dataclass
from typing import Literal

Side     = Literal["peg", "dex"]
Category = Literal["peg", "int_peg", "int", "int_dex", "dex"]

@dataclass
class CellFindings:
    # raw detection (carried so researchers can chart positions, compute
    # before/after speeds, run cell tracking later)
    centroid_x:           float
    centroid_y:           float
    area:                 int
    label:                int

    # derived geometry
    interface_x_at_y_px:  float   # spline(centroid_y), for plotting
    distance_signed_um:   float   # negative = left of interface, positive = right
    distance_abs_um:      float
    side:                 Side
    category:             Category

@dataclass
class FrameFindings:
    frame_index:          int
    cells:                list[CellFindings]
    n_peg:                int
    n_int_peg:            int
    n_int:                int
    n_int_dex:            int
    n_dex:                int
    mean_channel_width_um: float

@dataclass
class ExperimentFindings:
    frames:                  dict[int, FrameFindings]   # only valid frames
    invalid_frame_indices:   list[int]                  # skipped, logged for audit
    n_total_frames:          int                        # detections.keys() count before filtering
