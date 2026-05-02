from dataclasses import dataclass
from typing import Literal

Side     = Literal["peg", "dex"]
Category = Literal["peg", "int_peg", "int", "int_dex", "dex"]

@dataclass
class CellFindings:
    # raw detection in pixel space
    centroid_x:           float
    centroid_y:           float
    area:                 int
    label:                int

    # derived geometry
    interface_x_at_y_px:                float   # spline(centroid_y), for plotting
    distance_signed_um:                 float   # negative = left of interface, positive = right
    distance_abs_um:                    float
    side:                               Side
    category:                           Category

    # physical-world coords for cell tracking / speed calcs
    # x: per-frame, per-y channel midpoint (= midway between detected walls at this y) is x=0
    # y: image_h/2 is y=0; +y goes down (image convention) so it matches centroid_y_px
    centroid_x_um_from_channel_center:  float
    centroid_y_um_from_image_center:    float

@dataclass
class InterfaceSample:
    """One (y, x) sample of the interface spline at evenly-spaced y values."""
    y_px:                          float
    x_px:                          float
    y_um_from_image_center:        float
    x_um_from_channel_center:      float
    slope_dx_dy:                   float   # spline derivative at this y (dimensionless)

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

    # interface shape (long format) — sampled via analysis.interface_sample_step_px
    interface_samples:    list[InterfaceSample]

    # interface summary stats (wide format, per-frame, all in µm relative to channel center)
    interface_mean_x_um:        float
    interface_std_x_um:         float    # waviness (std of x along the curve)
    interface_amplitude_um:     float    # peak-to-peak swing (max - min)
    interface_slope_dx_dy:      float    # linear fit slope (dimensionless tilt)

@dataclass
class ExperimentFindings:
    frames:                  dict[int, FrameFindings]   # only valid frames
    invalid_frame_indices:   list[int]                  # skipped, logged for audit
    n_total_frames:          int                        # detections.keys() count before filtering
