import numpy as np
import cv2
from cytoscan.detections import FrameDetections, FrameFlags
from cytoscan.config import FlaggingConfig, ExperimentConfig

def compute_flags_all(flagging_cfg: FlaggingConfig,
                      experiment_cfg: ExperimentConfig,
                      detections: dict[int, FrameDetections]) -> None:
    """Compute and attach flags to every frame in-place."""
    for fd in detections.values():
        fd.flags = _compute_flags(flagging_cfg, experiment_cfg, fd)

def _compute_flags(flagging_cfg: FlaggingConfig,
                   experiment_cfg: ExperimentConfig,
                   fd: FrameDetections) -> FrameFlags:
    h = cv2.imread(str(fd.br)).shape[0]

    # ---- walls: data in both top half AND bottom half ----
    midpoint = h / 2

    def _spans_both_halves(centers: np.ndarray) -> bool:
        ys = centers[:, 0]
        return bool(ys.min() < midpoint and ys.max() > midpoint)

    left_spans_full  = _spans_both_halves(fd.left_centers)
    right_spans_full = _spans_both_halves(fd.right_centers)

    # ---- channel width ----
    sample_ys = np.linspace(0, h - 1, 20).astype(int)
    left_x  = np.polyval(fd.left_coeffs,  sample_ys)
    right_x = np.polyval(fd.right_coeffs, sample_ys)
    mean_channel_width_um = float(np.mean(right_x - left_x)) * experiment_cfg.pixel_size_um
    channel_width_valid = abs(mean_channel_width_um / flagging_cfg.expected_channel_width_um - 1.0) \
                          <= flagging_cfg.channel_width_tolerance

    # ---- interface: detrended curve amplitude ----
    if fd.interface_curve is not None and len(fd.interface_points) > 0:
        y_eval = np.linspace(fd.interface_points[:, 0].min(),
                             fd.interface_points[:, 0].max(), 200)
        curve_x = fd.interface_curve(y_eval)
        trend = np.polyfit(y_eval, curve_x, deg=1)
        detrended = curve_x - np.polyval(trend, y_eval)
        interface_curve_amplitude = float(detrended.max() - detrended.min())
    else:
        interface_curve_amplitude = float("inf")
    interface_valid = interface_curve_amplitude <= flagging_cfg.interface_curve_amplitude_max

    return FrameFlags(
        left_spans_full=left_spans_full,
        right_spans_full=right_spans_full,
        mean_channel_width_um=mean_channel_width_um,
        channel_width_valid=channel_width_valid,
        interface_curve_amplitude=interface_curve_amplitude,
        interface_valid=interface_valid,
    )

def print_flags(detections: dict[int, FrameDetections]) -> None:
    print(f"{'frame':<10}{'wallL':>7}{'wallR':>7}{'amp':>9}{'width_um':>11}{'valid':>8}")
    for i, fd in sorted(detections.items()):
        f = fd.flags
        print(f"frame{i:03d}  "
              f"{'✓' if f.left_spans_full else '✗':>5}  "
              f"{'✓' if f.right_spans_full else '✗':>5}  "
              f"{f.interface_curve_amplitude:>7.1f}  "
              f"{f.mean_channel_width_um:>9.1f}  "
              f"{str(f.frame_valid):>6}")


