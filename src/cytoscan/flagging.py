import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from cytoscan.detections import FrameDetections, FrameFlags
from cytoscan.config import ResearchConfig, FlaggingConfig, ChannelDetectionConfig
from cytoscan.channel_detector import _signed_vertical_edges, _load_gray

"""compute and attach flags to every frame in-place."""
def compute_flags_all(r_cfg: ResearchConfig, flagging_cfg: FlaggingConfig, channeld_cfg: ChannelDetectionConfig, detections: dict[int, FrameDetections]) -> None:
    for fd in detections.values():
        fd.flags = _compute_flags(r_cfg, flagging_cfg, channeld_cfg, fd)

"""takes in frame detections (Dict[int, FrameDetections]) and runs validations on the detections, adding the computed flags in-place for later, downstream checking"""
def _compute_flags(r_cfg: ResearchConfig, flagging_cfg: FlaggingConfig, channeld_cfg: ChannelDetectionConfig, fd: FrameDetections) -> FrameFlags:
    gray = _load_gray(fd.br)
    h, w = gray.shape

    # wall metrics
    sx = _signed_vertical_edges(gray)
    profile_pos = np.maximum(sx, 0.0).sum(axis=0)
    background = max(float(np.median(profile_pos)), 1.0)

    sample_ys = np.linspace(0, h - 1, 20).astype(int)
    right_wall_x_mean = int(np.clip(np.mean(np.polyval(fd.right_coeffs, sample_ys)), 0, w - 1))
    left_wall_x_mean  = int(np.clip(np.mean(np.polyval(fd.left_coeffs,  sample_ys)), 0, w - 1))
    right_wall_anchor_strength = float(profile_pos[right_wall_x_mean]) / background
    left_wall_anchor_strength  = float(profile_pos[left_wall_x_mean])  / background

    # channel width (diagnostic)
    widths_px = np.polyval(fd.right_coeffs, sample_ys) - np.polyval(fd.left_coeffs, sample_ys)
    mean_channel_width_um = float(np.mean(widths_px) * r_cfg.pixel_size_um)

    # interface metrics
    interface_signal_ratio, interface_residual_mad_px = _interface_metrics(
        gray, fd, channeld_cfg.interface_ridge_sigma_px
    )

    return FrameFlags(
        right_wall_anchor_strength    = right_wall_anchor_strength,
        left_wall_anchor_strength     = left_wall_anchor_strength,
        interface_signal_ratio        = interface_signal_ratio,
        interface_residual_mad_px     = interface_residual_mad_px,
        mean_channel_width_um         = mean_channel_width_um,
        wall_anchor_strength_min      = flagging_cfg.wall_anchor_strength_min,
        interface_signal_ratio_min    = flagging_cfg.interface_signal_ratio_min,
        interface_residual_mad_max_px = flagging_cfg.interface_residual_mad_max_px,
    )

def _interface_metrics(gray: np.ndarray, fd: FrameDetections, sigma: float) -> tuple[float, float]:
    """return (signal_ratio, residual_mad_px). signal_ratio is the median ridge
    response along the spline path divided by the median ridge response across
    the inset strip-a real interface lives on the strongest ridge, so the
    ratio is well above 1; locks on noise sit near 1. residual_mad_px is the
    robust std of (DP path points − spline) and catches scattered/jumpy paths
    even when the ridge response is strong."""
    if fd.interface_curve is None or len(fd.interface_points) == 0:
        return 0.0, float("inf")

    h, w = gray.shape
    ridge = np.abs(gaussian_filter1d(gray.astype(np.float32), sigma=sigma, order=2, axis=1))

    inset = fd.wall_inset
    left_in  = fd.left_coeffs.astype(np.float64).copy();  left_in[-1]  += inset
    right_in = fd.right_coeffs.astype(np.float64).copy(); right_in[-1] -= inset

    ys_all = np.arange(h)
    xls = np.clip(np.polyval(left_in,  ys_all).astype(int), 0, w - 1)
    xrs = np.clip(np.polyval(right_in, ys_all).astype(int), 0, w - 1)

    path_xs = np.clip(fd.interface_curve(ys_all).astype(int), 0, w - 1)
    path_response = ridge[ys_all, path_xs]
    path_median = float(np.median(path_response))

    chunks = [ridge[y, xls[y]:xrs[y]] for y in range(h) if xrs[y] > xls[y]]
    strip_response = np.concatenate(chunks) if chunks else np.array([1.0], dtype=np.float32)
    strip_median = max(float(np.median(strip_response)), 1e-9)
    signal_ratio = path_median / strip_median

    pts_y = fd.interface_points[:, 0]
    pts_x = fd.interface_points[:, 1]
    res = pts_x - fd.interface_curve(pts_y)
    residual_mad_px = float(np.median(np.abs(res - np.median(res))) * 1.4826)

    return signal_ratio, residual_mad_px

def print_flags(detections: dict[int, FrameDetections]) -> None:
    print(f"{'frame':<9}{'wallL_str':>11}{'wallR_str':>11}"
          f"{'iface_sig':>11}{'iface_mad':>11}"
          f"{'width_um':>11}{'valid':>8}")
    for i, fd in sorted(detections.items()):
        f = fd.flags
        print(f"frame{i:03d} "
              f"{f.left_wall_anchor_strength:>11.2f}"
              f"{f.right_wall_anchor_strength:>11.2f}"
              f"{f.interface_signal_ratio:>11.2f}"
              f"{f.interface_residual_mad_px:>11.2f}"
              f"{f.mean_channel_width_um:>11.1f}"
              f"{str(f.frame_valid):>8}")

