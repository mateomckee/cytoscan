import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

from cytoscan.config import ResearchConfig, ChannelDetectionConfig

"""
this module handles all channel-related detection for this lab. channels are small microfluidic tubes composed of a left/right wall, and the microfluidic interface that separates 2 liquids inside.
due to the design of the physical microfluidic device imaged under the microscope, the right wall is appears much more consistently and clearly than the left. So wall detect detects the right wall first
as the anchor of the channel, then setting the left wall parallel to the right wall offset by the known channel dimensions (provided by the researcher, typically 600 micrometers).

1. interface detection is complex, as it is a wavy, unpredictable and feint line in between the channel walls. it is done by checking the inter-wall region for the strongest change in gradient, that best
followws a line.

2. channel walls are stored as 2 degree polynomials, and the interface line is stored as a spline curve

AI usage: all detection logic was AI genereated as it required pretty complex CV algorithms/mathematical understanding to implement. however, all pipeline and design choices were made by me as 
the AI does not know what the lab needs/how it works in reality. after the heavy algorithmic work was done, i put together the rest of the pieces.
"""

def _load_gray(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _abs_vertical_edges(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))

def _signed_vertical_edges(gray):
    """Signed ∂I/∂x. Positive = brighter on the right of the pixel; negative =
    darker on the right. The inner (channel-facing) edge of either wall is a
    transition from wall-intensity to channel-intensity. In our setup the
    channel interior is medium gray; the LEFT wall reads as a *dark* band, so
    its inner edge is dark→gray = positive gradient. The RIGHT wall reads as a
    *bright* band with a dark border on the channel side, so its inner edge is
    also dark→bright = positive gradient. So `grad_pos` reliably fires on both
    inner edges in this dataset, and using it for both walls keeps them pinned
    to the channel boundary regardless of band polarity."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)

def _find_anchor(profile, target_x, tol_px, w, frame_path):
    lo = int(max(0, target_x - tol_px))
    hi = int(min(w, target_x + tol_px))
    if hi <= lo:
        raise RuntimeError(f"wall search window degenerate at target={target_x:.0f} (cw={w}) in {frame_path}")
    return lo + int(np.argmax(profile[lo:hi]))

def _fit_wall_at(gray, grad, anchor_x, half_width, degree):
    """Fit polynomial to the strongest absolute-vertical-edge per row inside a
    strip [anchor_x - half_width, anchor_x + half_width]."""
    h, w = gray.shape
    x0 = max(0, anchor_x - half_width)
    x1 = min(w, anchor_x + half_width)
    strip = grad[:, x0:x1]
    best_rel = np.argmax(strip, axis=1)
    best_val = strip[np.arange(h), best_rel]
    keep = best_val > np.percentile(best_val, 25)
    ys = np.arange(h)[keep]
    xs = best_rel[keep] + x0
    coeffs = np.polyfit(ys, xs, degree)
    half_widths = []
    threshold = 0.5 * best_val[keep]
    for i, y in enumerate(ys):
        row = strip[y]
        mask = row > threshold[i]
        if mask.any():
            idxs = np.where(mask)[0]
            half_widths.append((idxs.max() - idxs.min()) / 2.0)
    half_widths = np.array(half_widths) if half_widths else np.array([1.0])
    centers = np.column_stack([ys, xs])
    return centers, coeffs, half_widths

def _fit_left_parallel_to_right(gray, grad, right_coeffs, expected_w_px, narrow_tol_px, degree):
    """Anchor the left wall to the right wall: at each row, search a narrow window
    around right_wall(y) - expected_w_px and take the strongest gradient. Then fit
    the residual (left_obs - (right - expected_w)) with a degree-1 polynomial and
    add it to right_coeffs to produce a left_coeffs polynomial parallel-up-to-linear
    to the right wall. Robust to occluded left walls because only 2 parameters of
    correction are fit on top of the (reliable) right-wall shape."""
    h, w = gray.shape
    ys = np.arange(h)
    x_right = np.polyval(right_coeffs, ys)
    expected_x_left = x_right - expected_w_px

    half = max(1, int(round(narrow_tol_px)))

    xs_obs = np.full(h, np.nan)
    vals = np.zeros(h, dtype=np.float32)
    half_widths = []

    for y in range(h):
        xl = int(max(0, expected_x_left[y] - half))
        xr = int(min(w, expected_x_left[y] + half + 1))
        if xr <= xl:
            continue
        row = grad[y, xl:xr]
        rel = int(np.argmax(row))
        peak_val = float(row[rel])
        xs_obs[y] = xl + rel
        vals[y] = peak_val
        thresh = 0.5 * peak_val
        mask = row > thresh
        if mask.any():
            idxs = np.where(mask)[0]
            half_widths.append((idxs.max() - idxs.min()) / 2.0)

    valid = ~np.isnan(xs_obs)
    if valid.sum() < degree + 2:
        raise RuntimeError("left-wall parallel fit: too few valid rows")

    keep = valid & (vals > np.percentile(vals[valid], 25))
    ys_keep = ys[keep]
    xs_keep = xs_obs[keep]

    residual = xs_keep - (np.polyval(right_coeffs, ys_keep) - expected_w_px)
    med = np.median(residual)
    mad = np.median(np.abs(residual - med)) + 1e-9
    inliers = np.abs(residual - med) < 3 * 1.4826 * mad
    if inliers.sum() < 3:
        inliers = np.ones_like(residual, dtype=bool)

    res_coeffs = np.polyfit(ys_keep[inliers], residual[inliers], 1)

    left_coeffs = right_coeffs.astype(np.float64).copy()
    left_coeffs[-1] -= expected_w_px
    left_coeffs[-2] += res_coeffs[0]
    left_coeffs[-1] += res_coeffs[1]

    left_centers = np.column_stack([ys_keep[inliers], xs_keep[inliers]])
    half_widths = np.array(half_widths) if half_widths else np.array([1.0])
    return left_centers, left_coeffs, half_widths

def detect_walls(r_cfg: ResearchConfig, channeld_cfg: ChannelDetectionConfig, br_frame: str):
    """Two-stage wall detection that pins both walls to their inner (channel-
    facing) edges. Stage 1: column-sum the positive ∂I/∂x and pick the strongest
    peak in a window around each expected wall position — these correspond to
    the inner edges regardless of whether the wall reads as a bright or a dark
    band, since both polarities have a positive gradient at the channel-side
    transition. The distance between the two anchors gives the *measured*
    channel width in pixels (which can differ from `expected_channel_width_um`
    once the matched-filter centering of preprocessing has decided which pair
    of edges to align). Stage 2: polynomial-fit the right wall using the strip
    around its anchor, then anchor the left wall to right_wall(y) - measured_width
    per row, fit the linear residual, and produce left_coeffs as right_coeffs
    shifted by the measured width plus that small linear correction. Parallelism
    keeps the left wall stable when it's partly occluded by cells/debris."""
    gray = _load_gray(br_frame)
    h, w = gray.shape
    sx = _signed_vertical_edges(gray)
    grad_pos = np.maximum(sx, 0.0)

    expected_w_px = r_cfg.channel_width_um / r_cfg.pixel_size_um
    axis = w / 2.0

    profile_pos = grad_pos.sum(axis=0)
    tol_px = channeld_cfg.channel_width_search_tolerance * expected_w_px
    right_target = axis + expected_w_px / 2.0
    left_target = axis - expected_w_px / 2.0
    right_anchor = _find_anchor(profile_pos, right_target, tol_px, w, br_frame)
    left_anchor = _find_anchor(profile_pos, left_target, tol_px, w, br_frame)
    measured_w_px = right_anchor - left_anchor

    hw = channeld_cfg.wall_strip_half_width
    deg = channeld_cfg.channel_wall_degree
    right_centers, right_coeffs, right_hw_arr = _fit_wall_at(gray, grad_pos, right_anchor, hw, deg)

    narrow_tol_px = channeld_cfg.wall_parallelism_search_fraction * measured_w_px
    left_centers, left_coeffs, left_hw_arr = _fit_left_parallel_to_right(
        gray, grad_pos, right_coeffs, measured_w_px, narrow_tol_px, deg
    )

    suggested_inset = int(np.percentile(np.concatenate([left_hw_arr, right_hw_arr]), 95)) + channeld_cfg.channel_wall_base_inset
    sample_ys = np.linspace(0, h - 1, 20).astype(int)
    widths = np.polyval(right_coeffs, sample_ys) - np.polyval(left_coeffs, sample_ys)
    max_inset = int(np.median(widths) * channeld_cfg.channel_wall_max_inset_fraction)
    suggested_inset = min(suggested_inset, max_inset)

    return left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset


def detect_interface(channeld_cfg: ChannelDetectionConfig, br_frame: str, left_coeffs: np.ndarray, right_coeffs: np.ndarray, inset: int):
    """Find the membrane interface as the optimal vertical seam through the inset
    strip. Per-row ridge response is the absolute second derivative of a
    gaussian-smoothed row — this fires on thin lines (the interface) but not on
    broad gradients or wall edges. A Viterbi DP top-to-bottom finds the column
    sequence that maximizes ridge response while penalizing per-row jumps; the
    smoothness prior is what makes this robust to scattered debris that the old
    per-row argmax would have grabbed. Final output is a UnivariateSpline fit to
    the DP path with one MAD-rejection refinement."""
    gray = _load_gray(br_frame)
    h, w = gray.shape

    left_in  = left_coeffs.astype(np.float64).copy();  left_in[-1]  += inset
    right_in = right_coeffs.astype(np.float64).copy(); right_in[-1] -= inset

    ys_all = np.arange(h)
    xls = np.clip(np.polyval(left_in,  ys_all).astype(int), 0, w - 1)
    xrs = np.clip(np.polyval(right_in, ys_all).astype(int), 0, w - 1)

    sigma = channeld_cfg.interface_ridge_sigma_px
    ridge = np.abs(gaussian_filter1d(gray.astype(np.float32), sigma=sigma, order=2, axis=1))

    max_jump = int(channeld_cfg.interface_dp_max_jump_px)
    lam = float(channeld_cfg.interface_dp_jump_penalty)
    INF = np.float32(1e9)

    dp = np.full((h, w), INF, dtype=np.float32)
    parent = np.full((h, w), -1, dtype=np.int32)

    xl0, xr0 = xls[0], xrs[0]
    if xr0 > xl0:
        dp[0, xl0:xr0] = -ridge[0, xl0:xr0]

    n_k = 2 * max_jump + 1
    candidates = np.empty((n_k, w), dtype=np.float32)
    x_idx = np.arange(w)

    for y in range(1, h):
        xl, xr = xls[y], xrs[y]
        if xr <= xl:
            continue
        prev_dp = dp[y - 1]

        candidates.fill(INF)
        for ki, k in enumerate(range(-max_jump, max_jump + 1)):
            penalty = lam * abs(k)
            if k > 0:
                candidates[ki, :w - k] = prev_dp[k:] + penalty
            elif k < 0:
                candidates[ki, -k:] = prev_dp[:w + k] + penalty
            else:
                candidates[ki] = prev_dp + penalty

        best_k_idx = np.argmin(candidates, axis=0)
        best_val = candidates[best_k_idx, x_idx]

        new_dp = -ridge[y] + best_val
        new_parent = x_idx + (best_k_idx - max_jump)

        dp[y, xl:xr] = new_dp[xl:xr]
        parent[y, xl:xr] = new_parent[xl:xr]

    yend = h - 1
    while yend >= 0 and xls[yend] >= xrs[yend]:
        yend -= 1
    if yend < 0:
        return np.empty((0, 2)), None

    end_slice = dp[yend].copy()
    end_slice[:xls[yend]] = INF
    end_slice[xrs[yend]:] = INF
    end_x = int(np.argmin(end_slice))
    if dp[yend, end_x] >= INF:
        return np.empty((0, 2)), None

    path = np.full(h, -1, dtype=np.int32)
    path[yend] = end_x
    for y in range(yend - 1, -1, -1):
        prev_x = parent[y + 1, path[y + 1]]
        path[y] = prev_x if prev_x >= 0 else path[y + 1]
    for y in range(yend + 1, h):
        path[y] = path[yend]

    valid = path >= 0
    points = np.column_stack([ys_all[valid].astype(np.float64), path[valid].astype(np.float64)])

    if len(points) < 10:
        return points, None

    factor = channeld_cfg.channel_interface_smoothing_factor
    s = len(points) * factor
    spline = UnivariateSpline(points[:, 0], points[:, 1], k=3, s=s)
    res = points[:, 1] - spline(points[:, 0])
    mad = np.median(np.abs(res - np.median(res))) + 1e-9
    keep = np.abs(res) < 3 * 1.4826 * mad
    if keep.sum() >= 10 and not keep.all():
        s_keep = keep.sum() * factor
        spline = UnivariateSpline(points[keep, 0], points[keep, 1], k=3, s=s_keep)

    return points, spline
