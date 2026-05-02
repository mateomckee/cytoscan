import numpy as np

from cytoscan.config import AnalysisConfig, ExperimentConfig
from cytoscan.detections import FrameDetections
from cytoscan.findings import CellFindings, FrameFindings, ExperimentFindings, InterfaceSample, Side, Category


def analyze(an_cfg: AnalysisConfig, exp_cfg: ExperimentConfig, detections: dict[int, FrameDetections]) -> ExperimentFindings:
    """Convert per-frame detections into per-cell distance/category findings.
    Skips frames flagged invalid; records their indices on the result so the
    skip is auditable. Distance is signed perpendicular distance from the cell
    centroid to the interface spline (tangent-line approximation, exact in the
    near-vertical limit and accurate to <1% for typical interface tilts)."""
    pixel_size_um = exp_cfg.pixel_size_um
    left_fluid:  Side = an_cfg.left_fluid
    right_fluid: Side = "dex" if left_fluid == "peg" else "peg"
    interface_band_um        = an_cfg.interface_band_um
    transition_band_um       = an_cfg.transition_band_um
    interface_sample_step_px = max(1, int(an_cfg.interface_sample_step_px))

    frames: dict[int, FrameFindings] = {}
    invalid_frame_indices: list[int] = []

    n_total = len(detections)
    for fi, fd in sorted(detections.items()):
        print(f"\r[cytoscan] analyzing detections: frame {fi+1}/{n_total}", end="", flush=True)

        if fd.flags is None or not fd.flags.frame_valid or fd.interface_curve is None:
            invalid_frame_indices.append(fi)
            continue

        spline = fd.interface_curve
        spline_deriv = spline.derivative() if hasattr(spline, "derivative") else None

        image_h_half_px = fd.image_h_px / 2.0

        cell_findings: list[CellFindings] = []
        n_peg = n_int_peg = n_int = n_int_dex = n_dex = 0
        for cell in fd.cells:
            cy = float(cell.centroid_y)
            cx = float(cell.centroid_x)
            fy = float(spline(cy))
            fpy = float(spline_deriv(cy)) if spline_deriv is not None else 0.0

            distance_signed_px = (cx - fy) / np.sqrt(fpy * fpy + 1.0)
            distance_signed_um = distance_signed_px * pixel_size_um
            distance_abs_um = abs(distance_signed_um)

            side: Side = left_fluid if distance_signed_um < 0 else right_fluid
            category = _categorize(distance_abs_um, side, interface_band_um, transition_band_um)

            channel_midpoint_x_px = (np.polyval(fd.left_coeffs, cy) + np.polyval(fd.right_coeffs, cy)) / 2.0
            centroid_x_um_from_channel_center = (cx - channel_midpoint_x_px) * pixel_size_um
            centroid_y_um_from_image_center   = (cy - image_h_half_px)       * pixel_size_um

            cell_findings.append(CellFindings(
                centroid_x                        = cx,
                centroid_y                        = cy,
                area                              = cell.area,
                label                             = cell.label,
                interface_x_at_y_px               = fy,
                distance_signed_um                = distance_signed_um,
                distance_abs_um                   = distance_abs_um,
                side                              = side,
                category                          = category,
                centroid_x_um_from_channel_center = centroid_x_um_from_channel_center,
                centroid_y_um_from_image_center   = centroid_y_um_from_image_center,
            ))

            if   category == "peg":     n_peg     += 1
            elif category == "int_peg": n_int_peg += 1
            elif category == "int":     n_int     += 1
            elif category == "int_dex": n_int_dex += 1
            elif category == "dex":     n_dex     += 1

        # ---- interface sampling (long format) + per-frame summary stats ----
        ys_sample_px = np.arange(0, fd.image_h_px, interface_sample_step_px, dtype=np.float64)
        xs_sample_px = spline(ys_sample_px)
        slopes_sample = (spline_deriv(ys_sample_px)
                         if spline_deriv is not None else np.zeros_like(ys_sample_px))
        midpoints_px = (np.polyval(fd.left_coeffs,  ys_sample_px)
                        + np.polyval(fd.right_coeffs, ys_sample_px)) / 2.0
        xs_um_from_center = (xs_sample_px - midpoints_px) * pixel_size_um
        ys_um_from_center = (ys_sample_px - image_h_half_px) * pixel_size_um

        interface_samples = [
            InterfaceSample(
                y_px                       = float(ys_sample_px[i]),
                x_px                       = float(xs_sample_px[i]),
                y_um_from_image_center     = float(ys_um_from_center[i]),
                x_um_from_channel_center   = float(xs_um_from_center[i]),
                slope_dx_dy                = float(slopes_sample[i]),
            )
            for i in range(len(ys_sample_px))
        ]

        interface_mean_x_um    = float(np.mean(xs_um_from_center))
        interface_std_x_um     = float(np.std(xs_um_from_center))
        interface_amplitude_um = float(xs_um_from_center.max() - xs_um_from_center.min())
        # linear fit of x_um vs y_um → slope as dimensionless dx/dy
        slope_fit, _ = np.polyfit(ys_um_from_center, xs_um_from_center, 1)
        interface_slope_dx_dy = float(slope_fit)

        frames[fi] = FrameFindings(
            frame_index             = fi,
            cells                   = cell_findings,
            n_peg                   = n_peg,
            n_int_peg               = n_int_peg,
            n_int                   = n_int,
            n_int_dex               = n_int_dex,
            n_dex                   = n_dex,
            mean_channel_width_um   = fd.flags.mean_channel_width_um,
            interface_samples       = interface_samples,
            interface_mean_x_um     = interface_mean_x_um,
            interface_std_x_um      = interface_std_x_um,
            interface_amplitude_um  = interface_amplitude_um,
            interface_slope_dx_dy   = interface_slope_dx_dy,
        )

    print(f" done. ({len(frames)}/{n_total} valid)")
    if invalid_frame_indices:
        print(f"           skipped invalid: {invalid_frame_indices}")

    return ExperimentFindings(
        frames                = frames,
        invalid_frame_indices = invalid_frame_indices,
        n_total_frames        = n_total,
    )


def _categorize(distance_abs_um: float, side: Side, interface_band_um: float, transition_band_um: float) -> Category:
    if distance_abs_um <= interface_band_um:
        return "int"
    if distance_abs_um <= transition_band_um:
        return "int_peg" if side == "peg" else "int_dex"
    return side
