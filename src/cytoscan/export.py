import csv
import shutil
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from cytoscan import _logging
from cytoscan.config import ExportVisualsConfig, ExportDataConfig
from cytoscan.detections import FrameDetections
from cytoscan.findings import ExperimentFindings

log = logging.getLogger(__name__)

"""module that handles all forms of program output; detection visualization, CSV formatting and writing, etc"""

#return `path` if free, otherwise append _1, _2, ... until a free name is found
def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while (candidate := parent / f"{stem}_{i}{suffix}").exists():
        i += 1
    return candidate

def _as_callable(coeffs_or_curve):
    if coeffs_or_curve is None:
        return None
    if callable(coeffs_or_curve):
        return coeffs_or_curve
    return lambda y: np.polyval(coeffs_or_curve, y)

def _export_frame(ev_cfg: ExportVisualsConfig, output_dir: Path, fd: FrameDetections) -> None:
    import cv2
    exported_frame = {
        "brightfield": fd.br,
        "fluorescent": fd.fl,
        "mixed":       fd.mx,
    }[ev_cfg.exported_frame]

    img_bgr = cv2.imread(str(exported_frame), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if img_bgr.ndim == 3 else img_bgr
    h, w = img.shape[:2]
    ys = np.arange(h)

    dpi = 150
    fig, ax = plt.subplots(figsize=(w / dpi, h / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.imshow(img)

    # overlay sizing scales with image dimensions
    s = max(0.5, min(2.0, h / 800.0))
    line_w   = 1.0 * s
    badge_fs = max(4.0, min(7.0, h / 110.0))
    cell_fs  = max(3.0, 4.0 * s)
    cell_mk  = 7.0 * s
    pad_px   = max(4, int(6 * s))

    left_in  = fd.left_coeffs.copy();  left_in[-1]  += fd.wall_inset
    right_in = fd.right_coeffs.copy(); right_in[-1] -= fd.wall_inset

    candidates = [
        (fd.left_coeffs,      'lime',   ev_cfg.channel_walls),
        (fd.right_coeffs,     'red',    ev_cfg.channel_walls),
        (left_in,             'yellow', ev_cfg.channel_walls_inset),
        (right_in,            'yellow', ev_cfg.channel_walls_inset),
        (fd.interface_curve,  'cyan',   ev_cfg.channel_interface),
    ]
    for curve_or_coeffs, color, on in candidates:
        f = _as_callable(curve_or_coeffs)
        if not on or f is None:
            continue
        ax.plot(f(ys), ys, color=color, linewidth=line_w)

    if ev_cfg.origin:
        origin_y = h / 2.0
        origin_x = (np.polyval(fd.left_coeffs, origin_y) + np.polyval(fd.right_coeffs, origin_y)) / 2.0
        ax.plot(origin_x, origin_y, 'o', color='magenta',
                markersize=10 * s, markeredgewidth=1.5 * s, markerfacecolor='none')
        ax.plot(origin_x, origin_y, '.', color='magenta', markersize=3 * s)

    if ev_cfg.cells:
        for d in fd.cells:
            ax.plot(d.centroid_x, d.centroid_y, 'b+',
                    markersize=cell_mk, markeredgewidth=1.0 * s)
            ax.text(d.centroid_x + 5, d.centroid_y,
                    f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                    color='cyan', fontsize=cell_fs)

    if fd.flags is not None:
        f = fd.flags
        lines = [
            "VALID" if f.frame_valid else "INVALID",
            f"wall:      {'good' if f.walls_valid else 'bad'}",
            f"interface: {'good' if f.interface_valid else 'bad'}",
            f"width:     {f.mean_channel_width_um:.0f} um",
        ]
        bg = '#7a0000' if not f.frame_valid else 'black'
        ax.text(pad_px, pad_px, "\n".join(lines),
                color='white', fontsize=badge_fs, fontfamily='monospace', fontweight='bold',
                verticalalignment='top',
                bbox=dict(facecolor=bg, alpha=0.85, pad=4,
                          edgecolor='red' if not f.frame_valid else 'none', linewidth=2 * s))
        if not f.frame_valid:
            # red frame border so invalid frames are unmistakable in the gallery
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('red')
                spine.set_linewidth(max(2.0, 3.0 * s))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')

    basename = Path(exported_frame).stem
    out_path = output_dir / f"output_{basename}.png"
    if not ev_cfg.overwrite_existing:
        out_path = _unique_path(out_path)
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def export_visuals(ev_cfg: ExportVisualsConfig, experiment_dir: Path, detections: Dict[int, FrameDetections]) -> None:
    if not ev_cfg.enabled: return

    output_dir = experiment_dir / "output/visuals"

    if ev_cfg.clear_existing and output_dir.exists(): shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("exporting visuals for %d frames", len(detections))
    with plt.style.context('dark_background'):
        for fi, fd in _logging.progress(detections.items(), "exporting visuals", total=len(detections)):
            _export_frame(ev_cfg, output_dir, fd)
    log.info("visuals exported to %s", output_dir)

def export_data(ed_cfg: ExportDataConfig, experiment_dir: Path, findings: ExperimentFindings) -> None:
    if not ed_cfg.enabled:
        return

    output_dir = experiment_dir / "output/data"
    output_dir.mkdir(parents=True, exist_ok=True)

    cells_path     = output_dir / "cells.csv"
    frames_path    = output_dir / "frames.csv"
    interface_path = output_dir / "interface.csv"
    summary_path   = output_dir / "summary.txt"

    with open(cells_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index", "label",
            "centroid_x_px", "centroid_y_px", "area_px2",
            "centroid_x_um_from_origin", "centroid_y_um_from_origin",
            "interface_x_at_y_px",
            "distance_signed_um", "distance_abs_um",
            "side", "category",
        ])
        for fi, ff in sorted(findings.frames.items()):
            for c in ff.cells:
                w.writerow([
                    fi, c.label,
                    f"{c.centroid_x:.2f}", f"{c.centroid_y:.2f}", c.area,
                    f"{c.centroid_x_um_from_origin:.2f}",
                    f"{c.centroid_y_um_from_origin:.2f}",
                    f"{c.interface_x_at_y_px:.2f}",
                    f"{c.distance_signed_um:.2f}", f"{c.distance_abs_um:.2f}",
                    c.side, c.category,
                ])

    with open(frames_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index", "mean_channel_width_um",
            "n_total_cells",
            "n_peg", "n_int_peg", "n_int", "n_int_dex", "n_dex",
            "n_peg_pct", "n_int_peg_pct", "n_int_pct", "n_int_dex_pct", "n_dex_pct",
            "interface_mean_x_um", "interface_std_x_um",
            "interface_amplitude_um", "interface_slope_dx_dy",
        ])
        for fi, ff in sorted(findings.frames.items()):
            n_total = ff.n_peg + ff.n_int_peg + ff.n_int + ff.n_int_dex + ff.n_dex
            n_peg_pct, n_int_peg_pct, n_int_pct, n_int_dex_pct, n_dex_pct = 0.0, 0.0, 0.0, 0.0, 0.0
            if n_total != 0.0 :
                n_peg_pct, n_int_peg_pct, n_int_pct, n_int_dex_pct, n_dex_pct = (100 * x / n_total for x in (ff.n_peg, ff.n_int_peg, ff.n_int, ff.n_int_dex, ff.n_dex))
            
            w.writerow([
                fi, f"{ff.mean_channel_width_um:.1f}",
                n_total,
                ff.n_peg, ff.n_int_peg, ff.n_int, ff.n_int_dex, ff.n_dex,
                f"{n_peg_pct:.2f}", f"{n_int_peg_pct:.2f}", f"{n_int_pct:.2f}", f"{n_int_dex_pct:.2f}", f"{n_dex_pct:.2f}",
                f"{ff.interface_mean_x_um:.2f}", f"{ff.interface_std_x_um:.2f}",
                f"{ff.interface_amplitude_um:.2f}", f"{ff.interface_slope_dx_dy:.5f}",
            ])

    with open(interface_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_index",
            "y_px", "x_px",
            "y_um_from_origin", "x_um_from_origin",
            "slope_dx_dy",
        ])
        for fi, ff in sorted(findings.frames.items()):
            for s in ff.interface_samples:
                w.writerow([
                    fi,
                    f"{s.y_px:.2f}", f"{s.x_px:.2f}",
                    f"{s.y_um_from_origin:.2f}",
                    f"{s.x_um_from_origin:.2f}",
                    f"{s.slope_dx_dy:.5f}",
                ])

    with open(summary_path, "w") as f:
        f.write(f"frames_total:   {findings.n_total_frames}\n")
        f.write(f"frames_valid:   {len(findings.frames)}\n")
        f.write(f"frames_invalid: {len(findings.invalid_frame_reasons)}\n")
        if findings.invalid_frame_reasons:
            f.write("invalid_frames:\n")
            for fi in sorted(findings.invalid_frame_reasons.keys()):
                f.write(f"  frame{fi:03d}: {findings.invalid_frame_reasons[fi]}\n")

    log.info("exported %s, %s, %s, %s to %s",
             cells_path.name, frames_path.name, interface_path.name, summary_path.name, output_dir)

def export_all(ev_cfg: ExportVisualsConfig, ed_cfg: ExportDataConfig, experiment_dir: Path, detections: Dict[int, FrameDetections], findings: ExperimentFindings) -> None :
    if ev_cfg.enabled :
        export_visuals(ev_cfg, experiment_dir, detections)
    if ed_cfg.enabled :
        export_data(ed_cfg, experiment_dir, findings)

