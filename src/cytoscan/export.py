import shutil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from cytoscan.config import OutputConfig, ExportVisualsConfig, ExportDataConfig
from cytoscan.detections import FrameDetections
from cytoscan.findings import ExperimentFindings

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
    exported_frame = {
        "brightfield": fd.br,
        "fluorescent": fd.fl,
        "mixed":       fd.mx,
    }[ev_cfg.exported_frame]

    img = plt.imread(exported_frame)
    h, w = img.shape[:2]
    ys = np.arange(h)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img)

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
        ax.plot(f(ys), ys, color=color, linewidth=1)

    if ev_cfg.cells:
        for d in fd.cells:
            ax.plot(d.centroid_x, d.centroid_y, 'b+',
                    markersize=10, markeredgewidth=1.0)
            ax.text(d.centroid_x + 5, d.centroid_y,
                    f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                    color='cyan', fontsize=5)

    # validity badge
    if fd.flags is not None:
        f = fd.flags
        lines = [
            "VALID" if f.frame_valid else "INVALID",
            f"wall:      {'✓' if f.walls_valid else '✗'}",
            f"interface: {'✓' if f.interface_valid else '✗'}",
            f"width:     {'✓' if f.channel_width_valid else '✗'}",
        ]
        ax.text(10, 30, "\n".join(lines),
                color='white', fontsize=10, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.6, pad=4, edgecolor='none'))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis('off')
    plt.tight_layout()

    basename = Path(exported_frame).stem
    out_path = output_dir / f"output_{basename}.png"
    if not ev_cfg.overwrite_existing:
        out_path = _unique_path(out_path)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

""" public methods """
def export_visuals(ev_cfg: ExportVisualsConfig, experiment_dir: Path, detections: ExperimentFindings) -> None:
    if not ev_cfg.enabled: return

    output_dir = experiment_dir / "Output"

    #clear existing?
    if ev_cfg.clear_existing and output_dir.exists(): shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    for fi, fd in detections.items():
        print(f"\r[cytoscan] exporting visuals: frame {fi+1}/{len(detections)}", end="", flush=True)
        _export_frame(ev_cfg, output_dir, fd)
    print(f" done. (exported to {output_dir})")

def export_data(ed_cfg: ExportDataConfig, experiment_dir: Path, findings: ExperimentFindings) -> None :
    print("[cytoscan] exporting analysis data to csv files")
    return

def export_all(output_cfg: OutputConfig, experiment_dir: Path, detections: FrameDetections, findings: ExperimentFindings) -> None :
    if output_cfg.export_visuals.enabled :
        export_visuals(output_cfg.export_visuals, experiment_dir, detections)
    if output_cfg.export_data.enabled :
        export_data(output_cfg.export_data, experiment_dir, findings)

