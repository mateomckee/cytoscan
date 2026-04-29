import shutil
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from cytoscan.config import ExportVisualsConfig
from cytoscan.detections import FrameDetections

#return `path` if free, otherwise append _1, _2, ... until a free name is found
def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while (candidate := parent / f"{stem}_{i}{suffix}").exists():
        i += 1
    return candidate

def export_visuals(ev_cfg: ExportVisualsConfig, experiment_dir: Path, detections: Dict[int, FrameDetections]) -> None:
    if not ev_cfg.enabled: return

    #clear existing?
    output_dir = experiment_dir / "Output"
    if ev_cfg.clear_existing and output_dir.exists(): shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    for fi, fd in detections.items():
        _export_frame(ev_cfg, output_dir, fd)

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
        (fd.interface_coeffs, 'cyan',   ev_cfg.channel_interface),
    ]
    for coeffs, color, on in candidates:
        if not on or coeffs is None:
            continue
        ax.plot(np.polyval(coeffs, ys), ys, color=color, linewidth=1)

    if ev_cfg.cells:
        for d in fd.cells:
            ax.plot(d.centroid_x, d.centroid_y, 'b+',
                    markersize=10, markeredgewidth=1.0)
            ax.text(d.centroid_x + 5, d.centroid_y,
                    f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                    color='cyan', fontsize=5)

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

