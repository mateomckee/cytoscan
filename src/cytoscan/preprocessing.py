import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from cytoscan.config import ResearchConfig, PreprocessConfig

"""cleanup step before any detections run. crops frame to only include the channel, centers to the channel center to make all frames as uniform as possible"""

#reads input frame (.tif/.tiff) triples (brightfield, fluorescent, mixed) and outputs them as a dictionary
def load_frames(experiment_dir: str) -> Dict[int, tuple[Path, Path, Path]]:
    br_dir = os.path.join(experiment_dir, "raw/brightfield")
    fl_dir = os.path.join(experiment_dir, "raw/fluorescent")
    mx_dir = os.path.join(experiment_dir, "raw/mixed")

    def load_dir(path: str) -> list[Path]:
        encoded = os.fsencode(path)
        files = []
        for file in sorted(os.listdir(encoded)):
            filename = os.fsdecode(file)
            if filename.endswith(".tif") or filename.endswith(".tiff"):
                files.append(os.path.join(path, filename))
        return files

    br_files = load_dir(br_dir)
    fl_files = load_dir(fl_dir)
    mx_files = load_dir(mx_dir)

    if not (len(br_files) == len(fl_files) == len(mx_files)):
        raise RuntimeError(f"frame count mismatch: {len(br_files)} br, {len(fl_files)} fl, {len(mx_files)} mx")

    return {i: (br, fl, mx) for i, (br, fl, mx) in enumerate(zip(br_files, fl_files, mx_files))}

"""
For each frame: locate the channel center, then trim symmetrically
around it so the channel sits at the exact horizontal center of the
output. Output widths can vary between frames depending on how much
whitespace flanks the channel. Writes results to <experiment.dir>/preprocess/ and updates `frames` in place.
"""
def preprocess_frames(r_cfg: ResearchConfig, pre_cfg: PreprocessConfig, experiment_dir: Path, frames: Dict[int, tuple[Path, Path, Path]]) -> None:
    preprocess_dir = experiment_dir / "preprocess"

    if pre_cfg.clear_existing and preprocess_dir.exists():
        shutil.rmtree(preprocess_dir)

    br_out = preprocess_dir / "brightfield"
    fl_out = preprocess_dir / "fluorescent"
    mx_out = preprocess_dir / "mixed"
    for d in (br_out, fl_out, mx_out):
        d.mkdir(parents=True, exist_ok=True)

    expected_channel_width_px = r_cfg.channel_width_um / r_cfg.pixel_size_um

    invalid: list[tuple[int, str]] = []

    for fi, (br, fl, mx) in list(frames.items()):
        print(f"\r[cytoscan] preprocessing: frame {fi+1}/{len(frames)}", end="", flush=True)

        br_img = cv2.imread(str(br), cv2.IMREAD_UNCHANGED)
        if br_img is None:
            invalid.append((fi, "could_not_load_image"))
            del frames[fi]
            continue

        w = br_img.shape[1]
        center_x, reason = _find_channel_center(
            br_img, expected_channel_width_px, pre_cfg.snr_threshold
        )
        if center_x is None:
            invalid.append((fi, reason))
            del frames[fi]
            continue

        radius = min(center_x, w - center_x)
        crop_left  = center_x - radius
        crop_right = center_x + radius

        new_br = _crop_and_write(br, br_out, crop_left, crop_right)
        new_fl = _crop_and_write(fl, fl_out, crop_left, crop_right)
        new_mx = _crop_and_write(mx, mx_out, crop_left, crop_right)

        frames[fi] = (new_br, new_fl, new_mx)

    print(f" done. (exported to {preprocess_dir})")
    if invalid:
        print(f"[cytoscan] warning: {len(invalid)} frame(s) excluded by preprocessing:")
        for fi, reason in invalid:
            print(f"           frame{fi:03d}: {reason}")

"""
Find the channel center via matched filter on the column-gradient profile.
The template is a pair of impulses separated by the expected channel width,
so the convolution peaks where two wall edges sit at that spacing. Returns the center in input-image coordinates.
"""
def _find_channel_center(img: np.ndarray, expected_channel_width_px: float, snr_threshold: float) -> Tuple[Optional[int], Optional[str]]:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    grad_x  = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))
    profile = grad_x.sum(axis=0)
    profile = gaussian_filter1d(profile, sigma=5.0)

    half = int(round(expected_channel_width_px / 2))
    template = np.zeros(2 * half + 1, dtype=profile.dtype)
    template[0]  = 1.0
    template[-1] = 1.0
    score = np.convolve(profile, template, mode="valid")
    center = int(np.argmax(score)) + half

    snr = score.max() / max(float(np.median(score)), 1e-9)
    if snr < snr_threshold:
        return None, f"weak_signal (peak/median = {snr:.2f}, need >= {snr_threshold})"

    return center, None

def _crop_and_write(src_path: Path, dst_dir: Path, crop_left: int, crop_right: int) -> Path:
    img = cv2.imread(str(src_path), cv2.IMREAD_UNCHANGED)
    cropped = img[:, crop_left:crop_right]
    dst_path = dst_dir / Path(src_path).name
    cv2.imwrite(str(dst_path), cropped)
    return dst_path

