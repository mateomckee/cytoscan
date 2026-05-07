import os
import re
import shutil
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from importlib.resources import files

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from cytoscan import _logging
from cytoscan.config import ResearchConfig, PreprocessConfig

log = logging.getLogger(__name__)

"""this module handles all frame manipulation tasks such as scaffolding the experiment dir, loading frames, and performing preprocessing on frames before detection. crops frame to only include the channel, centers to the channel center to make all frames as uniform as possible"""

def _read_default_template() -> str:
    return files("cytoscan").joinpath("templates/default.yaml").read_text()

_SEP = r"(?:^|[_\-\s\.])"
_END = r"(?:[_\-\s\.]|$)"

# matches frames categories
# added specific matches for ch00, ch01, ch02 strings as requested by head researcher for their convenience
_CHANNEL_PATTERNS = {
    "br": re.compile(_SEP + r"((br|bf|bright(?:field)?)|(ch01))" + _END, re.IGNORECASE),
    "fl": re.compile(_SEP + r"((fl|fluor(?:escent)?)|(ch00))"   + _END, re.IGNORECASE),
    "mx": re.compile(_SEP + r"((mx|mix(?:ed)?|merged?)|(ch02))" + _END, re.IGNORECASE),
}

def _classify_frame(filename: str) -> Optional[str]:
    matches = [k for k, pat in _CHANNEL_PATTERNS.items() if pat.search(filename)]
    return matches[0] if len(matches) == 1 else None

"""sort loose .tif files at experiment_dir's root into input/{brightfield,
fluorescent,mixed}/ based on filename pattern (br/bf/bright, fl/fluor, mx/mix/merged)."""
def scaffold_experiment(experiment_dir: Path) -> None:
    # create experiment dir if doesnt exist
    experiment_dir.mkdir(parents=True, exist_ok=True)

    cfg_dest = experiment_dir / "config.yaml"
    
    # create config if doesnt exist
    if cfg_dest.exists():
        log.info("found config.yaml at %s, skipping", cfg_dest)
    else:
        cfg_dest.write_text(_read_default_template())
        log.info("created config.yaml at %s", cfg_dest)

    # create input dir if doesnt exist
    input_dir = experiment_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # create category dirs if they dont exist
    dirs = {
        "br": input_dir / "brightfield",
        "fl": input_dir / "fluorescent",
        "mx": input_dir / "mixed",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # experiment structure is now complete, proceed to sort any loose frames into their categories.
    # in order for it to be a valid experiment, all categories must have the same amount of frames

    files = sorted(p for p in experiment_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".tif", ".tiff"))
    if not files:
        return

    if len(files) % 3 != 0:
        log.error("%s: %d files is not a multiple of 3", experiment_dir.name, len(files))
        sys.exit(1)

    # go through all raw frames and place them into their category based on RE filename match
    grouped: dict[str, list[Path]] = {"br": [], "fl": [], "mx": []}
    for f in files:
        ch = _classify_frame(f.name)
        if ch is None:
            log.error("could not classify %s as br/fl/mx", f.name)
            sys.exit(1)
        grouped[ch].append(f)

    # check for uneven category frame counts
    if not (len(grouped["br"]) == len(grouped["fl"]) == len(grouped["mx"])):
        log.error("uneven category counts (brightfield, fluorescent, mixed): br=%d, fl=%d, mx=%d",
                  len(grouped["br"]), len(grouped["fl"]), len(grouped["mx"]))
        sys.exit(1)

    # rename frames into a more uniform and clean format
    for ch, paths in grouped.items():
        for i, f in enumerate(paths):
            new_name = f"frame{i:03d}_{ch}{f.suffix.lower()}"
            dest = dirs[ch] / new_name
            if dest.exists():
                log.error("refusing to overwrite %s", dest)
                sys.exit(1)
            f.rename(dest)
    log.info("sorted %d frames into input/{brightfield,fluorescent,mixed}/", len(files))

    log.info("experiment directory ready: %s", experiment_dir)

#reads input frame (.tif/.tiff) triples (brightfield, fluorescent, mixed) and outputs them as a dictionary
def load_frames(experiment_dir: Path) -> Dict[int, tuple[Path, Path, Path]]:
    br_dir = os.path.join(experiment_dir, "input/brightfield")
    fl_dir = os.path.join(experiment_dir, "input/fluorescent")
    mx_dir = os.path.join(experiment_dir, "input/mixed")

    for d in (br_dir, fl_dir, mx_dir):
        if not os.path.isdir(d):
            log.error("missing %s. Run 'cytoscan init %s' first", d, experiment_dir)
            sys.exit(1)

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

    if len(br_files) == 0:
        log.error("no .tif/.tiff frames found under %s/input/. "
                  "Drop your frames into %s and rerun 'cytoscan init %s' to sort them",
                  experiment_dir, experiment_dir, experiment_dir)
        sys.exit(1)

    if not (len(br_files) == len(fl_files) == len(mx_files)):
        log.error("frame count mismatch in %s/input/: %d brightfield, %d fluorescent, %d mixed "
                  "(each frame needs all three channels)",
                  experiment_dir, len(br_files), len(fl_files), len(mx_files))
        sys.exit(1)

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

    log.info("preprocessing %d frames", len(frames))
    for fi, (br, fl, mx) in _logging.progress(list(frames.items()), "preprocessing", total=len(frames)):
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

    log.info("preprocessing done — exported to %s", preprocess_dir)
    if invalid:
        log.warning("%d frame(s) excluded by preprocessing:", len(invalid))
        for fi, reason in invalid:
            log.warning("  frame%03d: %s", fi, reason)

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

