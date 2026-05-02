# CLAUDE.md

Project-specific context for Claude Code working on cytoscan.

## What this project is

Cell migration analysis tool for microfluidic channels. See `README.md` for
the high-level pipeline. This file covers conventions, gotchas, and "load-bearing"
decisions that aren't obvious from the code.

## Architecture rules

- **Data shapes live in `detections.py`.** This is the leaf module — it imports
  nothing project-internal. `FrameDetections`, `FrameFlags`, `CellDetection`
  are dataclasses with no behavior. Computation lives elsewhere.
- **Module dependencies flow downward.** `detections` ← `config` ← (`channel_detector`,
  `cell_detector`, `flagging`, `analysis`, `export`) ← `cli`. If a leaf module
  needs to import from a higher one, something is in the wrong place.
- **Stages over loops.** Pipeline is `detect → flag → analyze → export`. Each
  stage takes the full set of frames, does one pass, and produces enriched data
  for the next stage. Don't fold stages together to "save a loop" — frame counts
  are <30, loop overhead is irrelevant, and stage separation is the architecture.

## Coordinate conventions (load-bearing)

- All polynomials and spline curves stored on `FrameDetections` are in
  **full-image coordinates**, not crop coordinates.
- Detection internally crops to the right half of the image (`x_offset = w // 2`),
  computes there, then shifts results back before returning. The shift adds
  `x_offset` to the constant term of polynomial coefficients (`coeffs[-1] += x_offset`)
  and wraps splines in a lambda that adds `x_offset` to their output.
- When `flagging.py` needs to operate in crop space (e.g., to use the precomputed
  gradient image), it does the inverse shift locally — does NOT mutate the
  full-image coeffs on `FrameDetections`.
- `pixel_size_um` (in `cfg.experiment`) converts pixels to micrometers. All
  user-facing measurements (channel width, distances, etc.) should be in µm.

## Walls vs interface — different model choices

- **Walls** are polynomials (`np.polyfit`, degree 3). Walls are nearly straight,
  polynomial is the right shape. Don't switch to splines — splines give noise
  more freedom to wiggle, which makes weak-wall frames worse.
- **Interface** is a `UnivariateSpline`. Real membranes can have multiple bends
  that polynomials physically cannot fit. Smoothing parameter `s` controls
  flexibility — see `INTERFACE_SMOOTHING_FACTOR` in `channel_detector.py`.
- **Both are callable.** `np.polyval(coeffs, y)` for walls, `spline(y)` for
  interface. `_as_callable` in `export.py` wraps polynomials so consumer code
  can treat them uniformly.

## Flagging philosophy

- Flags catch frames where detection succeeded technically but produced
  unreliable output. Three independent metrics for the interface (signal ratio,
  residual MAD, curve amplitude) catch three different failure modes:
  - Signal ratio low → no real interface present (membrane absent)
  - Residual high → picks scattered, fit can't track them
  - Amplitude high → spline making big swings (S-curves through nothing)
- Wall flag uses **top+bottom span check**, not coverage fraction. A patchy
  wall with gaps but real data near both edges is fine — the polynomial
  interpolates safely. A wall with data only in the top half is dangerous —
  the polynomial extrapolates wildly.
- Channel width vs expected value catches "wall detection grabbed the wrong
  feature" cases. Researchers report 600µm channels.
- **Thresholds live in `FlaggingConfig`** and get copied into `FrameFlags`.
  Frames carry their own thresholds so derived booleans (`frame_valid`) work
  without config access in downstream code.

## Config

- Pydantic models in `config.py`. Required fields have no defaults; defaults
  are reserved for fields with sensible fallbacks.
- Nested groups for related settings (`detection`, `flagging`, `output`,
  `experiment`).
- `cfg.experiment` is a structured group, not a path. Path is
  `cfg.experiment.dir`. (This was refactored once — don't reintroduce
  `cfg.experiment` as a Path.)
- Pass config slices to stages, not the full config: `compute_flags_all(cfg.flagging,
  cfg.experiment, detections)`. Two-config signatures are fine; a single
  giant-config signature hides what the function actually depends on.

## Code style

- Stdlib > third-party where comparable. We use `pathlib`, `argparse`, `csv`
  (when CSV export is implemented). Pydantic is justified for config validation;
  pandas is acceptable but not required for CSV writing.
- Prefer paraphrasing-quality variable names over comments. If a variable
  needs a comment to explain it, rename it.
- Vectorize numpy where natural (`np.polyval(coeffs, ys)` over comprehension)
  but don't contort code to avoid loops on small N.
- `print` for user-facing pipeline output (progress, flag tables). No logging
  framework yet — add when there's a use case.

## Things not to do

- **Don't import `cv2` or `numpy` inside functions.** Always at module top.
- **Don't add `from .module import X` (relative imports).** Absolute imports
  only: `from cytoscan.module import X`. Survives refactors better.
- **Don't auto-fix detection by trying smarter algorithms when a frame is bad.**
  The flagging system exists to identify bad frames; don't try to rescue them
  in detection. Bad frames are excluded downstream, not patched.
- **Don't add file I/O to `analysis.py`.** Analysis is pure computation —
  takes detections, returns findings. Export does I/O.
- **Don't pickle/cache detection results yet.** Re-running on 30 frames is
  fast; caching adds invalidation problems we don't need.

## Current status / in progress

- Detection: working, tuned on a calibration set.
- Flagging: working with researcher-confirmed channel width (600µm).
- Analysis (`analysis.py`): in progress. Will produce per-cell distances,
  categories (dex/int_dex/int/int_peg/peg), per-frame aggregates.
- Export (`export.py`): visuals working; CSV export in progress (`cells.csv`,
  `frames.csv`).
- Cell tracking: deferred. Researchers will switch to higher-frame-rate
  capture before we add it. Current approach is population-based per frame.

## Running

```
pip install -e .
cytoscan -c configs/example.yaml
```

Run from project root. Always.
