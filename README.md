[![DOI](https://zenodo.org/badge/1186920701.svg)](https://doi.org/10.5281/zenodo.20034555)
[![PyPI](https://img.shields.io/pypi/v/cytoscan.svg)](https://pypi.org/project/cytoscan/)
[![Python versions](https://img.shields.io/pypi/pyversions/cytoscan.svg)](https://pypi.org/project/cytoscan/)
[![Downloads](https://static.pepy.tech/badge/cytoscan)](https://pepy.tech/project/cytoscan)
[![License](https://img.shields.io/pypi/l/cytoscan.svg)](https://github.com/mateomckee/cytoscan/blob/main/LICENSE)

# cytoscan

Cell migration analysis for microfluidic ATPS channels. Detects cells,
channel walls, and the fluid interface from brightfield + fluorescent
microscopy frames; produces per-cell physical-coordinate, distance, and
category data ready for downstream analysis in Excel / MATLAB / Python.

## Install

    pip install cytoscan

## Quickstart

    cytoscan run my_experiment            # scaffold dir + config, then run the full pipeline
    cytoscan version                      # version + dependency info

On the first invocation against a fresh directory, `run` will:

1. Create `my_experiment/` if it doesn't exist
2. Drop a default `config.yaml` at the root
3. Create `input/{brightfield,fluorescent,mixed}/` and sort any loose `.tif`
   files at the root into the correct category by filename pattern
4. Run the full pipeline if the inputs are present, otherwise exit with a
   message telling you to drop frames in and rerun

## Outputs

All results land in `my_experiment/output/`:

- **`cells.csv`** — one row per detected cell: pixel + µm coordinates, distance to interface, side (peg/dex), category (int / int_peg / int_dex / outside)
- **`frames.csv`** — one row per frame: cell counts by category, interface geometry summary, validity flags
- **`interface.csv`** — long-format interface samples for downstream curve analysis
- **`summary.txt`** — human-readable run summary
- **`visuals/*.png`** — per-frame overlays of cells, channel walls, and interface

## Logging and verbosity

Three global flags, usable in any position (before or after the subcommand):

- **`-v`, `--verbose`** — DEBUG output: per-frame diagnostics and algorithm internals
- **`-q`, `--quiet`** — WARNING-and-up only; silences INFO progress messages
- **`--log-file PATH`** — also write a full DEBUG-level, timestamped log to `PATH` (uncolored)

## Configuration

Every run is driven by `<experiment>/config.yaml`:

Required:
- **`research`** — `pixel_size_um`, `channel_width_um`, ... (defines the physical experiment)

Optional (defaults set by developer):
- **`cell_detection`** — `threshold` (fluorescent cell detection)
- **`channel_detection`** — wall and interface detection parameters
- **`flagging`** — quality-gate thresholds (anchor strength, signal ratio, residual MAD)
- **`analysis`** — `interface_band_um`, `transition_band_um` (categorization bands)
- **`export_visuals`**, **`export_data`** — output toggles

## Citing

If cytoscan helps your research, please cite the archived release:

> McKee, M. (2026). *cytoscan: Cell migration analysis for microfluidic ATPS channels*.
> Zenodo. https://doi.org/10.5281/zenodo.20034555

A `CITATION.cff` is included for tools like Zotero and Mendeley.

## Acknowledgements

Built for the [Sun Lab](https://www.sunlabutsa.org/) at UTSA by [Mateo McKee](https://github.com/mateomckee).
