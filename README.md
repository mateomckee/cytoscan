[![DOI](https://zenodo.org/badge/1186920701.svg)](https://doi.org/10.5281/zenodo.20034555)
[![PyPI](https://img.shields.io/pypi/v/cytoscan.svg)](https://pypi.org/project/cytoscan/)
[![Python versions](https://img.shields.io/pypi/pyversions/cytoscan.svg)](https://pypi.org/project/cytoscan/)
[![Downloads](https://static.pepy.tech/badge/cytoscan)](https://pepy.tech/project/cytoscan)
[![License](https://img.shields.io/pypi/l/cytoscan.svg)](https://github.com/mateomckee/cytoscan/blob/main/LICENSE)

# cytoscan

Cell migration analysis for microfluidic ATPS channels. Detects cells,
channel walls, and the membrane interface from brightfield + fluorescent
microscopy frames; produces per-cell distance and category data.

## Install

    pip install cytoscan

## Usage

    cytoscan init my_experiment           # scaffold dir + config
    # drop tifs (br/fl/mx triples) into my_experiment/
    cytoscan run my_experiment            # full pipeline
    cytoscan validate my_experiment       # detection-only, with flag table
    cytoscan version

Outputs land in `my_experiment/output/`:
- `cells.csv`, `frames.csv`, `interface.csv`, `summary.txt`
- `output_frame*.png` (visual overlays)

Built for the Sun Lab — https://www.sunlabutsa.org/
by Mateo McKee
