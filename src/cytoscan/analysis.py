import numpy as np
from pathlib import Path

from cytoscan.config import ExportDataConfig 
from cytoscan.detections import FrameDetections
from cytoscan.findings import *

def analyze(ed_cfg: ExportDataConfig, experiment_dir: Path, detections: FrameDetections) -> ExperimentFindings:
    for fi, fd in detections.items():
        print(f"\r[cytoscan] analyzing detections: frame {fi+1}/{len(detections)}", end="", flush=True)
        
        cell_findigns: CellFindings
        frame_findings: FrameFindings
        experiment_findings: ExperimentFindings

        #TODO
        #populate
        # 1. CellFindings - per cell fata

        

        # 2. FrameFindings - per frame data


    
        # 3. ExperimentFindings -per experiment data (everything together)


        


    print(f" done.")
    return None

