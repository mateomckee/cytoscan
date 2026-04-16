from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage

@dataclass
class Detection :
    centroid_x: float
    centroid_y: float
    area: int           #pixels
    label: int
    
def detect_cells(prob_map: np.ndarray, prob_threshold: float = 0.7, min_cell_area: int = 40) -> List[Detection] :
    #simple scipy workflow

    mask = prob_map > prob_threshold

    labeled, num_features = ndimage.label(mask)

    detections = []
    for i in range(1, num_features + 1) :
        region = labeled == i
        area = np.sum(region)

        #filter out noise
        if area < min_cell_area :
            continue
        
        #create Detection and append
        cy, cx = ndimage.center_of_mass(region)
        detections.append(Detection(
            centroid_x = float(cx),
            centroid_y = float(cy),
            area = area,
            label = i,
        ))

    return detections
