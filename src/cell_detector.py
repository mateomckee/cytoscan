from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

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

def visualize_detections(image_path: str, detections: list):
    img = plt.imread(image_path)
    
    plt.imshow(img)
    for d in detections:
        plt.plot(d.centroid_x, d.centroid_y, 'b+', markersize=10, markeredgewidth=1.0)
        plt.text(d.centroid_x + 5, d.centroid_y, f"({d.centroid_x:.0f}, {d.centroid_y:.0f})",
                 color='cyan', fontsize=5)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("detections_out.png", dpi=150)
    plt.show()
