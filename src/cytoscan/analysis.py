import numpy as np

def count_cells_per_region(cells: list, interface_coeffs: np.ndarray) -> dict:
    counts = {"left": 0, "right": 0}
    for cell in cells:
        interface_x = np.polyval(interface_coeffs, cell.centroid_y)
        if cell.centroid_x < interface_x:
            counts["left"] += 1
        else:
            counts["right"] += 1
    return counts
