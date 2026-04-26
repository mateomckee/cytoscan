import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_walls(br_frame: str):
    img = cv2.imread(br_frame)
    h, w = img.shape[:2]
    x_offset = int(w * 0.6)
    img = img[:, x_offset:]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 30, 100)

    y_pts, x_pts = np.where(edges > 0)
    mid_x = edges.shape[1] // 2

    def fit_wall(mask):
        wy = y_pts[mask]
        wx = x_pts[mask]
        rows = np.unique(wy)
        median_x = [np.median(wx[wy == r]) for r in rows]
        return np.polyfit(rows, median_x, deg=2)

    left_coeffs = fit_wall(x_pts < mid_x)
    right_coeffs = fit_wall(x_pts >= mid_x)

    left_coeffs[-1] += x_offset
    right_coeffs[-1] += x_offset

    return left_coeffs, right_coeffs

def detect_interface(br_frame: str, left_coeffs, right_coeffs):
    img = cv2.imread(br_frame)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    edges = cv2.Canny(enhanced, 20, 60)

    y_pts, x_pts = np.where(edges > 0)

    inter_wall_mask = np.array([
        (int(np.polyval(left_coeffs, y)) + 5 < x < int(np.polyval(right_coeffs, y)) - 5)
        and abs(x - (np.polyval(left_coeffs, y) + np.polyval(right_coeffs, y)) / 2) < 30
        for x, y in zip(x_pts, y_pts)
    ])

    wy = y_pts[inter_wall_mask]
    wx = x_pts[inter_wall_mask]

    if len(wy) < 10:
        raise RuntimeError("not enough edge points in inter-wall region")

    rows = np.unique(wy)
    median_x = [np.median(wx[wy == r]) for r in rows]
    interface_coeffs = np.polyfit(rows, median_x, deg=2)

    return interface_coeffs

def visualize_curves(br_frame: str, curves: list):
    img = cv2.imread(br_frame)
    h, w = img.shape[:2]

    for coeffs, color in curves:
        for y in range(h):
            x = int(np.polyval(coeffs, y))
            if 0 <= x < w:
                cv2.circle(img, (x, y), 1, color, -1)

    cv2.imwrite("curves_out.png", img)

