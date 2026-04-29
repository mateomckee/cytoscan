import cv2
import numpy as np
from scipy.signal import medfilt
from scipy.interpolate import UnivariateSpline

MAX_INSET_FRACTION = 0.175

WALL_DEG = 2
INTERFACE_DEG = 4

BASE_INSET = 10

INTERFACE_SMOOTHING_FACTOR = 32.0 

def _load_right_half(path):
    img = cv2.imread(path)
    x_offset = int(img.shape[1] * 0.5)
    gray = cv2.cvtColor(img[:, x_offset:], cv2.COLOR_BGR2GRAY)
    return gray, x_offset

def _uncrop_coeffs(coeffs, x_offset):
    out = coeffs.copy()
    out[-1] += x_offset
    return out

def detect_walls(br_frame: str):
    gray, x_offset = _load_right_half(br_frame)
    h = gray.shape[0]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    thresh = (mag > np.percentile(mag, 92)).astype(np.uint8) * 255
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1)))
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25)))

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    left_c, right_c = sorted(contours, key=lambda c: c[:, 0, 0].mean())

    def fit(c):
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [c], -1, 255, -1)
        ys, xs = np.where(mask > 0)
        rows = np.unique(ys)
        centers     = np.array([(y, xs[ys == y].mean()) for y in rows])
        half_widths = np.array([(xs[ys == y].max() - xs[ys == y].min()) / 2 for y in rows])
        coeffs = np.polyfit(centers[:, 0], centers[:, 1], WALL_DEG)
        return centers, coeffs, half_widths

    left_centers,  left_coeffs,  left_hw  = fit(left_c)
    right_centers, right_coeffs, right_hw = fit(right_c)

    # cap inset to a fraction of channel width
    suggested_inset = int(np.percentile(np.concatenate([left_hw, right_hw]), 95)) + BASE_INSET
    sample_ys = np.linspace(0, h - 1, 20).astype(int)
    widths = np.polyval(right_coeffs, sample_ys) - np.polyval(left_coeffs, sample_ys)
    max_inset = int(np.median(widths) * MAX_INSET_FRACTION)
    suggested_inset = min(suggested_inset, max_inset)

    # shift back to full-image coordinates
    left_coeffs  = _uncrop_coeffs(left_coeffs,  x_offset)
    right_coeffs = _uncrop_coeffs(right_coeffs, x_offset)
    left_centers[:, 1]  += x_offset
    right_centers[:, 1] += x_offset

    return left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset

def detect_interface(br_frame: str, left_coeffs: np.ndarray, right_coeffs: np.ndarray, inset: int = 5):
    gray, x_offset = _load_right_half(br_frame)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = np.abs(cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3))

    left_in  = left_coeffs.copy();  left_in[-1]  -= x_offset; left_in[-1]  += inset
    right_in = right_coeffs.copy(); right_in[-1] -= x_offset; right_in[-1] -= inset

    points = []
    for y in range(h):
        xl = max(0, int(np.polyval(left_in,  y)))
        xr = min(w, int(np.polyval(right_in, y)))
        if xr - xl < 3:
            continue
        x_best = xl + int(np.argmax(grad[y, xl:xr]))
        points.append((y, x_best))
    points = np.array(points, dtype=np.float64)
    points[:, 1] = medfilt(points[:, 1], kernel_size=31)

    # robust spline fit
    s = len(points) * INTERFACE_SMOOTHING_FACTOR
    keep = np.ones(len(points), dtype=bool)
    spline = None
    for _ in range(5):
        if keep.sum() < 10:
            break
        spline = UnivariateSpline(points[keep, 0], points[keep, 1], k=3, s=s)
        res = points[:, 1] - spline(points[:, 0])
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        new_keep = np.abs(res) < 3 * 1.4826 * mad
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    # shift points back to full-image coords
    # (the spline is in crop space — wrap it to shift output)
    points[:, 1] += x_offset
    interface_curve = _shifted_spline(spline, x_offset) if spline is not None else None

    return points, interface_curve


def _shifted_spline(spline, x_offset):
    return lambda y: spline(y) + x_offset
