import cv2
import numpy as np
from scipy.signal import medfilt

DEG = 3

def _load_right_half(path):
    img = cv2.imread(path)
    x_offset = img.shape[1] // 2
    gray = cv2.cvtColor(img[:, x_offset:], cv2.COLOR_BGR2GRAY)
    return gray, x_offset


def _uncrop_coeffs(coeffs, x_offset):
    out = coeffs.copy()
    out[-1] += x_offset
    return out


def detect_walls(br_frame: str):
    gray, x_offset = _load_right_half(br_frame)
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
        coeffs = np.polyfit(centers[:, 0], centers[:, 1], DEG)
        return centers, coeffs, half_widths

    left_centers,  left_coeffs,  left_hw  = fit(left_c)
    right_centers, right_coeffs, right_hw = fit(right_c)

    suggested_inset = int(np.percentile(np.concatenate([left_hw, right_hw]), 95)) + 5

    #shift back to full-image coordinates
    left_coeffs  = _uncrop_coeffs(left_coeffs,  x_offset)
    right_coeffs = _uncrop_coeffs(right_coeffs, x_offset)
    left_centers[:, 1]  += x_offset
    right_centers[:, 1] += x_offset

    return left_centers, left_coeffs, right_centers, right_coeffs, suggested_inset


def detect_interface(br_frame: str, left_coeffs: np.ndarray, right_coeffs: np.ndarray, polarity: str = 'bright', inset: int = 5):
    gray, x_offset = _load_right_half(br_frame)
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #walls arrive in full-image coords, shift to crop space, then inset
    left_in  = left_coeffs.copy();  left_in[-1]  -= x_offset; left_in[-1]  += inset
    right_in = right_coeffs.copy(); right_in[-1] -= x_offset; right_in[-1] -= inset

    pick = np.argmin if polarity == 'dark' else np.argmax
    points = []
    for y in range(h):
        xl = max(0, int(np.polyval(left_in,  y)))
        xr = min(w, int(np.polyval(right_in, y)))
        if xr - xl < 3:
            continue
        x_best = xl + int(pick(blurred[y, xl:xr]))
        points.append((y, x_best))
    points = np.array(points, dtype=np.float64)
    points[:, 1] = medfilt(points[:, 1], kernel_size=15)

    keep = np.ones(len(points), dtype=bool)
    for _ in range(5):
        coeffs = np.polyfit(points[keep, 0], points[keep, 1], DEG)
        res = points[:, 1] - np.polyval(coeffs, points[:, 0])
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        new_keep = np.abs(res) < 3 * 1.4826 * mad
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep

    #shift back to full-image coordinates
    coeffs = _uncrop_coeffs(coeffs, x_offset)
    points[:, 1] += x_offset

    return points, coeffs
