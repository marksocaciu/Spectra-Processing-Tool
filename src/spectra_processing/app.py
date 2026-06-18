from operator import lt

import matplotlib
try:
    matplotlib.use("TkAgg")  # preferred for the Tkinter GUI
except ImportError:
    matplotlib.use("Agg")  # allows syntax/import checks in headless environments
from tkinter import  ttk, filedialog, messagebox, colorchooser
import pandas as pd
import re
import csv
import os
import shutil
import matplotlib.pyplot as plt
import tkinter as tk
from scipy.signal import find_peaks
from sklearn import preprocessing
import numpy as np
from collections import defaultdict
try:
    import pywt
except ImportError:
    pywt = None
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy import sparse
from scipy.sparse.linalg import spsolve


class Measurement:
    def __init__(self, name, value, std, wave, alias):
        self.name = name
        self.value = value
        self.std = std
        self.wave = wave
        self.alias = alias
        self.afw=[]
        self.af=[]
        self.af_std=[]
        self.visualize=[]
    
    def find_max(self):
        try:
            max_index = np.argmax(self.value)
        except ValueError:
            return 0,0,0
        max_index = np.argmax(self.value)
        return self.value[max_index], self.wave[max_index], self.std[max_index]

class Fluorophor:
    def __init__(self,name,auto):
        self.name = name
        self.auto = auto

def extract_nume_label(filename):
    """Extracts the NUME_LABEL from a filename."""
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None

def is_control(filename):
    match = re.search(r"\_\_\[\[(.*?)\]\]\_", filename)
    if match:
        return match.group(1)
    else:
        return None

# Apply an optimal neighborhood filter (Savitzky-Golay filter for smoothing)
def savgol(data, window_size, poly_order):
    return savgol_filter(data, window_size, poly_order)


def _canonical_solution_name(solution: str | None) -> str:
    """Return a normalized spectrometer/solution name for robust comparisons."""
    return (solution or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_sers_solution(solution: str | None) -> bool:
    """True for all SERS modes, including custom/capitalization variants."""
    return "sers" in _canonical_solution_name(solution)


def is_sers_renishaw_solution(solution: str | None) -> bool:
    name = _canonical_solution_name(solution)
    return "sers" in name and "renishaw" in name


def is_sers_bwtek_solution(solution: str | None) -> bool:
    name = _canonical_solution_name(solution)
    return "sers" in name and "bwtek" in name


def is_sers_avantes_solution(solution: str | None) -> bool:
    name = _canonical_solution_name(solution)
    return "sers" in name and "avantes" in name


def _normalise_method_name(method: str | None) -> str:
    return (method or "None").strip().lower().replace("_", " ").replace("-", " ")

def get_denoise_params(level: str | None) -> tuple[int, int] | None:
    """
    Map UI denoise level to Savitzky-Golay parameters.

    Lower window -> less smoothing and better peak preservation.
    Extreme is intentionally aggressive and can broaden or attenuate narrow
    SERS peaks; use it only for very noisy display plots.
    """
    mapping = {
        "off": None,
        "low": (5, 2),
        "medium": (7, 3),
        "high": (9, 3),
        "very high": (15, 3),
        "very_high": (15, 3),
        "veryhigh": (15, 3),
        "extreme": (35, 3),
    }
    key = (level or "Medium").strip().lower()
    return mapping.get(key, (7, 3))

def get_flatten_params(level: str | None) -> dict:
    """
    Map UI flatten level to baseline and literature-mode parameters.

    IASLS/arPLS/SNIP values are tuned for display-quality SERS plotting.
    Higher levels remove stronger background but can also attenuate broad bands.
    """
    mapping = {
        "low": {
            "ma_window": 3,
            "iasls_lam": 1e6,
            "iasls_p": 0.01,
            "arpls_lam": 1e5,
            "snip_iterations": 35,
            "despike_window": 5,
            "despike_threshold": 8.0,
            "lit_savgol_window": 9,
            "lit_savgol_poly": 3,
        },
        "medium": {
            "ma_window": 5,
            "iasls_lam": 3e5,
            "iasls_p": 0.02,
            "arpls_lam": 3e5,
            "snip_iterations": 50,
            "despike_window": 7,
            "despike_threshold": 7.0,
            "lit_savgol_window": 11,
            "lit_savgol_poly": 3,
        },
        "high": {
            "ma_window": 7,
            "iasls_lam": 1e5,
            "iasls_p": 0.03,
            "arpls_lam": 1e6,
            "snip_iterations": 70,
            "despike_window": 7,
            "despike_threshold": 6.0,
            "lit_savgol_window": 15,
            "lit_savgol_poly": 3,
        },
        "very high": {
            "ma_window": 9,
            "iasls_lam": 5e4,
            "iasls_p": 0.05,
            "arpls_lam": 3e6,
            "snip_iterations": 90,
            "despike_window": 9,
            "despike_threshold": 5.5,
            "lit_savgol_window": 21,
            "lit_savgol_poly": 3,
        },
        "very_high": {
            "ma_window": 9,
            "iasls_lam": 5e4,
            "iasls_p": 0.05,
            "arpls_lam": 3e6,
            "snip_iterations": 90,
            "despike_window": 9,
            "despike_threshold": 5.5,
            "lit_savgol_window": 21,
            "lit_savgol_poly": 3,
        },
        "veryhigh": {
            "ma_window": 9,
            "iasls_lam": 5e4,
            "iasls_p": 0.05,
            "arpls_lam": 3e6,
            "snip_iterations": 90,
            "despike_window": 9,
            "despike_threshold": 5.5,
            "lit_savgol_window": 21,
            "lit_savgol_poly": 3,
        },
        "extreme": {
            "ma_window": 13,
            "iasls_lam": 1e4,
            "iasls_p": 0.07,
            "arpls_lam": 1e7,
            "snip_iterations": 120,
            "despike_window": 11,
            "despike_threshold": 5.0,
            "lit_savgol_window": 35,
            "lit_savgol_poly": 3,
        },
    }
    key = (level or "Medium").strip().lower()
    return mapping.get(key, mapping["medium"])

def linear_baseline(y_row: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Straight-line baseline through the first and last finite spectral points.
    """
    y_row = np.asarray(y_row, dtype=float)
    x = np.asarray(x, dtype=float)

    valid = np.isfinite(y_row) & np.isfinite(x)
    if np.sum(valid) < 2:
        return np.zeros_like(y_row, dtype=float)

    xv = x[valid]
    yv = y_row[valid]
    x0, x1 = xv[0], xv[-1]
    y0, y1 = yv[0], yv[-1]

    if x1 == x0:
        return np.full_like(y_row, y0, dtype=float)

    slope = (y1 - y0) / (x1 - x0)
    return y0 + slope * (x - x0)


def iasls_baseline(
    y: np.ndarray,
    lam: float = 1e6,
    p: float = 0.01,
    diff_order: int = 2,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Asymmetric least-squares baseline approximation.
    """
    y = np.asarray(y, dtype=float)

    if not np.any(np.isfinite(y)):
        return np.full_like(y, np.nan)

    idx = np.arange(len(y))
    valid = np.isfinite(y)
    if np.sum(valid) < 2:
        return np.zeros_like(y, dtype=float)

    y_filled = y.copy()
    y_filled[~valid] = np.interp(idx[~valid], idx[valid], y[valid])

    n = len(y_filled)
    E = sparse.eye(n, format="csc")
    D = E[1:] - E[:-1]
    for _ in range(diff_order - 1):
        D = D[1:] - D[:-1]

    DTD = (D.T @ D).tocsc()
    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)

    for _ in range(max_iter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z_new = spsolve(W + lam * DTD, w * y_filled)

        diff = y_filled - z_new
        w_new = np.where(diff > 0, p, 1 - p)

        if np.linalg.norm(z_new - z) / (np.linalg.norm(z_new) + 1e-12) < tol:
            z = z_new
            break

        z = z_new
        w = w_new

    return z



def _fill_nonfinite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        return values.copy()
    idx = np.arange(values.size)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=float)
    if np.sum(finite) == 1:
        return np.full_like(values, float(values[finite][0]), dtype=float)
    out = values.copy()
    out[~finite] = np.interp(idx[~finite], idx[finite], values[finite])
    return out


def _safe_savgol_1d(values: np.ndarray, window_size: int, poly_order: int) -> np.ndarray:
    """Apply Savitzky-Golay safely to one spectrum."""
    values = _fill_nonfinite(np.asarray(values, dtype=float))
    if values.size < 3:
        return values

    max_valid_window = values.size
    if max_valid_window % 2 == 0:
        max_valid_window -= 1

    current_window = min(int(window_size), max_valid_window)
    if current_window % 2 == 0:
        current_window -= 1

    min_required_window = int(poly_order) + 2
    if min_required_window % 2 == 0:
        min_required_window += 1

    if current_window >= min_required_window and current_window >= 3:
        return savgol(values, current_window, int(poly_order))
    return values


def _interp_to_axis(source_x: np.ndarray, source_y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    """Interpolate a spectrum/background trace onto another x-axis safely."""
    source_x = np.asarray(source_x, dtype=float)
    source_y = np.asarray(source_y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)

    valid = np.isfinite(source_x) & np.isfinite(source_y)
    if np.sum(valid) == 0:
        return np.zeros_like(target_x, dtype=float)
    if np.sum(valid) == 1:
        return np.full_like(target_x, float(source_y[valid][0]), dtype=float)

    sx = source_x[valid]
    sy = source_y[valid]
    order = np.argsort(sx)
    sx = sx[order]
    sy = sy[order]

    unique_x, inverse = np.unique(sx, return_inverse=True)
    if unique_x.size != sx.size:
        summed = np.zeros_like(unique_x, dtype=float)
        counts = np.zeros_like(unique_x, dtype=float)
        np.add.at(summed, inverse, sy)
        np.add.at(counts, inverse, 1.0)
        sy = summed / np.maximum(counts, 1.0)
        sx = unique_x

    return np.interp(target_x, sx, sy, left=sy[0], right=sy[-1])


def robust_mad_sigma(values: np.ndarray) -> float:
    """Robust Gaussian-equivalent sigma from the median absolute deviation."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    med = np.median(finite)
    mad = np.median(np.abs(finite - med))
    return float(1.4826 * mad)


def despike_spectrum(
    y: np.ndarray,
    *,
    window: int = 7,
    threshold: float = 7.0,
    passes: int = 2,
) -> np.ndarray:
    """
    Remove isolated 1-3 point spike artifacts before baseline correction.

    A local median is used as the robust estimate of the expected signal.
    Points whose residual is larger than threshold*MAD are replaced by the
    local median. This is intended for cosmic-ray-like artifacts, not for
    removing true Raman/SERS bands.
    """
    y_work = _fill_nonfinite(np.asarray(y, dtype=float))
    if y_work.size < 5:
        return y_work

    window = max(3, int(window))
    if window % 2 == 0:
        window += 1

    for _ in range(max(1, int(passes))):
        local = median_filter(y_work, size=window, mode="nearest")
        residual = y_work - local
        sigma = robust_mad_sigma(residual)
        if not np.isfinite(sigma) or sigma <= 0:
            break
        spike_mask = np.abs(residual) > float(threshold) * sigma
        if not np.any(spike_mask):
            break
        y_work[spike_mask] = local[spike_mask]

    return y_work


def arpls_baseline(
    y: np.ndarray,
    lam: float = 3e5,
    diff_order: int = 2,
    max_iter: int = 80,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Adaptive reweighted penalized least-squares baseline.

    arPLS is often more stable than simple ALS/IASLS for Raman spectra with
    curved fluorescence backgrounds because the weights are updated smoothly
    from the negative residual distribution.
    """
    y = _fill_nonfinite(np.asarray(y, dtype=float))
    n = y.size
    if n < 3:
        return np.zeros_like(y, dtype=float)

    E = sparse.eye(n, format="csc")
    D = E[1:] - E[:-1]
    for _ in range(max(1, int(diff_order)) - 1):
        D = D[1:] - D[:-1]
    H = (float(lam) * (D.T @ D)).tocsc()

    w = np.ones(n, dtype=float)
    z = np.zeros(n, dtype=float)
    for _ in range(max_iter):
        W = sparse.diags(w, 0, shape=(n, n), format="csc")
        z_new = spsolve(W + H, w * y)
        residual = y - z_new
        negative = residual[residual < 0]
        if negative.size < 3:
            z = z_new
            break
        mean_neg = np.mean(negative)
        std_neg = np.std(negative)
        if not np.isfinite(std_neg) or std_neg <= 1e-12:
            z = z_new
            break
        # The clipped exponent avoids overflow while retaining the arPLS shape.
        exponent = np.clip(2.0 * (residual - (2.0 * std_neg - mean_neg)) / std_neg, -60, 60)
        w_new = 1.0 / (1.0 + np.exp(exponent))
        denom = np.linalg.norm(w) + 1e-12
        if np.linalg.norm(w_new - w) / denom < tol:
            z = z_new
            break
        w = w_new
        z = z_new
    return z


def snip_baseline(y: np.ndarray, iterations: int = 60) -> np.ndarray:
    """
    SNIP-like baseline estimator for positive Raman/SERS spectra.

    The algorithm works on a log-transformed signal and progressively clips
    broad convex background. It is robust for broad fluorescence backgrounds,
    but high iteration counts may subtract broad Raman features.
    """
    y = _fill_nonfinite(np.asarray(y, dtype=float))
    if y.size < 5:
        return np.zeros_like(y, dtype=float)

    offset = min(0.0, float(np.nanmin(y)))
    y_pos = y - offset
    y_pos = y_pos - np.nanmin(y_pos) + 1.0
    z = np.log1p(y_pos)
    n = z.size
    max_iter = min(int(iterations), max(1, (n - 1) // 2))

    for k in range(1, max_iter + 1):
        left = z[:-2 * k]
        right = z[2 * k:]
        avg = 0.5 * (left + right)
        center = z[k:n - k]
        z[k:n - k] = np.minimum(center, avg)

    baseline = np.expm1(z) + offset
    return baseline


def estimate_background_scale(
    x: np.ndarray,
    y: np.ndarray,
    background: np.ndarray,
    *,
    max_scale: float = 5.0,
) -> float:
    """
    Estimate a non-negative scale for autofluorescence/background subtraction.

    The scale is fitted only on baseline-dominated points, selected by a
    robust residual mask after a light Savitzky-Golay smoothing. This prevents
    strong Raman peaks from controlling the background scale.
    """
    y = _fill_nonfinite(np.asarray(y, dtype=float))
    b = _fill_nonfinite(np.asarray(background, dtype=float))
    if y.size != b.size or y.size < 5:
        return 1.0
    if not np.any(np.isfinite(b)) or np.nanmax(np.abs(b)) <= 0:
        return 0.0

    smooth_y = _safe_savgol_1d(y, 31, 3)
    residual = y - smooth_y
    sigma = robust_mad_sigma(residual)
    if not np.isfinite(sigma) or sigma <= 0:
        mask = np.isfinite(y) & np.isfinite(b)
    else:
        mask = np.isfinite(y) & np.isfinite(b) & (np.abs(residual) < 2.5 * sigma)

    if np.sum(mask) < 10:
        mask = np.isfinite(y) & np.isfinite(b)
    denom = float(np.dot(b[mask], b[mask]))
    if denom <= 1e-12:
        return 0.0
    scale = float(np.dot(y[mask], b[mask]) / denom)
    if not np.isfinite(scale):
        return 1.0
    return float(np.clip(scale, 0.0, max_scale))


def nonnegative_baseline_corrected(y: np.ndarray) -> np.ndarray:
    """Shift a corrected spectrum so small negative baseline residuals do not dominate plotting."""
    y = _fill_nonfinite(np.asarray(y, dtype=float))
    finite = y[np.isfinite(y)]
    if finite.size == 0:
        return y
    # Use a percentile rather than the absolute minimum so one bad point does not shift everything.
    low = np.percentile(finite, 1.0)
    if np.isfinite(low) and low < 0:
        y = y - low
    return y


def apply_sers_baseline(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    *,
    flatten_params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the selected SERS baseline correction and return
    (corrected_spectrum, estimated_baseline).
    """
    method_raw = (method or "None").strip()
    method_key = _normalise_method_name(method_raw)
    x = np.asarray(x, dtype=float)
    y = _fill_nonfinite(np.asarray(y, dtype=float))
    params = flatten_params or {}

    if method_key == "linear":
        baseline = linear_baseline(y, x)
        return y - baseline, baseline

    if method_key in {"iasls", "als"}:
        baseline = iasls_baseline(
            y,
            lam=params.get("iasls_lam", 3e5),
            p=params.get("iasls_p", 0.02),
            diff_order=2,
            max_iter=100,
        )
        return y - baseline, baseline

    if method_key == "vancouver":
        corrected, baseline, _ = vancouver_raman_filter(
            x,
            y,
            ma_window=params.get("ma_window", 5),
        )
        return corrected, baseline

    if method_key in {"arpls", "ar pls"}:
        baseline = arpls_baseline(
            y,
            lam=params.get("arpls_lam", 3e5),
            diff_order=2,
            max_iter=100,
        )
        return y - baseline, baseline

    if method_key == "snip":
        baseline = snip_baseline(y, iterations=params.get("snip_iterations", 60))
        return y - baseline, baseline

    if method_key in {"sers literature", "literature", "lit"}:
        # The complete literature workflow is handled before averaging.
        # If this function is called directly, use arPLS as a safe fallback.
        baseline = arpls_baseline(
            y,
            lam=params.get("arpls_lam", 3e5),
            diff_order=2,
            max_iter=100,
        )
        return y - baseline, baseline

    baseline = np.zeros_like(y, dtype=float)
    return y.copy(), baseline

def vector_normalize_array(values: np.ndarray) -> np.ndarray:
    """
    Vector-normalize a one-dimensional spectrum safely.
    """
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return values.copy()

    norm = np.linalg.norm(values[finite])
    if not np.isfinite(norm) or norm == 0:
        return values.copy()

    return values / norm

# Wavelet Denoising
def wavelet(data, wavelet='db4', level=None):
    if pywt is None:
        raise ImportError("PyWavelets is required for wavelet denoising. Install the PyWavelets package.")
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.std(coeffs[-1])  # Thresholding noise
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)[:len(data)]

def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    """Simple mean filter (windowed moving average)."""
    y = np.asarray(y, dtype=float)
    if window is None or window <= 1:
        return y
    window = int(window)
    kernel = np.ones(window, dtype=float) / window
    # "same" keeps array length; edges are effectively zero-padded
    return np.convolve(y, kernel, mode="same")

def nearest_bin_mask(x: np.ndarray, centers: list[float], n_bins_each_side: int = 3) -> np.ndarray:
    mask = np.zeros_like(x, dtype=bool)
    for c in centers:
        idx = np.argmin(np.abs(x - c))
        i0 = max(0, idx - n_bins_each_side)
        i1 = min(len(x), idx + n_bins_each_side + 1)
        mask[i0:i1] = True
    return mask

def gaussian_peak_gain(
    x: np.ndarray,
    centers: list[float],
    gain_factors: list[float],
    sigma: float = 4.0,
) -> np.ndarray:
    """
    Build a smooth multiplicative gain profile around selected Raman shifts.

    gain(x) = 1 + sum_i ((gain_factors[i] - 1) * gaussian_i)

    sigma is in cm^-1.
    """
    x = np.asarray(x, dtype=float)
    gain = np.ones_like(x, dtype=float)

    for c, gf in zip(centers, gain_factors):
        gain += (gf - 1.0) * np.exp(-0.5 * ((x - c) / sigma) ** 2)

    return gain

def enhance_target_peaks(
    x: np.ndarray,
    y: np.ndarray,
    baseline: np.ndarray,
    *,
    centers: list[float],
    gain_factors: list[float],
    sigma: float = 4.0,
    only_positive_residual: bool = True,
) -> np.ndarray:
    """
    Display-only enhancement of selected peaks.

    Amplifies the signal relative to the baseline near specified centers.

    Parameters
    ----------
    x : Raman shift axis
    y : original spectrum
    baseline : estimated baseline
    centers : target peaks, e.g. [730, 1042, 1328]
    gain_factors : local gains, e.g. [1.8, 1.5, 1.7]
    sigma : width of enhancement window in cm^-1
    only_positive_residual : if True, only amplify signal above baseline
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    baseline = np.asarray(baseline, dtype=float)

    residual = y - baseline

    if only_positive_residual:
        residual_to_boost = np.maximum(residual, 0.0)
        residual_other = np.minimum(residual, 0.0)
    else:
        residual_to_boost = residual
        residual_other = 0.0

    gain = gaussian_peak_gain(
        x,
        centers=centers,
        gain_factors=gain_factors,
        sigma=sigma,
    )

    y_enhanced = baseline + gain * residual_to_boost + residual_other
    return y_enhanced


def vancouver_raman_filter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ma_window: int = 2,
    poly_order: int = 1,
    max_iter: int = 25,
    tol: float = 1e-6,
    clip_sigma: float = 3.55,
    soft_clip_alpha: float = 0.1,
    protect_centers=[732, 787, 1338]
    protect_halfwidth=5.0
    protect_weight=0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Peak-preserving Vancouver-style baseline correction with protected Raman bands.

    protect_centers:
        Raman shifts to preserve, e.g. [730, 1042, 1328]
    protect_halfwidth:
        Half-width in cm^-1 around each protected band
    protect_weight:
        Weight used in polynomial fit inside protected windows.
        Smaller -> stronger protection.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    use_nearest_bin_protection = True
    n_bins_each_side = 2

    if x.size != y.size:
        raise ValueError("x and y must have the same length.")
    if x.size < poly_order + 2:
        y_smooth = _moving_average(y, ma_window)
        baseline = np.zeros_like(y_smooth)
        return y - baseline, baseline, y_smooth

    flipped = False
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]
        flipped = True

    y_smooth = _moving_average(y, ma_window)

     # ------------------------------------------------------------------
    # Build protection mask
    # ------------------------------------------------------------------
    protect_mask = np.zeros_like(x, dtype=bool)

    if protect_centers is not None:
        # Continuous protection in Raman-shift units
        for c in protect_centers:
            protect_mask |= np.abs(x - c) <= protect_halfwidth

        # Additional discrete protection on actual sampled bins
        if use_nearest_bin_protection:
            protect_mask |= nearest_bin_mask(
                x,
                protect_centers,
                n_bins_each_side=n_bins_each_side,
            )


    y_work = y_smooth.copy()
    baseline = np.zeros_like(y_smooth)

    for _ in range(max_iter):
        # Robust weighted polynomial fit
        weights = np.ones_like(x, dtype=float)
        weights[protect_mask] = protect_weight

        coeffs = np.polyfit(x, y_work, poly_order, w=weights)
        baseline_new = np.polyval(coeffs, x)

        resid = y_smooth - baseline_new
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
        robust_sigma = 1.4826 * mad

        thresh = baseline_new + clip_sigma * robust_sigma
        is_peak = y_smooth > thresh

        # Never clip protected DNA peak zones
        is_peak[protect_mask] = False

        y_work_new = y_smooth.copy()

        # Soft clipping only outside protected regions
        y_work_new[is_peak] = (
            (1.0 - soft_clip_alpha) * y_smooth[is_peak]
            + soft_clip_alpha * thresh[is_peak]
        )

        denom = np.linalg.norm(y_work) + 1e-12
        if np.linalg.norm(y_work_new - y_work) / denom < tol:
            baseline = baseline_new
            break

        y_work = y_work_new
        baseline = baseline_new

    y_corrected = y - baseline

    if flipped:
        y_corrected = y_corrected[::-1]
        baseline = baseline[::-1]
        y_smooth = y_smooth[::-1]

    return y_corrected, baseline, y_smooth

def clear_input_files():
    input_files_var.set("")
    input_files_entry.delete(0, tk.END)  # Clear the entry field
    if len(alias_entries.values())> 0:
        for entry in alias_entries.values():
            entry.destroy()
        alias_entries.clear()
        submit_button.grid(row=7, column=0, columnspan=3, pady=10)


def update_color_button_state(button: tk.Button, color: str | None) -> None:
    """Update the per-row color button to show either Auto or the selected color."""
    color = (color or "").strip()

    if color:
        try:
            button.config(text="", bg=color, activebackground=color)
        except Exception:
            button.config(text="Color")
    else:
        try:
            default_bg = button.master.cget("background")
            button.config(text="Auto", bg=default_bg, activebackground=default_bg)
        except Exception:
            button.config(text="Auto")


def choose_alias_color(color_var: tk.StringVar, color_button: tk.Button) -> None:
    """Open a Tk color chooser and store the selected hex color for one row."""
    current_color = (color_var.get() or "").strip() or "#1f77b4"
    try:
        parent = root
    except NameError:
        parent = None

    selected_color = colorchooser.askcolor(
        title="Choose plot color",
        color=current_color,
        parent=parent,
    )[1]

    if selected_color is None:
        return

    color_var.set(selected_color)
    update_color_button_state(color_button, selected_color)


def reset_alias_color(color_var: tk.StringVar, color_button: tk.Button) -> None:
    """Return one row to Matplotlib's automatic color cycle."""
    color_var.set("")
    update_color_button_state(color_button, "")


def rebuild_alias_entries(grouped_files, preserved_state=None):
    if preserved_state is None:
        preserved_state = {}

    if len(alias_entries) > 0:
        for entry in alias_entries.values():
            entry.destroy()
        alias_entries.clear()

    for i, group in enumerate(grouped_files.keys()):
        previous_alias = preserved_state.get(group, {}).get("alias", "")
        previous_baseline = preserved_state.get(group, {}).get("baseline", "None")
        # Backward compatibility with older UI state: checked Flatten -> Vancouver.
        if "baseline" not in preserved_state.get(group, {}) and preserved_state.get(group, {}).get("flatten", False):
            previous_baseline = "Vancouver"
        previous_dna = preserved_state.get(group, {}).get("dna", False)
        previous_color = preserved_state.get(group, {}).get("color", "")

        entry_label = tk.Label(frame, text=f"{group.split('/')[-1]}: ")
        entry_label.grid(row=6+i, column=0, padx=10, pady=5, sticky="e")

        entry_entry = tk.Entry(frame, width=60)
        entry_entry.grid(row=6+i, column=1, padx=10, pady=5, sticky="we")
        entry_entry.insert(0, previous_alias)

        baseline_var = tk.StringVar(value=previous_baseline)
        baseline_dropdown = ttk.Combobox(
            frame,
            textvariable=baseline_var,
            state="readonly",
            width=12,
            values=["None", "Linear", "IASLS", "Vancouver", "SNIP", "arPLS", "SERS Literature"],
        )
        baseline_dropdown.grid(row=6+i, column=2, padx=10, pady=5, sticky="w")

        dna_row_var = tk.BooleanVar(value=previous_dna)
        dna_row_checkbox = tk.Checkbutton(frame, text="DNA", variable=dna_row_var)
        dna_row_checkbox.grid(row=6+i, column=3, padx=10, pady=5, sticky="w")

        color_var = tk.StringVar(value=previous_color)
        color_button = tk.Button(frame, text="Auto", width=8)
        color_button.grid(row=6+i, column=4, padx=10, pady=5, sticky="w")
        color_button.config(command=lambda cv=color_var, btn=color_button: choose_alias_color(cv, btn))
        color_button.bind("<Button-3>", lambda _event, cv=color_var, btn=color_button: reset_alias_color(cv, btn))
        color_button.bind("<Control-Button-1>", lambda _event, cv=color_var, btn=color_button: reset_alias_color(cv, btn))
        update_color_button_state(color_button, previous_color)

        entry_delete = ttk.Button(frame, text="Delete")
        entry_delete.grid(row=6+i, column=5, padx=10, pady=5, sticky="w")

        entry = AliasEntry(
            entry_label,
            entry_entry,
            baseline_dropdown,
            baseline_var,
            dna_row_checkbox,
            dna_row_var,
            color_button,
            color_var,
            entry_delete,
            group,
        )
        entry.button.config(command=lambda e=entry: delete_alias_entry(e))
        alias_entries[group] = entry

    submit_button.grid(row=7+len(grouped_files), column=0, columnspan=3, pady=10)
    container.update_scrollbar_visibility(threshold_height=400)
    on_resize()

def clear_autofluorescence_files():
    autofluorescence_var.set("")
    autofluorescence_entry.config(state="disabled")
    autofluorescence_button.config(state="disabled")
    autofluorescence_entry.delete(0, tk.END)  # Clear the entry field

def delete_alias_entry(entry):
    preserved_state = {
        name: {
            "alias": alias_entry.entry.get(),
            "baseline": alias_entry.baseline_var.get(),
            "dna": alias_entry.dna_var.get(),
            "color": alias_entry.color_var.get(),
        }
        for name, alias_entry in alias_entries.items()
        if name != entry.name
    }

    input_files = [x.strip() for x in input_files_var.get().split(',') if x.strip() != ""]
    input_files = [x for x in input_files if entry.name not in os.path.splitext(os.path.basename(x))[0] and not os.path.splitext(os.path.basename(x))[0].startswith(f"{entry.name}_")]
    input_files_var.set(", ".join(input_files))

    grouped_files = defaultdict(list)
    for file_path in input_files:
        basename = "".join(file_path.split('/')[-1]).split('.')[0]
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files[basename].append(file_path)
    grouped_files.pop('', None)

    rebuild_alias_entries(grouped_files, preserved_state)

# Select input files
def select_input_files():
    files = filedialog.askopenfilenames(title="Select Input Files",filetypes=[("Text files", "*.txt")])
    existing_files = [x.strip() for x in input_files_var.get().split(',') if x.strip() != ""]
    file_paths = existing_files + [x for x in files if x not in existing_files]
    input_files_var.set(", ".join(file_paths))
    print("filepaths: ",file_paths)

    grouped_files = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0]
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files[basename].append(file_path)
    grouped_files.pop('', None)

    preserved_state = {
        name: {
            "alias": entry.entry.get(),
            "baseline": entry.baseline_var.get(),
            "dna": entry.dna_var.get(),
            "color": entry.color_var.get(),
        }
        for name, entry in alias_entries.items()
    }

    rebuild_alias_entries(grouped_files, preserved_state)


# Select autofluorescence files
def select_autofluorescence_files():
    if fluorophors[solution_var.get()] == "True":
        files = filedialog.askopenfilenames(title="Select Autofluorescence Files",filetypes=[("Text files", "*.txt")])
        autofluorescence_var.set(", ".join(files))

# Update autofluorescence field based on solution selection
def on_solution_change(event):
    print(f"solution: {solution_var.get()} fd {fluorophors[solution_var.get()]}")
    if solution_var.get()!="":
        # Enable or disable autofluorescence field based on solution selection
        if fluorophors[solution_var.get()] == "True":
            autofluorescence_button.config(state="normal")
            autofluorescence_entry.config(state="readonly")
        else:
            autofluorescence_button.config(state="disabled")
            autofluorescence_entry.config(state="disabled")
        
# Save new fluorophor
def save_new(new_name_var,check_var):
    # Append-adds at last
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing", "Executable")
    if os.path.exists(folder_path):
        file1 = open(f"{folder_path}/fluorophor_data.txt", "a")  # append mode
        file1.write(f"\n{new_name_var.get()}, {check_var.get()} ")
        file1.close()

# Add new fluorophor
def add_fluorophor():
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(root)
    newWindow.title("Add new Fluorophor")
    # Variables
    new_name_var = tk.StringVar()
    check_var = tk.BooleanVar()
 
    # Name field
    tk.Label(newWindow, text="Fluorophor Name:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(newWindow, textvariable=new_name_var, width=40)
    name_entry.grid(row=0, column=1, padx=10, pady=5, columnspan=2)
    
    # Checkbox for autofluorescence property
    checkbox = tk.Checkbutton(newWindow, text="Autofluorescence ",variable=check_var)
    checkbox.grid(row=1, column=0, padx=10, pady=5)
    
    # Submit button
    new_submit_button = ttk.Button(newWindow, text="Save", command=lambda: save_new(new_name_var,check_var))
    new_submit_button.grid(row=2, column=0, columnspan=3, pady=10)

class pHMeasurement:
    def __init__(self, name, alias, ph):
        self.name = name
        self.alias = alias
        self.ph = ph
    
# Select measurements to be compared 
def select_measurement_files(newWindow: tk.Toplevel,alias: dict, new_submit_button: ttk.Button, new_name_var: tk.StringVar):
    files = filedialog.askopenfilenames(title="Select Measurement Files",filetypes=[("CSV files", "*.csv")])
    file_paths = [x.strip() for x in files if x.strip() != ""]
    new_name_var.set(", ".join(files))
    
    for i,file_path in enumerate(file_paths):
        basename = "".join(file_path.split('/')[-1]).split('.')[0] # Extract basename (e.g., 'a', 'b')
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        entry_label = tk.Label(newWindow, text=f"{basename}: ")
        entry_label.grid(row=3+i, column=0, padx=10, pady=5, sticky="e")
        entry_entry = tk.Entry(newWindow, width=80)
        entry_entry.grid(row=3+i, column=1, padx=10, pady=5, columnspan=2)
        entry_ph_label = tk.Label(newWindow, text="pH: ")
        entry_ph_label.grid(row=3+i, column=3, padx=10, pady=5, sticky="e")
        entry_ph_entry = tk.Entry(newWindow, width=10)
        entry_ph_entry.insert(0, "7.0")
        entry_ph_entry.grid(row=3+i, column=4, padx=10, pady=5)
        alias[basename] = pHMeasurement(file_path,entry_entry,entry_ph_entry)
    
    # new_submit_button.grid(row=3+len(files), column=0, pady=10)

def save_ph(newWindow: tk.Toplevel, alias: dict, output_name: str = "output"):
    base_dir = find_folder("Spectra Processing")
    base_dir = [os.path.join(b, output_name) for b in base_dir]
    # base_dir=os.path.join(base_dir, output_name)
    try:
        for b in base_dir:
            os.mkdir(b)
    except OSError:
        pass
    
    # aliases = {filename: entry.entry.get() for filename, entry in alias.items()}
    # iterate over the dictionary 
    max_norom=1
    md = defaultdict(list)
    for key, value in alias.items():
        print(f"{key} : {value.alias.get()} {value.ph.get()}")
        with open(value.name, "r") as f:
            lines = f.readlines()
            data_lines = lines[1:]
            if value.alias.get() == "normalize": max_norom = float(data_lines[0].strip().split(",")[2])
            for line in data_lines:
                data = line.strip().split(",")
                md[data[0]].append((data[2],data[3],value.ph.get()))
    
    # Add a button to select interpolation method
    interpolation_methods = ["none","polynomial_1st", "polynomial_2nd", "polynomial_3rd", "polynomial_4th", "polynomial_5th", "spline","logarithmic","exponential"]
    fits = {}
    displacement = 0
    for i in md.values():
        if len(i) > displacement:
            displacement = len(i)
            
    displacement += 3
    for i,key in  enumerate(md.keys()):
        tk.Label(newWindow, text=f"Fit {key}: ").grid(row=displacement + i, column=0, padx=10, pady=5, sticky="e")
        fit_dropdown = ttk.Combobox(newWindow, textvariable= tk.StringVar(), state="readonly")
        fit_dropdown["values"] = interpolation_methods
        fit_dropdown.grid(row=displacement + i, column=1, padx=10, pady=5)
        fit_dropdown.current(0)  # Set default value
        fits[key] = fit_dropdown
    tk.Button(newWindow, text="continue",command=lambda: save_ph_cont(md, fits, max_norom,base_dir, output_name)).grid(row=displacement+1+len(md.keys()), column=0, padx=10, pady=5)
    newWindow.update_idletasks()  # Update the window to ensure the new widgets are displayed

def save_ph_cont(md: dict, fits: dict, max_norom:float, base_dirs:list[str], output_name: str = "output"):
    plt.figure(figsize=(20,14))
    fs = []
    for base_dir in base_dirs:
        fs.append(open(f"{base_dir}/interpol_{output_name}.txt","w"))
    # Iterate over the dictionary and plot the data
    for key, value in md.items():
        intensity = [float(x[0])/max_norom for x in value]
        std = [float(x[1])/max_norom for x in value]
        ph = [float(x[2]) for x in value]
        print(f"the values are {intensity} {std} {ph}")
        s = plt.scatter(ph,intensity,label=key)
        plt.errorbar(ph, intensity, yerr=std, fmt='o',  markersize=8, capsize=10)
        
        fit = fits[key].get()
        # setup the fit
        if fit == "polynomial_1st":
            # z = np.polyfit(ph, intensity, 1)
            z = curve_fit(lambda x, a, b: a*x + b, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.array(ph) + z[0][1], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_2nd":
            # z = np.polyfit(ph, intensity, 2)
            z = curve_fit(lambda x, a, b, c: a*x**2 + b*x + c, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**2 + z[0][1]*np.array(ph) + z[0][2], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_3rd":
            # z = np.polyfit(ph, intensity, 3)
            z = curve_fit(lambda x, a, b, c, d: a*x**3 + b*x**2 + c*x + d, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**3 + z[0][1]*np.array(ph)**2 + z[0][2]*np.array(ph) + z[0][3], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_4th":
            # z = np.polyfit(ph, intensity, 4)
            z = curve_fit(lambda x, a, b, c, d, e: a*x**4 + b*x**3 + c*x**2 + d*x + e, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]} {z[0][4]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**4 + z[0][1]*np.array(ph)**3 + z[0][2]*np.array(ph)**2 + z[0][3]*np.array(ph) + z[0][4], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "polynomial_5th":
            # z = np.polyfit(ph, intensity, 5)
            z = curve_fit(lambda x, a, b, c, d, e, f: a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]} {z[0][2]} {z[0][3]} {z[0][4]} {z[0][5]}\n")
            plt.plot(ph, z[0][0]*np.array(ph)**5 + z[0][1]*np.array(ph)**4 + z[0][2]*np.array(ph)**3 + z[0][3]*np.array(ph)**2 + z[0][4]*np.array(ph) + z[0][5], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "spline":
            pass
        elif fit == "logarithmic":
            # z = np.polyfit(np.log(ph), intensity, 1)
            z = curve_fit(lambda x, a, b: a*np.log(x) + b, ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.log(np.array(ph)) + z[0][1], label=f"{key} fit", color=s.get_facecolor())
        elif fit == "exponential":
            # z = np.polyfit(ph, np.log(intensity), 1)
            z = curve_fit(lambda x, a, b: a*np.exp(b*x), ph, intensity)
            for f in fs:
                f.write(f"{key} {z[0][0]} {z[0][1]}\n")
            plt.plot(ph, z[0][0]*np.exp(z[0][1]*np.array(ph)), label=f"{key} fit", color=s.get_facecolor())
        else:
            pass
        
        plt.xlabel("pH", fontsize = 30)
        plt.ylabel("Intensity [a.u]", fontsize = 30)
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.tick_params(axis='both', direction='in')
        leg = plt.legend(fontsize=32, frameon=True, framealpha=0.5, edgecolor="black")
        for l in leg.get_lines():
            l.set_linewidth(4.5)
    for base_dir in base_dirs:
        plt.savefig(f"{base_dir}/pH_plot_{output_name}.png")
    for f in fs:
        f.close()


def pH_plot():
    # Toplevel object which will 
    # be treated as a new window
    newWindow = tk.Toplevel(root)
    newWindow.title("Create pH plot")

    # Variables
    new_name_var = tk.StringVar()
    name_var = tk.StringVar()
    solution_var = tk.StringVar()
    alias = {}
 
    # Name field
    tk.Label(newWindow, text="Measurements to be compared:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    ttk.Button(newWindow, text="Select Files",command=lambda: select_measurement_files(newWindow,alias, new_submit_button, new_name_var)).grid(row=0, column=2, padx=10, pady=5)
    name_entry = tk.Entry(newWindow, textvariable=new_name_var, width=40, state="readonly")
    name_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(newWindow, text="Output folder: ").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    output_name = tk.Entry(newWindow, textvariable=name_var, width=40)
    output_name.grid(row=1, column=1, padx=10, pady=5)

    # Add a button to select interpolation method
    # interpolation_methods = ["none","polynomial_1st", "polynomial_2nd", "polynomial_3rd", "polynomial_4th", "polynomial_5th", "spline","logarithmic","exponential"]
    # tk.Label(newWindow, text="Fit: ").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    # fit_dropdown = ttk.Combobox(newWindow, textvariable=solution_var, state="readonly")
    # fit_dropdown["values"] = interpolation_methods
    # fit_dropdown.grid(row=2, column=1, padx=10, pady=5)
    # fit_dropdown.current(0)  # Set default value
    
    # Submit button
    new_submit_button = ttk.Button(newWindow, text="Save", command=lambda: save_ph(newWindow,alias, output_name.get()))
    new_submit_button.grid(row=1, column=2, pady=10)


# Find the folder with the given name in the user's Pictures directory
def find_folder(folder_name):
    documents_path = os.path.join(os.path.expanduser("~"), "Pictures")
    folder_path = os.path.join(documents_path, folder_name)
    oneDrive_path_mac = os.path.join(os.path.expanduser("~"), "Library/CloudStorage/OneDrive-UniversitateaBabeș-Bolyai/SpectraProcessing")
    oneDrive_path_win = os.path.join(os.path.expanduser("~"), "OneDrive - Universitatea Babeș-Bolyai/SpectraProcessing")
    l = []
    print(documents_path)
    if os.path.exists(folder_path):
        l.append(folder_path)
    if os.path.exists(oneDrive_path_mac):
        l.append(oneDrive_path_mac)
    if os.path.exists(oneDrive_path_win):
        l.append(oneDrive_path_win)
    if len(l) == 0: 
        return None
    else:
        return l 
    

# Retrieve data from UI
def submit():
    solution = solution_var.get()
    input_files = input_files_var.get()
    autofluorescence_files = autofluorescence_var.get()
    min_spectra = min_spectra_var.get()
    max_spectra = max_spectra_var.get()
    output_name = name_var.get()
    if output_name == "": output_name = "output"
    if min_spectra == "": min_spectra = 0.0
    else: min_spectra = float(min_spectra)
    if max_spectra == "": max_spectra = 0.0
    else: max_spectra = float(max_spectra)
    aliases = {filename: entry.entry.get() for filename, entry in alias_entries.items()}
    baseline_methods = {filename: entry.baseline_var.get() for filename, entry in alias_entries.items()}
    plot_colors = {filename: entry.color_var.get().strip() for filename, entry in alias_entries.items()}
    dna_groups = {filename: entry.dna_var.get() for filename, entry in alias_entries.items()}
    peaks = peak_display_var.get()
    normalize = normalize_var.get()
    normalize_all = normalize_all_var.get()
    denoise = denoise_var.get()
    denoise_level = denoise_level_var.get()
    flatten_level = flatten_level_var.get()
    dna = dna_var.get()
    sers_literature_mode = sers_literature_var.get()

    base_dir = find_folder("Spectra Processing")
    if base_dir is None:
        # Fall back to a local Pictures/Spectra Processing folder when the
        # expected folders do not already exist.
        fallback = os.path.join(os.path.expanduser("~"), "Pictures", "Spectra Processing")
        os.makedirs(fallback, exist_ok=True)
        base_dir = [fallback]
    base_dir = [os.path.join(b, output_name) for b in base_dir]
    # base_dir=os.path.join(base_dir, output_name)
    try:
        for b in base_dir:
            os.mkdir(b)
    except OSError:
        pass
    
    process_files(
        solution,
        input_files,
        autofluorescence_files,
        min_spectra,
        max_spectra,
        output_name,
        base_dir,
        aliases,
        baseline_methods,
        plot_colors,
        dna_groups,
        peaks,
        normalize,
        normalize_all,
        denoise,
        denoise_level,
        flatten_level,
        dna,
        sers_literature_mode,
    )


# Open a Tk window to select multiple txt files
def select_files():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_paths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
    return file_paths

# Function to normalize a spectrum
def normalize_spectrum(intensities):
    max_intensity = max(intensities)
    if max_intensity == 0:
        return intensities
    return [intensity / max_intensity for intensity in intensities]

# Process files and perform tasks
def get_spectra_reader_settings(solution: str) -> tuple[int, str, int, int]:
    """Return skip_header, delimiter, skip_footer, read_col for the selected input format."""
    skip_lines = 8
    delimiter = ';'
    skip_footer = 0
    read_col = 1

    if is_sers_bwtek_solution(solution):
        skip_lines = 1
    elif is_sers_renishaw_solution(solution):
        skip_lines = 1
        delimiter = '\t'
    elif is_sers_avantes_solution(solution):
        skip_lines = 0
    elif solution == "FT-IR":
        skip_lines = 19
        delimiter = '\t'
        skip_footer = 42
    elif solution == "UV-Vis":
        skip_lines = 19
        delimiter = '\t'
        skip_footer = 42
    elif solution == "UV-Vis absorbtion simulation":
        skip_lines = 1
        delimiter = ','
        read_col = 3
    elif solution == "UV-Vis scattering simulation":
        skip_lines = 1
        delimiter = ','
        read_col = 2
    elif solution == "UV-Vis excitation simulation":
        skip_lines = 1
        delimiter = ','
        read_col = 1

    return skip_lines, delimiter, skip_footer, read_col


def read_spectrum_file(file_path: str, solution: str) -> tuple[np.ndarray, np.ndarray]:
    """Read one spectrum as x/y arrays using the selected spectrometer format."""
    skip_lines, delimiter, skip_footer, read_col = get_spectra_reader_settings(solution)
    file_data = np.genfromtxt(
        file_path,
        encoding="UTF-8",
        dtype=np.float64,
        skip_header=skip_lines,
        delimiter=delimiter,
        skip_footer=skip_footer,
    )
    if file_data.ndim == 1:
        file_data = file_data.reshape(1, -1)
    if file_data.shape[1] <= read_col:
        raise ValueError(f"Could not read intensity column {read_col} from {file_path}")
    return np.asarray(file_data[:, 0], dtype=float), np.asarray(file_data[:, read_col], dtype=float)


def sort_and_crop_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    min_spectra: float,
    max_spectra: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort by increasing x and crop to the requested spectral limits."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    lo = np.nanmin(x) if min_spectra == 0.0 else float(min_spectra)
    hi = np.nanmax(x) if max_spectra == 0.0 else float(max_spectra)
    if lo > hi:
        lo, hi = hi, lo
    mask = (x >= lo) & (x <= hi)
    if np.sum(mask) >= 2:
        x = x[mask]
        y = y[mask]
    return x, y


def combine_processed_spectra(
    axes: list[np.ndarray],
    spectra: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate replicate spectra onto the first axis and return mean/std."""
    if len(axes) == 0:
        return np.array([]), np.array([]), np.array([])
    base_x = np.asarray(axes[0], dtype=float)
    rows = []
    for x, y in zip(axes, spectra):
        rows.append(_interp_to_axis(x, y, base_x))
    arr = np.vstack(rows)
    return base_x, np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def prepare_background_trace(
    grouped_files_baseline: dict,
    solution: str,
    min_spectra: float,
    max_spectra: float,
    flatten_params: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Read, despike, lightly smooth and average the selected autofluorescence/background files."""
    if len(grouped_files_baseline.keys()) == 0:
        return None

    # Current UI supplies a single autofluorescence/background group. Use the first group.
    background_group = list(grouped_files_baseline.keys())[0]
    axes: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    for file_path in grouped_files_baseline[background_group]:
        try:
            x, y = read_spectrum_file(file_path, solution)
        except Exception:
            # Keep backward compatibility with older AF files that used the default text format.
            file_data = np.genfromtxt(file_path, encoding="UTF-8", dtype=np.float64, skip_header=8, delimiter=';')
            if file_data.ndim == 1:
                file_data = file_data.reshape(1, -1)
            x, y = file_data[:, 0], file_data[:, 1]
        x, y = sort_and_crop_spectrum(x, y, min_spectra, max_spectra)
        if x.size < 2:
            continue
        y = despike_spectrum(
            y,
            window=flatten_params.get("despike_window", 7),
            threshold=flatten_params.get("despike_threshold", 7.0),
            passes=2,
        )
        y = _safe_savgol_1d(y, 11, 3)
        axes.append(x)
        spectra.append(y)

    if len(axes) == 0:
        return None
    return combine_processed_spectra(axes, spectra)


def process_one_sers_replicate(
    x: np.ndarray,
    y: np.ndarray,
    *,
    method: str,
    background: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    flatten_params: dict,
    denoise: bool,
    denoise_level: str,
    literature_override: bool,
    normalize_for_literature: bool,
) -> np.ndarray:
    """Apply the selected SERS workflow to a single replicate before averaging."""
    method_key = _normalise_method_name(method)
    use_literature = literature_override or method_key in {"sers literature", "literature", "lit"}
    y_work = _fill_nonfinite(np.asarray(y, dtype=float))

    if use_literature:
        # 1) Despike before subtraction/baseline so cosmic rays do not influence either step.
        y_work = despike_spectrum(
            y_work,
            window=flatten_params.get("despike_window", 7),
            threshold=flatten_params.get("despike_threshold", 7.0),
            passes=2,
        )

        # 2) Subtract an interpolated and scale-matched autofluorescence/background trace.
        if background is not None:
            bg_x, bg_y, _bg_std = background
            bg_on_axis = _interp_to_axis(bg_x, bg_y, x)
            scale = estimate_background_scale(x, y_work, bg_on_axis)
            y_work = y_work - scale * bg_on_axis

        # 3) Adaptive baseline correction. arPLS is used for the complete workflow
        # because it is less sensitive to the exact number of SNIP iterations.
        baseline = arpls_baseline(
            y_work,
            lam=flatten_params.get("arpls_lam", 3e5),
            diff_order=2,
            max_iter=100,
        )
        y_work = y_work - baseline
        y_work = nonnegative_baseline_corrected(y_work)

        # 4) Light display smoothing. If the Denoise checkbox is active, use its level;
        # otherwise use the literature-mode default from the flatten level.
        if denoise:
            sg_params = get_denoise_params(denoise_level)
            if sg_params is not None:
                y_work = _safe_savgol_1d(y_work, sg_params[0], sg_params[1])
        else:
            y_work = _safe_savgol_1d(
                y_work,
                flatten_params.get("lit_savgol_window", 11),
                flatten_params.get("lit_savgol_poly", 3),
            )

        # 5) Normalize replicate before averaging, as commonly done for stacked
        # comparison plots. This should be used for visualization, not for absolute
        # quantitative peak-height comparison.
        if normalize_for_literature:
            y_work = vector_normalize_array(y_work)
        return y_work

    # Legacy/manual SERS path: keep old choices, but make background subtraction safer.
    if background is not None:
        bg_x, bg_y, _bg_std = background
        bg_on_axis = _interp_to_axis(bg_x, bg_y, x)
        scale = estimate_background_scale(x, y_work, bg_on_axis)
        y_work = y_work - scale * bg_on_axis

    if method_key in {"snip", "arpls", "ar pls", "iasls", "als", "vancouver", "linear"}:
        y_work, _baseline = apply_sers_baseline(x, y_work, method, flatten_params=flatten_params)

    if denoise:
        sg_params = get_denoise_params(denoise_level)
        if sg_params is not None:
            y_work = _safe_savgol_1d(y_work, sg_params[0], sg_params[1])

    return y_work


def process_files(solution: str, input_files: str, autofluorescence_files: str, min_spectra: float, max_spectra: float,
                  output_name: str, basedirs: str, aliases: dict, baseline_methods: dict, plot_colors: dict, dna_groups: dict, show_peaks: bool, normalize: bool, normalize_all: bool,
                  denoise: bool, denoise_level: str, flatten_level: str, dna: bool, sers_literature_mode: bool = False):
    # Split the input files string into a list of file paths
    file_paths = input_files.split(',')
    file_paths = [x.strip() for x in file_paths if x.strip() != ""]
    file_paths_af = autofluorescence_files.split(',')
    file_paths_af = [x.strip() for x in file_paths_af if x.strip() != ""]

    # Group files by their basename prefix
    grouped_files = defaultdict(list)
    grouped_files_baseline = defaultdict(list)
    for file_path in file_paths:
        basename = "".join(file_path.split('/')[-1]).split('.')[0]
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files[basename].append(file_path)
    print("my grouped files are:\n", grouped_files)

    for file_path in file_paths_af:
        basename = "".join(file_path.split('/')[-1]).split('.')[0]
        if re.search(r"_\d$", basename):
            basename = basename[:basename.rfind('_')]
        grouped_files_baseline[basename].append(file_path)
    grouped_files_baseline.pop('', None)

    flatten_params = get_flatten_params(flatten_level)
    background = prepare_background_trace(
        grouped_files_baseline,
        solution,
        min_spectra,
        max_spectra,
        flatten_params,
    )

    data = []
    max_ctr_nom = 1.0
    control = False
    lit_groups: set[str] = set()

    for group, files in grouped_files.items():
        print(f"Processing group: {group}")
        axes: list[np.ndarray] = []
        spectra: list[np.ndarray] = []
        method = baseline_methods.get(group, "None")
        method_key = _normalise_method_name(method)
        group_literature = is_sers_solution(solution) and (
            sers_literature_mode or method_key in {"sers literature", "literature", "lit"}
        )
        if group_literature:
            lit_groups.add(group)

        for file_path in files:
            x, y = read_spectrum_file(file_path, solution)
            x, y = sort_and_crop_spectrum(x, y, min_spectra, max_spectra)
            if x.size < 2:
                continue

            if is_sers_solution(solution):
                y_processed = process_one_sers_replicate(
                    x,
                    y,
                    method=method,
                    background=background,
                    flatten_params=flatten_params,
                    denoise=denoise,
                    denoise_level=denoise_level,
                    literature_override=sers_literature_mode,
                    normalize_for_literature=True,
                )
            else:
                y_processed = _fill_nonfinite(y)
                if denoise:
                    sg_params = get_denoise_params(denoise_level)
                    if sg_params is not None:
                        y_processed = _safe_savgol_1d(y_processed, sg_params[0], sg_params[1])
                if background is not None:
                    bg_x, bg_y, _bg_std = background
                    y_processed = y_processed - _interp_to_axis(bg_x, bg_y, x)

            axes.append(x)
            spectra.append(y_processed)

        if len(axes) == 0:
            continue

        mean_x, mean_y, std_y = combine_processed_spectra(axes, spectra)
        measurement = Measurement(group, mean_y, std_y, mean_x, aliases.get(group, "") or group)
        data.append(measurement)

        if "_contr_" in group or "_control_" in group or "_ctr_" in group or "_ctrl_" in group:
            max_ctr_nom, _, _ = measurement.find_max()
            control = True

    max_control = max_ctr_nom

    # SERS normalization. Literature-mode groups are already vector-normalized per replicate.
    if is_sers_solution(solution):
        for measurement in data:
            if measurement.name in lit_groups:
                continue
            if normalize_all:
                max_val, _, _ = measurement.find_max()
                if np.isfinite(max_val) and max_val != 0:
                    measurement.value = np.asarray(measurement.value, dtype=float) / max_val
                    measurement.std = np.asarray(measurement.std, dtype=float) / abs(max_val)
            elif normalize:
                norm = np.linalg.norm(np.asarray(measurement.value, dtype=float)[np.isfinite(measurement.value)])
                if np.isfinite(norm) and norm != 0:
                    measurement.value = np.asarray(measurement.value, dtype=float) / norm
                    measurement.std = np.asarray(measurement.std, dtype=float) / norm

        # Do not artificially enhance or suppress DNA peaks. The DNA checkbox is kept
        # only for backwards compatibility with older saved workflows.
        for measurement in data:
            measurement.visualize = []
    elif normalize:
        if not control:
            for measurement in data:
                maxi, _, _ = measurement.find_max()
                if maxi > max_ctr_nom:
                    max_ctr_nom = maxi
        for measurement in data:
            measurement.value = np.array([float(x) / max_ctr_nom for x in measurement.value])
            measurement.std = np.array([float(x) / max_ctr_nom for x in measurement.std])
    elif normalize_all:
        for measurement in data:
            max_ctr_nom, _, _ = measurement.find_max()
            if max_ctr_nom != 0:
                measurement.value = np.array([float(x) / max_ctr_nom for x in measurement.value])
                measurement.std = np.array([float(x) / abs(max_ctr_nom) for x in measurement.std])

    # Export processed mean/std data.
    NUM_COLOR = 0
    output = []
    for measurement in data:
        NUM_COLOR += 1
        output.append({
            'Name': measurement.alias,
            **dict(zip(measurement.wave, measurement.value))
        })
        if normalize:
            output.append({
                'Name': f"{measurement.alias}_raw_std_scaled",
                **dict(zip(measurement.wave, measurement.std * max_ctr_nom))
            })
        output.append({
            'Name': f"{measurement.alias}_std",
            **dict(zip(measurement.wave, measurement.std))
        })

    df = pd.DataFrame(output)

    for basedir in basedirs:
        df.to_excel(f'{basedir}/output_{output_name}.xlsx', index=False)
        print(f"Data exported to {basedir}/output_{output_name}.xlsx")
        print(basedir)

    # Plot the data
    if NUM_COLOR < 10:
        cm = plt.get_cmap("tab10")
        plt.rcParams['axes.prop_cycle'] = plt.cycler("color", [cm(1. * i / 10) for i in range(10)])
    else:
        cm = plt.get_cmap('gist_rainbow')
        plt.rcParams['axes.prop_cycle'] = plt.cycler("color", [cm(1. * i / NUM_COLOR) for i in range(NUM_COLOR)])

    plot_arrays = []
    for measurement in data:
        arr = np.asarray(measurement.value, dtype=float)
        if len(measurement.visualize) > 0:
            arr = np.asarray(measurement.visualize, dtype=float)
        plot_arrays.append(arr)

    if plot_arrays:
        max_intensity = max(
            (np.nanmax(arr) for arr in plot_arrays if arr.size > 0 and np.any(np.isfinite(arr))),
            default=1.0,
        )
    else:
        max_intensity = 1.0

    spacing = 0.8 * max_intensity if np.isfinite(max_intensity) and max_intensity > 0 else 1.0
    colors = []
    plt.figure(figsize=(20, 14))
    selected_dna_groups = {name for name, enabled in dna_groups.items() if enabled}
    use_group_dna_selection = len(selected_dna_groups) > 0

    for plot_index, measurement in enumerate(data):
        offset = plot_index * spacing if is_sers_solution(solution) or solution == "FT-IR" else 0.0
        selected_color = (plot_colors or {}).get(measurement.name, "").strip() or None
        color = selected_color
        plot_kwargs = {"linewidth": 2.5}
        if selected_color:
            plot_kwargs["color"] = selected_color

        try:
            use_dna_visualization = False
            if use_dna_visualization and len(measurement.visualize) > 0:
                line = plt.plot(measurement.wave, measurement.visualize + offset, label=measurement.alias, **plot_kwargs)
            else:
                line = plt.plot(measurement.wave, measurement.value + offset, label=measurement.alias, **plot_kwargs)
            color = line[0].get_color()
            colors.append(color)
            if show_peaks:
                width = 30
                if is_sers_solution(solution):
                    width = 10
                elif solution == "FT-IR":
                    width = 25
                elif solution == "UV-Vis":
                    width = 15
                peaks, _ = find_peaks(measurement.value, width=width)
                label = [str(round(f)) for f in list(measurement.wave[peaks])]
                h = [measurement.value[p] + offset for p in peaks]
                plt.scatter(measurement.wave[peaks], h, color=color)
                for i, p_idx in enumerate(peaks):
                    plt.text(measurement.wave[p_idx], h[i], label[i], fontsize=24, color=color)
        except ValueError:
            messagebox.showerror("Error", "Please adjust the domain to be plotted !")

        try:
            if len(measurement.visualize) > 0:
                plt.fill_between(
                    measurement.wave,
                    measurement.visualize + measurement.std + offset,
                    measurement.visualize - measurement.std + offset,
                    alpha=0.5,
                    color=color,
                )
            else:
                plt.fill_between(
                    measurement.wave,
                    measurement.value + measurement.std + offset,
                    measurement.value - measurement.std + offset,
                    alpha=0.5,
                    color=color,
                )
        except ValueError:
            pass

    if is_sers_solution(solution):
        plt.xlabel("Raman Shift [cm$^{-1}$]", fontsize=35)
        if normalize or normalize_all or sers_literature_mode or len(lit_groups) > 0:
            plt.ylabel("Normalized intensity [a.u.]", fontsize=35)
        else:
            plt.ylabel("Intensity [a.u.]", fontsize=35)
    elif solution == "UV-Vis":
        plt.xlabel("Wavelength [nm]", fontsize=35)
        plt.ylabel("Extinction [a.u]", fontsize=35)
        if normalize or normalize_all:
            plt.ylabel("Normalized intensity [a.u]", fontsize=35)
    elif solution == "FT-IR":
        plt.xlabel("Wavenumber [cm$^{-1}$]", fontsize=35)
        plt.ylabel("Transmittance [a.u]", fontsize=35)
        plt.yticks([])
    else:
        plt.xlabel("Wavelength [nm]", fontsize=35)
        plt.ylabel("Normalized intensity [a.u]", fontsize=35)

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.tick_params(axis='both', direction='in')
    leg = plt.legend(fontsize=28, frameon=True, framealpha=0.5, edgecolor="black")
    for l in leg.get_lines():
        l.set_linewidth(2.5)

    for basedir in basedirs:
        plt.savefig(f"{basedir}/plot_{output_name}.png")
        plt.show()

    plt.close()

    # Save relevant data to a CSV file
    for basedir in basedirs:
        with open(f'{basedir}/max_values_{output_name}.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Name', 'Max Wave', 'Max Value', 'Max Std', 'Color']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for measurement, color in zip(data, colors):
                max_value, max_wave, max_std = measurement.find_max()
                writer.writerow({
                    'Name': measurement.alias,
                    'Max Value': max_value,
                    'Max Wave': max_wave,
                    'Max Std': max_std,
                    'Color': color,
                })
        for file in file_paths:
            dest = shutil.copyfile(file, f"{basedir}/{file.split('/')[-1]}")
            print(f"File {file} copied to {dest}")

def read_fluorophor(l):
    documents_path = os.path.join(os.path.expanduser("~"), "Documents")
    folder_path = os.path.join(documents_path, "Spectra Processing","Executable")
    if os.path.exists(folder_path):
        with open(f"{folder_path}/fluorophor_data.txt", "r") as file:
            for line in file:
                l[line.strip().split(',')[0].strip()] = line.strip().split(',')[1].strip()
    else: print("help!")

def updatelist():
    read_fluorophor(fluorophors)
    solution_dropdown["values"]=[f for f in fluorophors.keys()]

def replace_dots_in_filenames(directory):
    for root_dir, _, files in os.walk(directory):
        for filename in files:
            old_path = os.path.join(root_dir, filename)
            # Split filename and extension
            name, ext = os.path.splitext(filename)
            
            # Check for version suffix (e.g., p1.1.txt)
            match = re.search("([0-9].[0-9])$", name)
            print(name,match)
            if match:
                base_name = name[:match.start()]  # Extract the part before the version suffix
                new_name = base_name.replace('.', '_')
                new_name = new_name.replace(' ', '_')
                new_name = new_name.replace(',', 'v')
                new_name += match.group(1)
                new_name += ext
            else:
                new_name = name.replace('.', '_')
                new_name = new_name.replace(' ', '_')
                new_name = new_name.replace(',', 'v')
                new_name += ext
        
            new_path = os.path.join(root_dir, new_name)
            if old_path != new_path:
                os.rename(old_path, new_path)
                print(f'Renamed: "{old_path}" -> "{new_path}"')

def rename_files():
    folder_path = filedialog.askdirectory()
    if folder_path:
        replace_dots_in_filenames(folder_path)
    
class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        self.canvas_window = canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.canvas = canvas
        self.scrollbar = scrollbar

    def update_scrollbar_visibility(self, threshold_height):
        self.update_idletasks()
        content_height = self.scrollable_frame.winfo_height()
        if content_height > threshold_height:
            self.scrollbar.pack(side="right", fill="y")
        else:
            self.scrollbar.pack_forget()

class AliasEntry():
    def __init__(self, label: tk.Label, entry: tk.Entry, baseline_dropdown: ttk.Combobox, baseline_var: tk.StringVar,
                 dna_checkbox: tk.Checkbutton, dna_var: tk.BooleanVar, color_button: tk.Button, color_var: tk.StringVar,
                 button: ttk.Button, name: str):
        self.label = label
        self.entry = entry
        self.baseline_dropdown = baseline_dropdown
        self.baseline_var = baseline_var
        self.dna_checkbox = dna_checkbox
        self.dna_var = dna_var
        self.color_button = color_button
        self.color_var = color_var
        self.button = button
        self.name = name
    
    def destroy(self):
        self.label.destroy()
        self.entry.destroy()
        self.baseline_dropdown.destroy()
        self.dna_checkbox.destroy()
        self.color_button.destroy()
        self.button.destroy()
    
def on_resize():
        container.update_scrollbar_visibility(threshold_height=400)

def main() -> None:
    # root is created by module-level code above; globals are fine
    # root.mainloop()
    pass


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Spectra processing")
    root.geometry("1400x850")  # Starting size

    container = ScrollableFrame(root)
    container.pack(fill="both", expand=True)

    frame = container.scrollable_frame

    # Variables
    solution_var = tk.StringVar()
    input_files_var = tk.StringVar()
    autofluorescence_var = tk.StringVar()
    min_spectra_var = tk.StringVar()
    max_spectra_var = tk.StringVar()
    name_var = tk.StringVar()
    peak_display_var = tk.BooleanVar()
    normalize_var = tk.BooleanVar()
    normalize_all_var = tk.BooleanVar()
    denoise_var = tk.BooleanVar()
    denoise_level_var = tk.StringVar(value="Medium")
    flatten_level_var = tk.StringVar(value="Medium")
    sers_literature_var = tk.BooleanVar(value=False)
    dna_var = tk.BooleanVar()
    fluorophors = {"": "False"}
    alias_entries = {}

    read_fluorophor(fluorophors)
    
    # Solution dropdown
    tk.Label(frame, text="Spectra:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    solution_dropdown = ttk.Combobox(frame, textvariable=solution_var, state="readonly", postcommand=updatelist)
    solution_dropdown["values"] = [f for f in fluorophors.keys()]
    solution_dropdown.grid(row=0, column=1, padx=10, pady=5)
    solution_dropdown.bind("<<ComboboxSelected>>", on_solution_change)
    
    # Insert new Fluorophor
    new_spectra_button = ttk.Button(frame, text="Add new Spectrometer", command=add_fluorophor)
    new_spectra_button.grid(row=0, column=2, pady=10)

    # Rename files to naming convention
    rename_files_button = ttk.Button(frame, text="Rename Files", command=rename_files)
    rename_files_button.grid(row=0, column=3, pady=10)

    # Input files field
    tk.Label(frame, text="Input Files:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    input_files_button = ttk.Button(frame, text="Select Files", command=select_input_files)
    # input_files_button = ttk.Button(frame, text="Select Files", command=select_input_files, postcommand=get_aliases)
    input_files_button.grid(row=1, column=1, padx=10, pady=5)
    input_files_entry = tk.Entry(frame, textvariable=input_files_var, state="readonly", width=40)
    input_files_entry.grid(row=1, column=2, padx=10, pady=5)
    input_files_button_clear = ttk.Button(frame, text="Clear", command=clear_input_files)
    input_files_button_clear.grid(row=1, column=3, padx=10, pady=5)

    # Autofluorescence field
    tk.Label(frame, text="Autofluorescence:").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    autofluorescence_button = ttk.Button(frame, text="Select Files", command=select_autofluorescence_files)
    autofluorescence_button.grid(row=2, column=1, padx=10, pady=5)
    autofluorescence_entry = tk.Entry(frame, textvariable=autofluorescence_var, state='disabled', width=40)
    autofluorescence_entry.grid(row=2, column=2, padx=10, pady=5)
    autofluorescence_files_button_clear = ttk.Button(frame, text="Clear", command=clear_autofluorescence_files)
    autofluorescence_files_button_clear.grid(row=2, column=3, padx=10, pady=5)

    # Spectra limits
    tk.Label(frame, text="Spectra Min:").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    min_spectra_entry = tk.Entry(frame, textvariable=min_spectra_var, width=10)
    min_spectra_entry.grid(row=3, column=1, padx=10, pady=5, sticky="w")

    tk.Label(frame, text="Spectra Max:").grid(row=4, column=0, padx=10, pady=5, sticky="e")
    max_spectra_entry = tk.Entry(frame, textvariable=max_spectra_var, width=10)
    max_spectra_entry.grid(row=4, column=1, padx=10, pady=5, sticky="w")

    # Peak display checkbox
    peak_display_var = tk.BooleanVar()
    peak_display_checkbox = tk.Checkbutton(frame, text="Show Peak Values", variable=peak_display_var)
    peak_display_checkbox.grid(row=3, column=2, padx=10, pady=5, sticky="w")

    # Normalize checkbox
    normalize_var = tk.BooleanVar()
    normalize_checkbox = tk.Checkbutton(frame, text="Normalize", variable=normalize_var)
    normalize_checkbox.grid(row=4, column=2, padx=10, pady=5, sticky="w")

    # Normalize_all_to_one checkbox
    normalize_all_var = tk.BooleanVar()
    normalize_all_checkbox = tk.Checkbutton(frame, text="Normalize all to 1", variable=normalize_all_var)
    normalize_all_checkbox.grid(row=4, column=3, padx=10, pady=5, sticky="w")

    # Normalize by group checkbox
    normalize_by_group_var = tk.BooleanVar()
    normalize_by_group_checkbox = tk.Checkbutton(frame, text="Normalize by group", variable=normalize_by_group_var)
    normalize_by_group_checkbox.grid(row=5, column=3, padx=10, pady=5, sticky="w")

    # Denoise checkbox
    denoise_var = tk.BooleanVar()
    denoise_checkbox = tk.Checkbutton(frame, text="Denoise", variable=denoise_var)
    denoise_checkbox.grid(row=3, column=3, padx=10, pady=5, sticky="w")

    # Denoise level dropdown
    tk.Label(frame, text="Denoise level:").grid(row=5, column=4, padx=10, pady=5, sticky="e")
    denoise_level_dropdown = ttk.Combobox(frame, textvariable=denoise_level_var, state="readonly", width=12)
    denoise_level_dropdown["values"] = ["Low", "Medium", "High", "Very High", "Extreme"]
    denoise_level_dropdown.grid(row=5, column=5, padx=10, pady=5, sticky="w")
    denoise_level_dropdown.current(1)  # Medium

    tk.Label(frame, text="Baseline strength:").grid(row=5, column=6, padx=10, pady=5, sticky="e")
    flatten_level_dropdown = ttk.Combobox(frame, textvariable=flatten_level_var, state="readonly", width=12)
    flatten_level_dropdown["values"] = ["Low", "Medium", "High", "Very High", "Extreme"]
    flatten_level_dropdown.grid(row=5, column=7, padx=10, pady=5, sticky="w")
    flatten_level_dropdown.current(1)  # Medium

    # DNA checkbox
    dna_var = tk.BooleanVar()
    dna_checkbox = tk.Checkbutton(frame, text="DNA", variable=dna_var)
    dna_checkbox.grid(row=3, column=4, padx=10, pady=5, sticky="w")

    # Complete SERS literature-style workflow
    sers_literature_checkbox = tk.Checkbutton(
        frame,
        text="SERS literature mode",
        variable=sers_literature_var,
    )
    sers_literature_checkbox.grid(row=4, column=4, padx=10, pady=5, sticky="w")

    # Name field
    tk.Label(frame, text="Output Name:").grid(row=5, column=0, padx=10, pady=5, sticky="e")
    name_entry = tk.Entry(frame, textvariable=name_var, width=40)
    name_entry.grid(row=5, column=1, padx=10, pady=5)

    # pH button
    pH_button = ttk.Button(frame, text="pH Plotting", command=pH_plot)
    pH_button.grid(row=5, column=2, padx=10, pady=5)

    # Submit button
    submit_button = ttk.Button(frame, text="Submit", command=submit)
    submit_button.grid(row=6, column=0, columnspan=3, pady=10)

    on_solution_change(None)

    root.mainloop()
    # main()
