"""
Task 1: Finding the Sea Floor
--------------------------------------------------------------------
Name : M Karthick
ID   : 2026A7PS0303H
--------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import savgol_filter

CSV_PATH = "Depth_Data.csv"
SAMPLE_PERIOD_S = 1  # sensor is sampled once per second


# ----------------------------------------------------------------------
# 1. GRAB THE DATA
# ----------------------------------------------------------------------
def load_depth_data(path: str) -> pd.DataFrame:
    """Read the CSV and coerce the depth column to numeric.

    Some rows contain garbage instead of a number (e.g. a literal
    '#VALUE!' string, presumably from an upstream spreadsheet error).
    errors='coerce' turns anything unparsable into NaN so we can find
    and deal with it explicitly instead of crashing on it.
    """
    df = pd.read_csv(path)
    df["time_s"] = (df["Point"] - df["Point"].min()) * SAMPLE_PERIOD_S
    df["depth_raw"] = pd.to_numeric(df["Depth (m)"], errors="coerce")
    return df


# ----------------------------------------------------------------------
# 2. FIND & REPAIR CORRUPTED / ERRATIC READINGS
# ----------------------------------------------------------------------
def clean_depth_data(df: pd.DataFrame, window: int = 11, z_thresh: float = 5.0) -> pd.DataFrame:
    """Flag and interpolate over corrupted readings.

    We can't use a plain mean/std z-score because a single huge spike
    drags the mean and std along with it, hiding itself. Instead:
      - a rolling median gives the *expected* depth at each second
        (robust to the spikes themselves, unlike a rolling mean), and
      - the Median Absolute Deviation (MAD) of the residuals, taken
        over the WHOLE run rather than a small local window, gives a
        stable noise scale (a small local window under-estimates MAD
        during quiet stretches and starts flagging ordinary jitter).
    Anything (a) unparsable, or (b) too many robust-sigmas from its
    expected value, is treated as a dropout and linearly interpolated
    from the surrounding good points.
    """
    depth = df["depth_raw"].copy()

    rolling_med = depth.rolling(window, center=True, min_periods=1).median()
    resid = depth - rolling_med
    global_mad = np.median(np.abs(resid.dropna() - resid.dropna().median()))
    robust_z = 0.6745 * resid / global_mad

    is_corrupted = robust_z.abs() > z_thresh
    is_corrupted = is_corrupted.fillna(False) | depth.isna()

    cleaned = depth.copy()
    cleaned[is_corrupted] = np.nan
    cleaned = cleaned.interpolate(method="linear", limit_direction="both")

    df["depth_clean"] = cleaned
    df["is_corrupted"] = is_corrupted
    return df


# ----------------------------------------------------------------------
# 3. SMOOTH OUT RANDOM SENSOR NOISE
# ----------------------------------------------------------------------
def smooth_depth(df: pd.DataFrame, window: int = 9, poly: int = 2) -> pd.DataFrame:
    """Savitzky-Golay filter: smooths noise while preserving the shape
    of the seafloor trend far better than a plain moving average does.
    """
    w = window if window % 2 == 1 else window + 1
    w = min(w, len(df) - (1 - len(df) % 2))  # must be odd and <= len(df)
    df["depth_smooth"] = savgol_filter(df["depth_clean"], window_length=w, polyorder=poly)
    return df


# ----------------------------------------------------------------------
# 4. ANIMATE: one new point every second
# ----------------------------------------------------------------------
def animate_depth(df: pd.DataFrame, live: bool = True, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_title("Odysseus's Ship -- Depth Below Sea Surface Over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(df["time_s"].min(), df["time_s"].max())
    pad = 0.1 * (df["depth_raw"].max() - df["depth_clean"].min())
    ax.set_ylim(df["depth_clean"].min() - pad, max(0, df["depth_clean"].max()) + pad)
    ax.grid(alpha=0.3)

    raw_line, = ax.plot([], [], color="tab:gray", lw=1, alpha=0.5, label="Raw reading")
    smooth_line, = ax.plot([], [], color="tab:blue", lw=2, label="Smoothed depth (noise reduced)")
    corrupt_pts = ax.scatter([], [], color="tab:red", zorder=5, s=40, label="Corrupted / repaired reading")

    ax.legend(loc="lower left")

    def init():
        raw_line.set_data([], [])
        smooth_line.set_data([], [])
        corrupt_pts.set_offsets(np.empty((0, 2)))
        return raw_line, smooth_line, corrupt_pts

    def update(frame):
        i = frame + 1
        sub = df.iloc[:i]
        raw_line.set_data(sub["time_s"], sub["depth_raw"])
        smooth_line.set_data(sub["time_s"], sub["depth_smooth"])

        flagged = sub[sub["is_corrupted"]]
        if len(flagged):
            corrupt_pts.set_offsets(np.c_[flagged["time_s"], flagged["depth_clean"]])
        return raw_line, smooth_line, corrupt_pts

    interval_ms = 1000 if live else 30  # 1 new point/sec when actually presenting live
    anim = animation.FuncAnimation(
        fig, update, frames=len(df), init_func=init,
        interval=interval_ms, blit=True, repeat=False,
    )

    if save_path:
        writer = "pillow" if save_path.endswith(".gif") else "ffmpeg"
        anim.save(save_path, writer=writer, fps=1000 / interval_ms)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return anim


if __name__ == "__main__":
    data = load_depth_data(CSV_PATH)
    data = clean_depth_data(data)
    data = smooth_depth(data)

    n_bad = int(data["is_corrupted"].sum())
    print(f"Loaded {len(data)} samples, flagged & repaired {n_bad} corrupted reading(s).")

    # live=True -> one new point per second, exactly as the task asks.
    # Change save_path below (e.g. "depth_animation.gif") to export instead
    # of showing an interactive window, if you need a file for your README.
    animate_depth(data, live=True, save_path=None)
