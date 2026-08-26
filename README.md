# SEDS-Avionics-
Name: M Karthick
ID: 2026A7PS0303H
Task 1: Finding the Sea Floor
Files: `depth_monitor.py`, `Depth_Data.csv`
Approach
Loading the data — `Depth_Data.csv` is read with pandas. The `Depth (m)`
column comes in as text in the raw file (one entry contains the literal
string `#VALUE!`), so it's converted with `pd.to_numeric(..., errors="coerce")`,
which turns any unparsable entry into `NaN` instead of crashing the script.
Detecting corrupted / erratic readings — A rolling median (11-second
window) gives an expected depth at each timestamp; a rolling median is far
less sensitive to outliers than a rolling mean would be. The residual
(actual − expected) at every point is compared against the Median Absolute
Deviation (MAD) of all residuals across the whole run, converted to a
"robust z-score" using the standard `0.6745` normal-distribution scaling
constant. Points whose robust z-score magnitude exceeds 5, or that failed
to parse as a number at all, are flagged as corrupted. This caught 3 points
in the sample data:
Point 97 — `#VALUE!` (unparsable)
Point 151 — a `-1271.1` spike against neighbours around `-270`
Point 276 — an anomalous `0.0` reading against neighbours around `-150`
Flagged points are blanked out and linearly interpolated from their
nearest good neighbours, so the timeline stays continuous.
Reducing random noise (brownie points) — A Savitzky–Golay filter
(9-point window, 2nd-degree polynomial) is applied on top of the cleaned
data. Unlike a plain moving average, it smooths point-to-point jitter
while still tracking genuine changes in the seafloor's shape.
Animating the graph — `matplotlib.animation.FuncAnimation` redraws
the plot once per second (`interval=1000`), revealing one additional data
point each frame: raw readings in gray, the smoothed trend in blue, and
any repaired points marked in red. Axes, title, and legend are labelled
throughout.
