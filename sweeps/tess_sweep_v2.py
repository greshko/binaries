#!/usr/bin/env python
"""
tess_sweep.py - sweep TESS stars for light curves worth inspecting by eye.

For every star in a target list the script

  1. asks MAST which TESS light curves exist (via Lightkurve),
  2. notes whether the star has full-frame-image (FFI) light curves
     (authors QLP or TESS-SPOC) and whether it also has 2-minute SPOC data,
  3. downloads a few sectors, flattens them, and looks for isolated
     transit-like dips (the kind a periodic search such as BLS misses),
  4. vets every dip it found:
       a. shared-time veto - a dip that shows up at the same moment in
          several well-separated stars comes from the spacecraft or the
          pipeline, not from any one star, so it is rejected;
       b. centroid check - for each dip that survives, a small cutout of the
          images is downloaded (TESScut) and the star's centre of light is
          tracked. If the image shifted during the dip, the "dip" is image
          motion or light from a different star, and it is rejected;
       c. asteroid check - the same cutout is searched for a bright spot
          that moves in a straight line during the dip. An asteroid drifting
          past a star spoils the sky-brightness estimate and carves a fake
          dip into the light curve, so such dips are rejected,
  5. writes ranked tables and diagnostic plots.

A star is flagged when it is "FFI-only" (FFI light curves but no 2-minute
data, so it has had less scrutiny) and/or when 1..--max-dips dips survive
vetting.

Usage
-----
Sweep a patch of sky (decimal degrees):

    python tess_sweep.py --ra 270.0 --dec 66.56 --radius 0.5 --max-tmag 13

Sweep your own list (one TIC ID per line, "TIC 12345" or "12345"):

    python tess_sweep.py --targets my_stars.txt

Output (in --outdir, default "sweep_output")
-------------------------------------------
    summary.csv                one row per star, best candidates first
    events.csv                 every dip found, with the verdict it was given
    plots/dip_TIC*.png         stars with surviving dips (red = candidate,
                               grey = rejected)
    plots/ffi_TIC*.png         FFI-only stars with no surviving dip
    plots/centroid_TIC*.png    centre-of-light evidence for each dip checked
    plots/frames_TIC*.png      five frames around each dip checked, with any
                               moving object's track drawn in

Verdicts in events.csv
----------------------
    shared      same time as dips in other, well-separated stars -> artifact
    moved       the star's image shifted during the dip -> motion or a blend
    asteroid    a moving object crossed the cutout during the dip
    candidate   passed all three checks
    unchecked   passed the shared-time veto; pixels could not be checked
    variable    the star has too many surviving dips to be a single transit

Caveats
-------
Reported depths are lower limits: flattening absorbs part of a long dip, and
the trial durations are coarse. Read the depth off the plot, not the table.

The centroid check cannot tell spacecraft jitter from a dip on a neighbouring
star: both move the centre of light, and both are rejected. A real transit on
a star with a bright close neighbour can be rejected for the same reason, so
glance at events.csv before discarding a "moved" dip you care about.

The asteroid check only sees objects bright enough to stand out in single
images (about TESS magnitude 17 or brighter) and that cross the 21-pixel
cutout while the dip is happening. Those are the ones that cause dips.

The shared-time veto needs company. With only a handful of stars in a sweep
there is little to compare against, so more dips will reach the centroid check.

A surviving dip is still only a prompt to look. Check ExoFOP and the other
sectors before getting excited.

Light curves are held in memory until the end (about 1 MB per star at six
sectors), so sweep a few hundred stars at a time rather than thousands.
"""

import argparse
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write PNGs without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lightkurve as lk

# ----------------------------------------------------------------------------
# Settings you are most likely to tune
# ----------------------------------------------------------------------------
FFI_AUTHORS = ("TESS-SPOC", "QLP")   # FFI pipelines, in order of preference
TWO_MIN_AUTHOR = "SPOC"              # the 2-minute / 20-second pipeline
DURATIONS_HR = (2.0, 4.0, 8.0, 13.0)  # trial dip durations
GAP_DAYS = 0.5           # a break longer than this counts as a data gap
EDGE_BUFFER_DAYS = 0.5   # ignore dips this close to a gap or sector edge
MERGE_DAYS = 0.75        # detections closer than this are one event
MIN_POINTS = 200         # skip light curves shorter than this
MAX_ERRORS_IN_A_ROW = 5  # give up if this many stars fail back to back

# Shared-time veto
SHARED_TOL_DAYS = 0.15   # dips closer in time than this count as simultaneous
NEIGHBOUR_ARCMIN = 3.0   # stars closer than this share pixels, so a shared dip
                         # between them is a blend, not a spacecraft artifact
WITNESS_MAX_DIPS = 15    # stars with more dips than this are too variable to
                         # be used as evidence against another star's dip

# Centroid check
CUTOUT_SIZE = 21         # pixels on a side for the image cutout
CENTROID_HALF_BOX = 2    # centre of light is measured in a 5x5 pixel box

# Asteroid check
ASTEROID_SNR = 6.0         # how far a spot must stand out in one (binned) frame
ASTEROID_BIN_MIN = 15.0    # frames are averaged to about this many minutes
ASTEROID_PAD_DAYS = 0.15   # search this long before and after the dip as well
ASTEROID_MIN_TRAVEL = 2.5  # pixels a spot must travel to count as moving
ASTEROID_CORE_PX = 2.5     # ignore pixels this close to the target itself
PIXEL_ARCSEC = 21.0        # size of a TESS pixel on the sky

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

GOOD_VERDICTS = ("candidate", "unchecked")


# ----------------------------------------------------------------------------
# Target lists
# ----------------------------------------------------------------------------
def targets_from_file(path):
    """Read TIC IDs from a text file, one per line. '#' starts a comment."""
    ids = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        ids.append(int(line.upper().replace("TIC", "").strip()))
    return pd.DataFrame({"tic": ids, "tmag": np.nan, "ra": np.nan, "dec": np.nan})


def targets_from_cone(ra, dec, radius_deg, min_tmag, max_tmag, include_giants):
    """Query the TESS Input Catalog for stars in a circle on the sky."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.mast import Catalogs

    coord = SkyCoord(ra * u.deg, dec * u.deg)
    cat = Catalogs.query_criteria(
        catalog="Tic",
        coordinates=coord,
        radius=radius_deg * u.deg,
        Tmag=[min_tmag, max_tmag],
        objType="STAR",
    )
    if len(cat) == 0:
        return pd.DataFrame(columns=["tic", "tmag", "ra", "dec"])

    df = cat[["ID", "Tmag", "ra", "dec", "lumclass"]].to_pandas()
    if not include_giants:
        # Giants are so large that planet transits are too shallow to see.
        df = df[df["lumclass"].astype(str).str.upper() != "GIANT"]
    df = df.rename(columns={"ID": "tic", "Tmag": "tmag"})
    df["tic"] = df["tic"].astype(int)
    return df.sort_values("tmag")[["tic", "tmag", "ra", "dec"]].reset_index(drop=True)


def load_known_tois():
    """Return the set of TIC IDs that already have a TOI, or None if offline."""
    try:
        toi = pd.read_csv(TOI_URL)
        return set(toi["TIC ID"].astype(int))
    except Exception as exc:  # network trouble, column rename, ...
        print(f"  (could not fetch the TOI list, 'known_toi' left blank: {exc})")
        return None


# ----------------------------------------------------------------------------
# Finding and preparing light curves
# ----------------------------------------------------------------------------
def sector_of(mission):
    """'TESS Sector 07' -> 7"""
    try:
        return int(str(mission).split()[-1])
    except ValueError:
        return -1


def search_star(tic):
    """Search MAST and keep only products that belong to this exact TIC ID."""
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    if len(sr) == 0:
        return sr
    digits = ["".join(ch for ch in str(n) if ch.isdigit()).lstrip("0") for n in sr.target_name]
    return sr[np.array(digits) == str(tic)]


def choose_curves(sr, max_sectors):
    """Pick one light curve per sector, preferring FFI products.

    Returns (indices into sr, has_ffi, has_2min).
    """
    authors = np.array([str(a) for a in sr.author])
    sectors = np.array([sector_of(m) for m in sr.mission])
    exptime = np.array(sr.exptime.value, dtype=float)

    has_ffi = bool(np.isin(authors, FFI_AUTHORS).any())
    has_2min = bool((authors == TWO_MIN_AUTHOR).any())
    preference = FFI_AUTHORS if has_ffi else (TWO_MIN_AUTHOR,)

    chosen = {}
    for author in preference:
        idx = np.where(authors == author)[0]
        # longest exposure first, so 120 s wins over 20 s for the same sector
        for i in idx[np.argsort(-exptime[idx])]:
            chosen.setdefault(sectors[i], i)

    newest_first = sorted(chosen, reverse=True)[:max_sectors]
    return [chosen[s] for s in newest_first], has_ffi, has_2min


def prepare(lc, window_days):
    """Clean and flatten one sector. Returns (time, flux) arrays or None."""
    lc = lc.remove_nans()
    if len(lc) < MIN_POINTS:
        return None
    dt = float(np.nanmedian(np.diff(lc.time.value)))
    window = max(int(round(window_days / dt)) | 1, 5)   # must be odd
    flat = lc.flatten(
        window_length=window,
        polyorder=2,
        break_tolerance=max(int(GAP_DAYS / dt), 5),
    )
    # clip flares and cosmic rays, but never clip downward (that is the signal)
    flat = flat.remove_outliers(sigma_upper=4, sigma_lower=np.inf)

    time = np.asarray(flat.time.value, dtype=float)
    flux = np.asarray(flat.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(flux)
    if good.sum() < MIN_POINTS:
        return None
    return time[good], flux[good]


# ----------------------------------------------------------------------------
# Single-dip detection
# ----------------------------------------------------------------------------
def rolling_mean(values, n):
    c = np.concatenate(([0.0], np.cumsum(values)))
    return (c[n:] - c[:-n]) / n


def find_dips(time, flux, snr_min=8.0, durations_hr=DURATIONS_HR):
    """Find isolated box-shaped dips in a flattened light curve.

    For each trial duration the mean depth in a sliding window is compared
    with the scatter of that same sliding mean across the whole sector, so
    slow wiggles that survive flattening raise the bar automatically.

    Returns a list of dicts: time (BTJD), snr, depth_ppt, duration_hr.
    """
    npts = len(time)
    dt = float(np.median(np.diff(time)))
    depth = 1.0 - flux / np.median(flux)

    best_snr = np.zeros(npts)
    best_depth = np.zeros(npts)
    best_dur = np.zeros(npts)

    for dur in durations_hr:
        n = int(round(dur / 24.0 / dt))
        if n < 3 or n > npts // 4:
            continue
        mean = rolling_mean(depth, n)
        span = time[n - 1:] - time[: npts - n + 1]
        ok = span <= 1.5 * n * dt            # window must not straddle a gap
        if ok.sum() < 20:
            continue
        scatter = 1.4826 * np.median(np.abs(mean[ok] - np.median(mean[ok])))
        if not np.isfinite(scatter) or scatter <= 0:
            continue
        snr = np.where(ok, mean / scatter, 0.0)
        centre = np.arange(len(mean)) + n // 2
        better = snr > best_snr[centre]
        best_snr[centre[better]] = snr[better]
        best_depth[centre[better]] = mean[better]
        best_dur[centre[better]] = dur

    # Discard anything hugging a gap or the ends of the sector.
    gaps = np.where(np.diff(time) > GAP_DAYS)[0]
    edges = np.concatenate(([time[0], time[-1]], time[gaps], time[gaps + 1]))
    near_edge = (np.abs(time[:, None] - edges[None, :]) < EDGE_BUFFER_DAYS).any(axis=1)
    best_snr[near_edge] = 0.0

    hits = np.where(best_snr >= snr_min)[0]
    events = []
    if len(hits):
        breaks = np.where(np.diff(time[hits]) > MERGE_DAYS)[0] + 1
        for group in np.split(hits, breaks):
            k = group[np.argmax(best_snr[group])]
            events.append(
                {
                    "time": float(time[k]),
                    "snr": float(best_snr[k]),
                    "depth_ppt": float(best_depth[k] * 1e3),
                    "duration_hr": float(best_dur[k]),
                }
            )
    return events


# ----------------------------------------------------------------------------
# Vetting step 1: the shared-time veto
# ----------------------------------------------------------------------------
def separation_arcmin(a, b):
    """Angle between two stars. Unknown coordinates count as far apart."""
    vals = np.array([a["ra"], a["dec"], b["ra"], b["dec"]], dtype=float)
    if not np.isfinite(vals).all():
        return np.inf
    ra1, dec1, ra2, dec2 = np.radians(vals)
    h = np.sin((dec2 - dec1) / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2) ** 2
    return float(np.degrees(2 * np.arcsin(np.sqrt(h))) * 60.0)


def mark_shared(stars, shared_min):
    """Reject dips that happen at the same time in unrelated stars.

    For each dip, count the other stars that dipped within SHARED_TOL_DAYS.
    Stars closer than NEIGHBOUR_ARCMIN to the target are not counted (they
    share pixels with it, so of course they share its dips) and are listed in
    the event's 'neighbours' field instead. Witnesses that sit close to one
    another are counted once. If the count reaches shared_min - 1, the dip was
    seen in at least shared_min separate places and gets the verdict 'shared'.
    """
    pairs = [(s, e) for s in stars for e in s["events"]]
    if not pairs:
        return
    times = np.array([e["time"] for _, e in pairs])
    eligible = np.array([len(s["events"]) <= WITNESS_MAX_DIPS for s, _ in pairs])

    for star, event in pairs:
        close = np.where(np.abs(times - event["time"]) <= SHARED_TOL_DAYS)[0]
        far, near = [], []
        for k in close:
            other = pairs[k][0]
            if other is star:
                continue
            if separation_arcmin(star, other) <= NEIGHBOUR_ARCMIN:
                if other["tic"] not in near:
                    near.append(other["tic"])
            elif eligible[k] and all(
                separation_arcmin(other, kept) > NEIGHBOUR_ARCMIN for kept in far
            ):
                far.append(other)
        event["n_shared"] = len(far)
        event["neighbours"] = "|".join(str(t) for t in near)
        if len(far) >= shared_min - 1:
            event["verdict"] = "shared"


# ----------------------------------------------------------------------------
# Vetting step 2: the centroid check
# ----------------------------------------------------------------------------
def fetch_cutout(tic, sector):
    """Download a small cutout of the images around a star.

    Returns (time, cube) with cube shaped (frames, rows, columns), or None.
    """
    sr = lk.search_tesscut(f"TIC {tic}", sector=sector)
    if len(sr) == 0:
        return None
    tpf = sr.download(cutout_size=CUTOUT_SIZE)
    if tpf is None:
        return None
    time = np.asarray(tpf.time.value, dtype=float)
    cube = np.asarray(tpf.flux.value, dtype=float)
    good = np.isfinite(time) & np.isfinite(cube).all(axis=(1, 2))
    if good.sum() < MIN_POINTS:
        return None
    return time[good], cube[good]


def track_centroid(cube):
    """Subtract the sky from a cutout and follow the target's centre of light.

    Returns (col, row, sky-subtracted cube, (target_x, target_y)).
    """
    _, ny, nx = cube.shape
    typical = np.median(cube, axis=0)
    # Sky level in each frame = median of the faintest 30% of pixels.
    sky_pixels = typical <= np.percentile(typical, 30)
    sky = np.median(cube[:, sky_pixels], axis=1)
    sub = cube - sky[:, None, None]

    # The target sits in the middle of the cutout; take the brightest pixel
    # of the central 3x3 as its position.
    cy, cx = ny // 2, nx // 2
    core = typical[cy - 1: cy + 2, cx - 1: cx + 2]
    j, i = np.unravel_index(np.argmax(core), core.shape)
    ty, tx = cy - 1 + j, cx - 1 + i

    b = CENTROID_HALF_BOX
    y0, y1 = max(ty - b, 0), min(ty + b + 1, ny)
    x0, x1 = max(tx - b, 0), min(tx + b + 1, nx)
    box = sub[:, y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    total = box.sum(axis=(1, 2))
    total = np.where(total > 0, total, np.nan)
    col = (box * xx).sum(axis=(1, 2)) / total
    row = (box * yy).sum(axis=(1, 2)) / total
    return col, row, sub, (tx, ty)


def tame_outliers(values, n_mad=10.0):
    """Pull single wild points (cosmic rays, thruster firings) back in."""
    med = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - med))
    if not np.isfinite(mad) or mad <= 0:
        return values
    return np.clip(values, med - n_mad * mad, med + n_mad * mad)


def window_contrast(time, values, t0, dur_days):
    """How unusual is the change in `values` during a dip?

    Compares the mean inside the dip with the mean just outside it, then
    makes the same comparison at every other position in the sector to learn
    how large such changes normally are.

    Returns (change, significance in sigma), or (nan, nan) if it cannot tell.
    """
    n = len(values)
    if n < MIN_POINTS:
        return np.nan, np.nan
    dt = float(np.median(np.diff(time)))
    n_in = max(int(round(dur_days / dt)), 3)
    n_gap = max(n_in // 4, 1)                       # skip ingress and egress
    n_out = max(n_in, int(round(0.5 / dt)))         # flank on each side
    c = np.concatenate(([0.0], np.cumsum(values)))
    i = np.arange(n)
    a = i - n_in // 2          # the dip occupies samples [a, b)
    b = a + n_in

    def mean(lo, hi):
        lo_c, hi_c = np.clip(lo, 0, n), np.clip(hi, 0, n)
        return (c[hi_c] - c[lo_c]) / np.maximum(hi_c - lo_c, 1)

    def unbroken(lo, hi):      # inside the data and not straddling a gap
        ok = (lo >= 0) & (hi <= n)
        lo_c, hi_c = np.clip(lo, 0, n - 1), np.clip(hi - 1, 0, n - 1)
        return ok & ((time[hi_c] - time[lo_c]) <= 1.5 * (hi - lo) * dt)

    inside = mean(a, b)
    left = mean(a - n_gap - n_out, a - n_gap)
    right = mean(b + n_gap, b + n_gap + n_out)
    # Prefer flanks on both sides; fall back to one side next to a data gap.
    variants = (
        (inside - 0.5 * (left + right), unbroken(a - n_gap - n_out, b + n_gap + n_out)),
        (inside - left, unbroken(a - n_gap - n_out, b)),
        (inside - right, unbroken(a, b + n_gap + n_out)),
    )
    k = int(np.argmin(np.abs(time - t0)))
    elsewhere = np.abs(time - t0) > (n_in + n_gap + n_out) * dt
    for stat, ok in variants:
        if not ok[k]:
            continue
        usual = stat[ok & elsewhere]
        if len(usual) < 50:
            continue
        centre = np.median(usual)
        scatter = 1.4826 * np.median(np.abs(usual - centre))
        if not np.isfinite(scatter) or scatter <= 0:
            continue
        change = float(stat[k] - centre)
        return change, abs(change) / scatter
    return np.nan, np.nan


def centroid_check(time, cube, t0, dur_hr):
    """Did the star's image move during the dip?

    Returns a dict with:
      sigma      how unusual the shift is for this sector (larger = worse)
      shift_mpx  size of the shift in thousandths of a pixel
      plus the series and difference image needed for the diagnostic plot.
    """
    col, row, sub, target = track_centroid(cube)
    ok = np.isfinite(col) & np.isfinite(row)
    t, col, row = time[ok], col[ok], row[ok]
    dur = max(dur_hr, 2.0) / 24.0

    if len(t) >= MIN_POINTS:
        dcol, scol = window_contrast(t, tame_outliers(col), t0, dur)
        drow, srow = window_contrast(t, tame_outliers(row), t0, dur)
    else:
        dcol = scol = drow = srow = np.nan
    sig = np.array([scol, srow], dtype=float)
    sigma = float(np.nanmax(sig)) if np.isfinite(sig).any() else np.nan
    shift = float(np.hypot(np.nan_to_num(dcol), np.nan_to_num(drow)) * 1000.0)

    # Light lost per pixel: average frame just outside the dip minus inside.
    away = np.abs(time - t0)
    inside = away < dur / 2
    outside = (away > 0.75 * dur) & (away < 0.75 * dur + max(dur, 0.5))
    diff = None
    if inside.sum() >= 3 and outside.sum() >= 3:
        diff = sub[outside].mean(axis=0) - sub[inside].mean(axis=0)

    return {
        "sigma": sigma, "shift_mpx": shift if np.isfinite(sigma) else np.nan,
        "time": t, "col": col, "row": row, "diff": diff,
        "image": np.median(sub, axis=0), "target": target, "dur": dur,
    }


def plot_centroid(title, event, res, threshold, path):
    """Pixel evidence for one dip: where light was lost, and whether the image moved."""
    fig = plt.figure(figsize=(13, 5))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5])
    ax_img = fig.add_subplot(grid[:, 0])
    ax_diff = fig.add_subplot(grid[:, 1])
    ax_col = fig.add_subplot(grid[0, 2])
    ax_row = fig.add_subplot(grid[1, 2], sharex=ax_col)
    tx, ty = res["target"]

    ax_img.imshow(res["image"], origin="lower", cmap="gray_r")
    ax_img.set_title("Average image", fontsize=10)
    if res["diff"] is not None:
        b = CENTROID_HALF_BOX + 1
        core = res["diff"][max(ty - b, 0): ty + b + 1, max(tx - b, 0): tx + b + 1]
        lim = float(np.max(np.abs(core))) or 1.0
        ax_diff.imshow(res["diff"], origin="lower", cmap="RdBu", vmin=-lim, vmax=lim)
    ax_diff.set_title("Light lost during the dip\n(blue = lost, red = gained)", fontsize=10)
    for ax in (ax_img, ax_diff):
        ax.plot(tx, ty, "+", color="k", ms=12)
        ax.set_xlabel("Column (pixels)")
    ax_img.set_ylabel("Row (pixels)")

    t0, dur = event["time"], res["dur"]
    half = max(3 * dur, 1.5)
    near = np.abs(res["time"] - t0) < half
    for ax, series, name in ((ax_col, res["col"], "Column"), (ax_row, res["row"], "Row")):
        t, y = res["time"][near], series[near]
        if len(t):
            y = (y - np.median(y)) * 1000.0
            ax.plot(t, y, ".", ms=2, color="0.6", rasterized=True)
            tb, yb = bin_xy(t, y, 2.0 / 24.0)
            ax.plot(tb, yb, ".", ms=5, color="k")
            spread = 1.4826 * np.median(np.abs(yb - np.median(yb))) if len(yb) > 3 else 0.0
            reach = max(6 * spread, 1.3 * float(np.max(np.abs(yb))) if len(yb) else 0.0, 1e-3)
            ax.set_ylim(-reach, reach)
        ax.axvspan(t0 - dur / 2, t0 + dur / 2, color="crimson", alpha=0.15)
        ax.set_ylabel(f"{name} shift\n(1/1000 pixel)")
    ax_col.set_title("Centre of light (shaded = the dip)", fontsize=10)
    ax_row.set_xlabel("Time (BTJD days)")
    plt.setp(ax_col.get_xticklabels(), visible=False)

    moved = res["sigma"] >= threshold
    verdict = "IMAGE MOVED - rejected" if moved else "image steady"
    fig.suptitle(
        f"{title}   sector {event['sector']}, BTJD {t0:.2f}   |   {verdict}: "
        f"{res['sigma']:.1f} sigma (limit {threshold:g}), shift {res['shift_mpx']:.1f}/1000 pixel",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Vetting step 3: the asteroid check
# ----------------------------------------------------------------------------
def bin_frames(time, stack, n):
    """Average consecutive frames in groups of n (leftover frames are dropped)."""
    if n <= 1:
        return time, stack
    m = (len(time) // n) * n
    if m == 0:
        return time[:0], stack[:0]
    return (
        time[:m].reshape(-1, n).mean(axis=1),
        stack[:m].reshape(-1, n, *stack.shape[1:]).mean(axis=1),
    )


def remove_plane(frames):
    """Take a tilted plane out of each frame (slow scattered-light gradients)."""
    _, ny, nx = frames.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    design = np.column_stack([np.ones(ny * nx), xx.ravel() - nx / 2, yy.ravel() - ny / 2])
    flat = frames.reshape(len(frames), -1)
    coeff, *_ = np.linalg.lstsq(design, flat.T, rcond=None)
    return (flat - (design @ coeff).T).reshape(frames.shape)


def find_spots(zmap, noise, keep, max_spots=3):
    """Brightest compact spots in one frame: list of (x, y, significance)."""
    from scipy.ndimage import uniform_filter

    z = np.where(keep, zmap, 0.0)
    score = uniform_filter(z, size=3, mode="constant") * 3.0   # 3x3 sum, unit variance
    ny, nx = z.shape
    spots = []
    for _ in range(max_spots):
        j, i = np.unravel_index(np.argmax(score), score.shape)
        if score[j, i] < ASTEROID_SNR:
            break
        y0, y1, x0, x1 = max(j - 2, 0), min(j + 3, ny), max(i - 2, 0), min(i + 3, nx)
        w = np.clip(z[y0:y1, x0:x1] * noise[y0:y1, x0:x1], 0, None)   # back to flux
        yy, xx = np.mgrid[y0:y1, x0:x1]
        if w.sum() > 0:
            spots.append((float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum()),
                          float(score[j, i])))
        score[y0:y1, x0:x1] = 0.0                                # look elsewhere next
    return spots


def find_track(spots, frame_times):
    """Look for spots that line up along a straight path in time.

    `spots` is a list of (frame index, x, y). Returns None, or a dict with the
    fitted straight-line motion and the frames that lie on it.
    """
    if len(spots) < 4:
        return None
    k = np.array([s[0] for s in spots])
    t = frame_times[k] * 24.0                                    # hours
    x = np.array([s[1] for s in spots])
    y = np.array([s[2] for s in spots])

    def on_track(vx, vy, x0, y0, tref):
        miss = np.hypot(x - (x0 + vx * (t - tref)), y - (y0 + vy * (t - tref)))
        close = miss <= 1.2
        frames = {}
        for idx in np.where(close)[0]:                           # one spot per frame
            if k[idx] not in frames or miss[idx] < miss[frames[k[idx]]]:
                frames[k[idx]] = idx
        return np.array(sorted(frames.values()), dtype=int)

    best = np.array([], dtype=int)
    for a in range(len(spots)):
        for b in range(a + 1, len(spots)):
            gap = t[b] - t[a]
            if k[a] == k[b] or abs(gap) < 0.4:
                continue
            members = on_track((x[b] - x[a]) / gap, (y[b] - y[a]) / gap, x[a], y[a], t[a])
            if len(members) > len(best):
                best = members
    if len(best) < 4:
        return None

    for _ in range(2):                                           # refit, then re-collect
        tref = t[best].mean()
        vx, x0 = np.polyfit(t[best] - tref, x[best], 1)
        vy, y0 = np.polyfit(t[best] - tref, y[best], 1)
        best = on_track(vx, vy, x0, y0, tref)
        if len(best) < 4:
            return None
    span = t[best].max() - t[best].min()
    speed = float(np.hypot(vx, vy))
    frames_spanned = k[best].max() - k[best].min() + 1
    if speed * span < ASTEROID_MIN_TRAVEL or len(best) < 0.5 * frames_spanned:
        return None

    # The spot must be seen moving all along the track. Two unrelated stars
    # that each brighten for a while can sit on one line, but neither moves.
    lag = min(max(1.0, 1.0 / speed), span / 2)                   # hours
    tb = t[best]
    along = (x[best] * vx + y[best] * vy) / speed                # position along the track
    local = []
    for i in range(len(tb)):
        gaps = tb - tb[i]
        j = int(np.argmin(np.abs(gaps - lag)))
        if 0.5 * lag <= gaps[j] <= 1.5 * lag:
            local.append((along[j] - along[i]) / gaps[j])
    if len(local) < 3 or np.median(local) < 0.6 * speed:
        return None
    return {"vx": vx, "vy": vy, "x0": x0, "y0": y0, "tref": tref, "speed": speed,
            "n_frames": int(len(best)), "x": x[best], "y": y[best], "t": t[best]}


def asteroid_check(time, cube, t0, dur_hr):
    """Did a moving object cross the cutout during the dip?

    Each frame near the dip has a reference image (the median of frames well
    away from the dip) subtracted from it. Every difference frame is searched
    for compact bright spots away from the target, and the spots are tested
    for lining up along a straight track in time.

    Returns a dict with:
      checked   False if there was too little data to tell
      found     True if a moving object was seen
      speed     its speed in arcseconds per hour
      closest   how near its track passes to the target, in pixels
      plus the frames and track needed for the diagnostic plot.
    """
    out = {"checked": False, "found": False, "speed": np.nan, "closest": np.nan,
           "n_frames": 0, "frames": None, "track": None}
    _, ny, nx = cube.shape
    dur = max(dur_hr, 2.0) / 24.0
    half = dur / 2 + ASTEROID_PAD_DAYS
    away = np.abs(time - t0)
    near = away <= half
    base = (away > half + 0.25) & (away < half + 1.5)

    dt_min = float(np.median(np.diff(time))) * 1440.0
    n_bin = max(int(round(ASTEROID_BIN_MIN / dt_min)), 1)
    if near.sum() < 6 * n_bin or base.sum() < 30 * n_bin:
        return out

    sub = cube - np.median(cube, axis=(1, 2))[:, None, None]     # per-frame sky level
    ref = np.median(sub[base], axis=0)
    _, base_frames = bin_frames(time[base], sub[base] - ref, n_bin)
    t_near, near_frames = bin_frames(time[near], sub[near] - ref, n_bin)
    base_frames, near_frames = remove_plane(base_frames), remove_plane(near_frames)

    noise = 1.4826 * np.median(np.abs(base_frames - np.median(base_frames, axis=0)), axis=0)
    noise = np.maximum(noise, 0.05 * np.median(noise) + 1e-9)
    cy, cx = ny // 2, nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    keep = np.hypot(xx - cx, yy - cy) > ASTEROID_CORE_PX

    spots = []
    for f, frame in enumerate(near_frames):
        spots += [(f, x, y) for x, y, _ in find_spots(frame / noise, noise, keep)]
    track = find_track(spots, t_near)

    out.update(checked=True, frames=near_frames, frame_times=t_near,
               scale=float(np.median(noise)), target=(cx, cy))
    if track is not None:
        hours = t_near * 24.0 - track["tref"]
        path_x = track["x0"] + track["vx"] * hours
        path_y = track["y0"] + track["vy"] * hours
        out.update(
            found=True, track=track, n_frames=track["n_frames"],
            speed=track["speed"] * PIXEL_ARCSEC,
            closest=float(np.min(np.hypot(path_x - cx, path_y - cy))),
            path=(path_x, path_y),
        )
    return out


def plot_frames(title, event, res, path):
    """Five frames around a dip, with the usual scene subtracted."""
    frames, times = res["frames"], res["frame_times"]
    picks = np.unique(np.linspace(0, len(frames) - 1, 5).round().astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(3.3 * len(picks), 4.4),
                             squeeze=False, sharey=True)
    ticks = np.arange(0, frames.shape[1], 5)
    lim = 8.0 * res["scale"]
    tx, ty = res["target"]
    for ax, f in zip(axes[0], picks):
        ax.imshow(frames[f], origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.plot(tx, ty, "+", color="k", ms=12)
        if res["found"]:
            ax.plot(*res["path"], "-", color="k", lw=0.8, alpha=0.6)
            ax.plot(res["path"][0][f], res["path"][1][f], "o", ms=14, mfc="none", mec="k")
        ax.set_title(f"{(times[f] - event['time']) * 24:+.1f} h", fontsize=10)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlabel("Column (pixels)")
    axes[0][0].set_ylabel("Row (pixels)")
    if res["found"]:
        verdict = (f"MOVING OBJECT - rejected: {res['speed']:.0f} arcsec/hour, "
                   f"passes {res['closest']:.1f} pixels from the star")
    else:
        verdict = "no moving object found"
    fig.suptitle(
        f"{title}   sector {event['sector']}, BTJD {event['time']:.2f}   |   {verdict}\n"
        "Red = brighter than usual, blue = fainter. A fixed red/blue pair is a star and "
        "slow pointing drift; an asteroid is a red spot that changes position.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def vet_pixels(stars, args, plot_dir):
    """Run the centroid and asteroid checks on every dip that survived the shared-time veto."""
    todo = []
    for star in stars:
        alive = [e for e in star["events"] if e["verdict"] == ""]
        if len(alive) > args.max_dips:
            for e in alive:
                e["verdict"] = "variable"
        elif alive:
            todo.append((star, alive))

    if args.no_centroid:
        for _, alive in todo:
            for e in alive:
                e["verdict"] = "unchecked"
        return
    if not todo:
        return

    n_dips = sum(len(alive) for _, alive in todo)
    print(f"\nPixel checks: {n_dips} dip(s) in {len(todo)} star(s) survived the shared-time veto.")
    try:
        for star, alive in todo:
            by_sector = {}
            for e in alive:
                by_sector.setdefault(e["sector"], []).append(e)
            for sector, events in sorted(by_sector.items()):
                print(f"  TIC {star['tic']} sector {sector}: ", end="", flush=True)
                try:
                    cutout = fetch_cutout(star["tic"], sector)
                except Exception as exc:
                    cutout = None
                    print(f"cutout failed ({str(exc)[:80]}) ", end="")
                notes = []
                for e in events:
                    if cutout is None:
                        e["verdict"] = "unchecked"
                        notes.append("could not check")
                        continue
                    tag = f"TIC{star['tic']}_S{sector}_{e['time']:.2f}.png"
                    cen = centroid_check(*cutout, e["time"], e["duration_hr"])
                    ast = asteroid_check(*cutout, e["time"], e["duration_hr"])
                    cen_ok = np.isfinite(cen["sigma"])
                    said = []
                    if cen_ok:
                        e["centroid_sigma"] = round(cen["sigma"], 1)
                        e["centroid_shift_mpx"] = round(cen["shift_mpx"], 2)
                        moved = cen["sigma"] >= args.centroid_sigma
                        said.append(f"{'image moved' if moved else 'steady'} ({cen['sigma']:.1f} sigma)")
                    if ast["found"]:
                        e["asteroid_speed"] = round(ast["speed"], 1)
                        e["asteroid_dist_px"] = round(ast["closest"], 1)
                        said.append(f"moving object ({ast['speed']:.0f} arcsec/h, "
                                    f"{ast['closest']:.1f} px away)")
                    elif ast["checked"]:
                        said.append("no moving object")

                    if cen_ok and moved:
                        e["verdict"] = "moved"
                    elif ast["found"]:
                        e["verdict"] = "asteroid"
                    elif cen_ok and ast["checked"]:
                        e["verdict"] = "candidate"
                    else:
                        e["verdict"] = "unchecked"
                        said.append("could not finish the checks")

                    if cen_ok:
                        path = plot_dir / f"centroid_{tag}"
                        plot_centroid(f"TIC {star['tic']}", e, cen, args.centroid_sigma, path)
                        e["centroid_plot"] = str(path)
                    if ast["checked"]:
                        path = plot_dir / f"frames_{tag}"
                        plot_frames(f"TIC {star['tic']}", e, ast, path)
                        e["frames_plot"] = str(path)
                    notes.append(", ".join(said))
                print("; ".join(notes))
    except KeyboardInterrupt:
        print("\nInterrupted - remaining dips left unchecked.")
    for _, alive in todo:
        for e in alive:
            if e["verdict"] == "":
                e["verdict"] = "unchecked"


# ----------------------------------------------------------------------------
# Plotting light curves
# ----------------------------------------------------------------------------
def bin_xy(time, flux, width_days):
    key = np.floor((time - time[0]) / width_days)
    df = pd.DataFrame({"k": key, "t": time, "f": flux}).groupby("k").mean()
    return df["t"].to_numpy(), df["f"].to_numpy()


def make_plot(title, sectors, path):
    """One row per sector: whole light curve on the left, zoom on the right.

    Red arrowheads mark dips that passed vetting, grey ones dips that were
    rejected. The zoom shows the strongest surviving dip in that sector.
    """
    nrows = len(sectors)
    fig, axes = plt.subplots(
        nrows, 2, figsize=(13, 3.2 * nrows), squeeze=False,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    for (ax_full, ax_zoom), s in zip(axes, sectors):
        t, f = s["time"], np.asarray(s["flux"], dtype=float)
        good = [e for e in s["events"] if e["verdict"] in GOOD_VERDICTS]
        f = f / np.median(f)
        sigma = 1.4826 * np.median(np.abs(f - 1.0))
        deepest = max([e["depth_ppt"] / 1e3 for e in good], default=0.0)
        ylim = (1 - max(6 * sigma, 1.8 * deepest), 1 + 5 * sigma)
        tb, fb = bin_xy(t, f, 2.0 / 24.0)

        for ax in (ax_full, ax_zoom):
            ax.plot(t, f, ".", ms=2, color="0.6", rasterized=True)
            ax.plot(tb, fb, ".", ms=4, color="k")
            ax.set_ylim(*ylim)
        for e in s["events"]:  # arrowhead above the dip, so the data stay visible
            passed = e["verdict"] in GOOD_VERDICTS
            ax_full.plot(
                e["time"], ylim[1], marker="v", ms=10, clip_on=False, zorder=5,
                color="crimson" if passed else "0.6",
                markerfacecolor="crimson" if passed else "none",
            )

        ax_full.set_ylabel("Normalised flux")
        ax_full.set_title(f"Sector {s['sector']}  ({s['author']})", loc="left", fontsize=10)

        if good:
            e = max(good, key=lambda ev: ev["snr"])
            ax_zoom.set_xlim(e["time"] - 1.0, e["time"] + 1.0)
            ax_zoom.set_title(
                f"SNR {e['snr']:.1f} | {e['depth_ppt']:.2f} ppt | ~{e['duration_hr']:.0f} h",
                fontsize=9,
            )
        else:
            ax_zoom.set_visible(False)

    axes[-1, 0].set_xlabel("Time (BTJD days)")
    if axes[-1, 1].get_visible():
        axes[-1, 1].set_xlabel("Time (BTJD days)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------
# One star: collect, then (after vetting) summarise
# ----------------------------------------------------------------------------
def blank_row(tic, tmag, known_tois):
    """A summary row with every column present and nothing found yet."""
    return {
        "tic": tic, "tmag": tmag, "status": "", "flag": "",
        "known_toi": None if known_tois is None else tic in known_tois,
        "authors": "", "has_ffi": False, "has_2min": False, "ffi_only": False,
        "n_products": 0, "sectors_examined": "",
        "n_dips": 0, "n_shared": 0, "n_moved": 0, "n_asteroid": 0, "n_candidate": 0,
        "best_snr": np.nan, "best_time_btjd": np.nan, "best_sector": np.nan,
        "best_depth_ppt": np.nan, "best_duration_hr": np.nan,
        "centroid_sigma": np.nan, "centroid_shift_mpx": np.nan,
        "neighbour_dips": "", "plot": "", "ra": np.nan, "dec": np.nan,
    }


def new_star(tic, tmag, ra, dec, known_tois):
    row = blank_row(tic, tmag, known_tois)
    return {"tic": tic, "tmag": tmag, "ra": ra, "dec": dec, "row": row, "sectors": [], "events": []}


def collect_star(star, args):
    """Download a star's light curves and find its dips. Fills `star` in place."""
    tic, row = star["tic"], star["row"]

    sr = search_star(tic)
    if len(sr) == 0:
        row["status"] = "no light curves at MAST"
        return

    picks, has_ffi, has_2min = choose_curves(sr, args.max_sectors)
    row.update(
        authors="|".join(sorted({str(a) for a in sr.author})),
        n_products=len(sr), has_ffi=has_ffi, has_2min=has_2min,
        ffi_only=has_ffi and not has_2min,
    )
    if not picks:
        row["status"] = "no QLP, TESS-SPOC or SPOC light curves"
        return

    for i in picks:
        try:
            lc = sr[i].download()
        except Exception as exc:
            print(f"\n    download failed ({exc})", end=" ")
            continue
        if lc is None:
            continue
        if not (np.isfinite(star["ra"]) and np.isfinite(star["dec"])):
            try:  # a target list gives no coordinates; the file header does
                star["ra"], star["dec"] = float(lc.meta["RA"]), float(lc.meta["DEC"])
            except (KeyError, TypeError, ValueError):
                pass
        prepared = prepare(lc, args.window)
        if prepared is None:
            continue
        time, flux = prepared
        sector, author = sector_of(sr.mission[i]), str(sr.author[i])
        events = find_dips(time, flux, snr_min=args.snr)
        for e in events:
            e.update(
                tic=tic, sector=sector, author=author, verdict="", n_shared=0,
                neighbours="", centroid_sigma=np.nan, centroid_shift_mpx=np.nan,
                asteroid_speed=np.nan, asteroid_dist_px=np.nan,
                centroid_plot="", frames_plot="",
            )
        star["sectors"].append(
            {"sector": sector, "author": author, "time": time,
             "flux": flux.astype(np.float32), "events": events}
        )
        star["events"].extend(events)

    if not star["sectors"]:
        row["status"] = "light curves unusable"
        return
    star["sectors"].sort(key=lambda s: s["sector"])
    row["sectors_examined"] = "|".join(str(s["sector"]) for s in star["sectors"])


def summarise_star(star, args, plot_dir):
    """Turn a vetted star into its summary row, and draw its plot if flagged."""
    row, events = star["row"], star["events"]
    row["ra"], row["dec"] = star["ra"], star["dec"]
    if not star["sectors"]:
        return row

    counts = Counter(e["verdict"] for e in events)
    good = [e for e in events if e["verdict"] in GOOD_VERDICTS]
    row.update(
        n_dips=len(events), n_shared=counts["shared"], n_moved=counts["moved"],
        n_asteroid=counts["asteroid"], n_candidate=len(good),
    )

    flags, notes = [], []
    if good:
        best = max(good, key=lambda e: e["snr"])
        row.update(
            best_snr=round(best["snr"], 1), best_time_btjd=round(best["time"], 3),
            best_sector=best["sector"], best_depth_ppt=round(best["depth_ppt"], 3),
            best_duration_hr=best["duration_hr"],
            centroid_sigma=best["centroid_sigma"],
            centroid_shift_mpx=best["centroid_shift_mpx"],
            neighbour_dips="|".join(sorted({t for e in good for t in e["neighbours"].split("|") if t})),
        )
        label = "single dip" if len(good) == 1 else f"{len(good)} dips"
        if any(e["verdict"] == "unchecked" for e in good):
            label += " (pixels unchecked)"
        flags.append(label)
    if row["ffi_only"]:
        flags.append("FFI-only")
    row["flag"] = ", ".join(flags)

    if counts["variable"]:
        notes.append(
            f"{counts['variable']} dips left after vetting "
            "(variable, binary or noisy; not flagged as single)"
        )
    rejected = [f"{counts[v]} {v}" for v in ("shared", "moved", "asteroid") if counts[v]]
    if rejected:
        notes.append("rejected: " + ", ".join(rejected))
    if row["neighbour_dips"]:
        notes.append("same dip in a neighbouring star")
    row["status"] = "; ".join([p for p in [row["flag"]] + notes if p]) or "nothing notable"

    if flags and not (args.dips_only and not good):
        prefix = "dip" if good else "ffi"
        path = plot_dir / f"{prefix}_TIC{star['tic']}.png"
        tmag = star["tmag"]
        tmag_txt = "" if not np.isfinite(tmag) else f"   Tmag {tmag:.1f}"
        toi_txt = "   (already a TOI)" if row["known_toi"] else ""
        make_plot(f"TIC {star['tic']}{tmag_txt}   [{row['flag']}]{toi_txt}", star["sectors"], path)
        row["plot"] = str(path)
    return row


def events_table(stars):
    """Every dip found, for events.csv."""
    cols = ["tic", "sector", "author", "time_btjd", "snr", "depth_ppt", "duration_hr",
            "verdict", "n_other_stars", "neighbour_tics", "centroid_sigma",
            "centroid_shift_mpx", "asteroid_arcsec_per_hr", "asteroid_dist_px",
            "centroid_plot", "frames_plot"]
    rows = [
        {
            "tic": e["tic"], "sector": e["sector"], "author": e["author"],
            "time_btjd": round(e["time"], 3), "snr": round(e["snr"], 1),
            "depth_ppt": round(e["depth_ppt"], 3), "duration_hr": e["duration_hr"],
            "verdict": e["verdict"], "n_other_stars": e["n_shared"],
            "neighbour_tics": e["neighbours"], "centroid_sigma": e["centroid_sigma"],
            "centroid_shift_mpx": e["centroid_shift_mpx"],
            "asteroid_arcsec_per_hr": e["asteroid_speed"],
            "asteroid_dist_px": e["asteroid_dist_px"],
            "centroid_plot": e["centroid_plot"], "frames_plot": e["frames_plot"],
        }
        for s in stars for e in s["events"]
    ]
    return pd.DataFrame(rows, columns=cols)


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep TESS stars for FFI-only light curves and isolated transit-like dips.",
    )
    src = p.add_argument_group("where to look (give --targets, or --ra and --dec)")
    src.add_argument("--targets", help="text file with one TIC ID per line")
    src.add_argument("--ra", type=float, help="field centre, decimal degrees")
    src.add_argument("--dec", type=float, help="field centre, decimal degrees")
    src.add_argument("--radius", type=float, default=0.5, help="cone radius in degrees (default 0.5)")
    src.add_argument("--min-tmag", type=float, default=6.0, help="bright limit (default 6)")
    src.add_argument("--max-tmag", type=float, default=12.0, help="faint limit (default 12)")
    src.add_argument("--include-giants", action="store_true", help="keep giant stars in a cone search")

    p.add_argument("--max-targets", type=int, default=50, help="stop after this many stars (default 50)")
    p.add_argument("--max-sectors", type=int, default=3, help="sectors to download per star (default 3)")
    p.add_argument("--snr", type=float, default=8.0, help="dip detection threshold (default 8)")
    p.add_argument("--max-dips", type=int, default=2,
                   help="more surviving dips than this means 'not a single transit' (default 2)")
    p.add_argument("--window", type=float, default=2.0,
                   help="flattening window in days; keep it well above the dip length (default 2)")

    vet = p.add_argument_group("vetting")
    vet.add_argument("--shared-min", type=int, default=3,
                     help="reject a dip seen at the same time in this many separate stars "
                          "(default 3; raise it for sweeps of many hundreds of stars, "
                          "0 turns the veto off)")
    vet.add_argument("--centroid-sigma", type=float, default=3.5,
                     help="reject a dip if the image shift during it is this unusual (default 3.5)")
    vet.add_argument("--no-pixel-checks", "--no-centroid", dest="no_centroid", action="store_true",
                     help="skip the centroid and asteroid checks (no image cutouts are downloaded)")

    p.add_argument("--dips-only", action="store_true", help="only plot stars with dips")
    p.add_argument("--outdir", default="sweep_output", help="output folder")
    args = p.parse_args()
    if not args.targets and (args.ra is None or args.dec is None):
        p.error("give either --targets FILE or both --ra and --dec")
    return args


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=lk.LightkurveWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    outdir = Path(args.outdir)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.targets:
        targets = targets_from_file(args.targets)
    else:
        print(f"Querying the TESS Input Catalog around RA {args.ra}, Dec {args.dec} ...")
        targets = targets_from_cone(
            args.ra, args.dec, args.radius, args.min_tmag, args.max_tmag, args.include_giants
        )
    print(f"{len(targets)} target(s) found; sweeping the first {min(len(targets), args.max_targets)}.")
    targets = targets.head(args.max_targets)

    known_tois = load_known_tois()

    # ---- Pass 1: download every star and find its dips -----------------------
    stars = []
    errors_in_a_row = 0
    try:
        for n, t in enumerate(targets.itertuples(index=False), start=1):
            print(f"[{n}/{len(targets)}] TIC {t.tic}: ", end="", flush=True)
            star = new_star(int(t.tic), float(t.tmag), float(t.ra), float(t.dec), known_tois)
            try:
                collect_star(star, args)
                errors_in_a_row = 0
                print(star["row"]["status"] or f"{len(star['events'])} dip(s) found")
            except Exception as exc:  # keep going if one star misbehaves
                star = new_star(int(t.tic), float(t.tmag), float(t.ra), float(t.dec), known_tois)
                star["row"]["status"] = f"error: {str(exc)[:200]}"
                errors_in_a_row += 1
                print(star["row"]["status"])
            stars.append(star)
            if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                print(
                    f"\n{MAX_ERRORS_IN_A_ROW} stars in a row failed - stopping. "
                    "Check your internet connection and whether MAST is up "
                    "(https://mast.stsci.edu), then run again."
                )
                break
    except KeyboardInterrupt:
        print("\nInterrupted - vetting and saving what has been done so far.")

    if not stars:
        print("Nothing to save.")
        return

    # ---- Pass 2: vet the dips -------------------------------------------------
    n_found = sum(len(s["events"]) for s in stars)
    if args.shared_min > 0:
        mark_shared(stars, args.shared_min)
    n_shared = sum(e["verdict"] == "shared" for s in stars for e in s["events"])
    print(f"\nShared-time veto: {n_shared} of {n_found} dip(s) rejected as common to several stars.")
    vet_pixels(stars, args, plot_dir)

    # ---- Pass 3: tables and plots ---------------------------------------------
    summary = pd.DataFrame([summarise_star(s, args, plot_dir) for s in stars])
    # Best candidates first: surviving dip, not a known TOI, not shared with a
    # neighbour, centroid actually checked, then by strength.
    summary["_rank"] = (
        (summary["n_candidate"] > 0).astype(int) * 16
        + (summary["known_toi"] != True).astype(int) * 8  # noqa: E712
        + (summary["neighbour_dips"] == "").astype(int) * 4
        + summary["centroid_sigma"].notna().astype(int) * 2
        + summary["ffi_only"].fillna(False).astype(int)
    )
    summary = summary.sort_values(
        ["_rank", "best_snr"], ascending=[False, False], na_position="last"
    ).drop(columns="_rank")
    summary.to_csv(outdir / "summary.csv", index=False)
    events_table(stars).to_csv(outdir / "events.csv", index=False)

    verdicts = Counter(e["verdict"] for s in stars for e in s["events"])
    with_dips = summary[summary["n_candidate"] > 0]
    print(
        f"\nDone. {len(summary)} stars swept, {n_found} dip(s) found: "
        f"{verdicts['shared']} shared, {verdicts['moved']} image moved, "
        f"{verdicts['asteroid']} asteroid, {verdicts['variable']} in variable stars, "
        f"{verdicts['candidate']} passed, {verdicts['unchecked']} unchecked."
    )
    if len(with_dips):
        print(f"\n{len(with_dips)} star(s) with surviving dips:")
        for r in with_dips.itertuples(index=False):
            checked = ("pixels unchecked" if not np.isfinite(r.centroid_sigma)
                       else f"centroid {r.centroid_sigma:.1f} sigma, no moving object")
            print(
                f"  TIC {r.tic}: sector {int(r.best_sector)}, BTJD {r.best_time_btjd:.2f}, "
                f"SNR {r.best_snr}, {r.best_depth_ppt:.2f} ppt, {checked}"
                + ("  [already a TOI]" if r.known_toi == True else "")  # noqa: E712
                + ("  [dip shared with a neighbour]" if r.neighbour_dips else "")
            )
    else:
        print("\nNo dips survived vetting.")
    print(f"\nSummary: {outdir / 'summary.csv'}\nEvents:  {outdir / 'events.csv'}\nPlots:   {plot_dir}/")


if __name__ == "__main__":
    main()
