#!/usr/bin/env python
"""
joint_orbit.py - what orbit and what size of companion fit everything known
about a star that has shown one eclipse?

It combines these kinds of evidence:

  A. The eclipse's shape (TESS). For a given period and companion size, how
     long and how deep an eclipse can be is fixed by geometry. The script fits
     a limb-darkened eclipse model to the TESS points for every combination.
  B. Missing eclipses (TESS, SuperWASP, Gaia photometry). Every time a light
     curve covers a moment where a trial period predicts an eclipse and shows
     none, that period loses support.
  C. Gaia astrometry. A companion makes the star wobble on the sky. The longer
     the period and the heavier the companion, the bigger the wobble. Gaia saw
     almost none, which limits the product of the two.
  D. Geometry. Wide orbits are less likely to be seen edge-on enough to eclipse.
  E. Measured velocities, where there are any. The companion pulls the star
     toward and away from us. Spectra taken on known dates (the LAMOST survey)
     and Gaia's median velocity must match what each trial orbit predicts.
  F. The companion's own eclipse. Half an orbit (in angle) after it crosses
     the star, the companion passes behind it, and the system dims by the
     companion's share of the light. When and how deeply depends on the
     orbit's shape, so the light curves at those moments test each orbit.

The orbit may be eccentric. The eclipse's length fixes how fast the companion
crossed the star, and a period fixes how fast a circular orbit would cross,
so the ratio of the two measures how eccentric the orbit must be.

The answer is a probability map over orbital period and companion mass (with
the companion's radius tied to its mass by the usual relation for low-mass
stars), plus the eccentricity that goes with it.

Usage
-----
Run it in the folder that holds wasp_lc.csv:

    python joint_orbit.py

For a given star:

    python joint_orbit.py --tic 123456789 --t0 2345.678 --duration 3.0

If the star has shown the same eclipse more than once, give one time as --t0
(the one with the best data) and the others with --also:

    python joint_orbit.py --tic 123456789 --t0 3377.806 --duration 2 --also 2643.848

The script looks for other eclipses itself and usually finds them, but --also
makes sure. The period must then fit every one of them, and any period that
puts a further eclipse on data showing none is ruled out.

A short period can survive that test if each further eclipse it predicts falls
in a gap in the light curves. The cleaned light curves the script downloads
have many gaps, because the pipelines drop whatever they could not clean. Add
--fill-gaps to download two other versions of each sector (the uncorrected
flux, and the QLP pipeline's light curve) and use them inside those gaps:

    python joint_orbit.py --tic 123456789 --t0 3377.806 --duration 2 --also 2643.848 --fill-gaps

For the stars' sizes and light to come from one isochrone fitted to the system's
photometry (rather than main-sequence relations), add --iso auto --feh <[Fe/H]>
(downloads MIST's isochrones once) or put a MIST .iso.cmd file in the output folder:

    python joint_orbit.py --tic 123456789 --t0 3377.806 --duration 2 --also 2643.848 --iso auto --feh -0.1

The first run needs the internet: it downloads the star's TESS light curves,
its TESS Input Catalog entry and its Gaia DR3 values and photometry, and saves
them in the output folder. Later runs reuse those files and work offline.

If you already have Gaia's per-visit photometry as gaia_epoch_photometry.csv
(columns BJD, FLUX, FLUX_ERR), put it next to wasp_lc.csv or in the output
folder and the script uses it.

Output (in --outdir, default "joint_output")
--------------------------------------------
    summary.txt         everything printed to the screen
    posterior.png       the probability map, with period and mass on the axes
    posterior_grid.csv  the same map as numbers
    eclipse_fit.png     the TESS eclipse with the best-fitting model
    allowed_periods.csv every period interval the photometry leaves open
    candidate_periods.csv   only if a second eclipse of the same kind is known:
                        the exact periods still open (the time between the two
                        eclipses divided by each whole number that survives)
    gaia_candidate_periods.csv   only if Gaia caught a possible second eclipse:
                        every period that fits both eclipses and is still open
    rv_epochs.csv       the velocities used (from LAMOST; add rows of your own
                        with columns JD, RV, RV_ERR, SOURCE and rerun). An
                        optional STAR column says whose velocity each row is:
                        1 (the eclipsed star, the default), 2 (the companion),
                        A and B (the two stars of a double-lined spectrum when
                        you cannot tell which is the eclipsed one), or blend
                        (both stars' lines merged into one measurement).
    allowed_periods_with_velocities.csv   the open periods that also fit them

Method, briefly
---------------
B is computed as in period_limits.py: a sliding window gives a depth and a
calibrated error at every testable time, and for each of millions of trial
periods the evidence for and against eclipses at the predicted times is
summed. A period is ruled out at 3 sigma. Each coarse period bin of the final
map is then weighted by the fraction of its periods that survive.

Survey data with only a few visits a night (ATLAS, ASAS-SN, ZTF; give them
with --extra, e.g. the atlas_c.csv and atlas_o.csv that atlas_check.py
writes) are too sparse for the sliding window. Each night is judged on its
own instead: for every period that puts an eclipse near that night, the
eclipse profile predicts how faint each visit should be, and the visits are
compared with that. A night with two or more visits deep enough to be an
eclipse is a DEEP DIP, and the period must put an eclipse on it (turn this
off with --ignore-dips if the dip looks like bad data).

A second eclipse that is certain is handled as a rule, not as evidence: the
period must put an eclipse on it, to within a few minutes. It is then left out
of the summed evidence. (An earlier version counted it there too, and its
strong support cancelled the evidence against a period from one missing
eclipse elsewhere, so periods the two sectors plainly exclude were kept.)
For the same reason, one predicted eclipse that the data at that moment rule
out by themselves is enough to rule a period out, whatever support the period
collects from dips elsewhere.

Gaia's photometry is sparse, so it is handled differently. Gaia sees a star
twice 1.8 hours apart and often again six hours later. Visits made within
hours of each other are compared with one another: an eclipse makes one of
them fainter than the rest. For each trial period the measured differences
are compared with what the known eclipse's profile predicts. A visit much
fainter than its group (5 sigma) is reported as a possible second eclipse.
Because it is a single measurement it is not taken as proven: the main
results ignore it, and a separate section gives the answer if it is real.
In that case the period must be the time between the two eclipses divided by
a whole number, which leaves a short list of exact candidate periods.

A uses a numerical model of one disc crossing a limb-darkened disc. For each
companion mass the model is fitted over the eclipse's total length and the
impact parameter (how far from the centre of the star the companion crosses).
The fit runs twice: a coarse pass finds where the good fits lie and how sharply
the data pin them down, and a fine pass covers that region in steps small
enough that the fit quality barely changes from one step to the next. The fit
quality becomes a likelihood, averaged over impact parameter and over the
speeds an eccentric orbit of each period could have. (An earlier version used
one coarse grid; for a well-measured eclipse that made the mass range
collapse onto a single grid point.)

C predicts the size of the wobble for each period and mass at the star's
Gaia distance, converts it to the extra scatter Gaia would have recorded
(half the wobble, for an edge-on orbit), and compares that with the scatter
Gaia did record.

E takes each trial period in turn. The eclipse time then fixes where in the
orbit each velocity was measured; the eccentricity and the crossing speed fix
the orbit's orientation up to a mirror image; the companion's mass fixes the
size of the swing. Only the system's overall velocity is left to fit. Gaia's
median velocity is compared with the median each orbit would give. Two
measurements are needed before this says anything. If a spectrum happened to
catch the brief, large swing near closest approach, it singles out a short
list of periods and settles which mirror image is right; if not, it says
little.

F runs once the period is pinned down to a few dozen candidates (by a second
eclipse, say). For every remaining period, companion mass, crossing speed,
eccentricity and mirror image it works out where the companion would pass
behind the star. On an eccentric orbit the two stars are farther apart then
than at the known eclipse (or closer), so the companion may pass clear of the
star altogether; when it does not, the depth follows from its share of the
light (a main-sequence star of that mass) and how much of it the star covers,
and the timing and length from Kepler's laws. The light curves are measured
with a box of that length at every predicted time, with the primary eclipses
cut out, and compared with an eclipse of the predicted depth and with none,
as in part B. Orbits whose companion eclipse would be plain where the data
show nothing lose their probability, which mostly trims the highest
eccentricities. Support works differently: one orbit's worth of dip counts for
at most 20 to 1 (a lone dip of the right timing is too easily a ramp or a bad
stretch), and boxes beside a data gap are not used. Only a dip that repeats
every orbit, or shows in two light curves, can settle the orbit. If some orbits
find dips where they predict them, the script says so as a POSSIBLE COMPANION
ECLIPSE and lists what every light curve shows at those times. The ECCENTRICITY section then also
gives one representative orbit that respects every test; system_view.py and
system_movie.py draw that one.

Stars and Gaia (new)
--------------------
* If the folder holds stellar_split.json (from stellar_model.py), both stars come
  from one MIST isochrone fitted to the system's photometry: radii, temperatures
  and light in each band follow from mass at the fitted age, and the eclipsed star
  supplies whatever TESS light the companion does not. Otherwise the catalogue's
  light is shared as mass^4.5, as before. Either way the companion may now be the
  brighter star (--eclipsed-brighter restores the old limit for the mass^4.5 split).
* Part A averages over the eclipsed star's radius (--radius-err; default from the
  isochrone fit, else 7%) and, with --third-light, over extra light from other
  stars. Crossing speeds are reported in solar radii per day.
* Part C uses Gaia's RUWE, predicted transit by transit from the scan law
  (gost.csv, the Gaia Observation Forecast Tool's CSV export, in the output folder)
  and judged against comparison stars (gaia_calibration.json). Both are fetched or
  explained by gaia_tools.py. The pair's own wobble can only add to RUWE, so orbits
  predicting more than Gaia saw lose probability; an observed excess is reported,
  not explained (triple_fit.py tests a third star).
* Part E models Gaia's velocities transit by transit when the period is pinned
  down: the median and the robust amplitude are both compared, with each transit's
  velocity following the brighter star once the lines separate (--rvs-resolve).

Assumptions and weak points
---------------------------
* The companion is an ordinary star on the same isochrone (or, without
  stellar_split.json, on the main-sequence relation). A young, inflated object or
  a white dwarf would break this.
* Eccentricities from 0 to 0.95 are taken as equally likely beforehand. Use
  --circular to force a circular orbit and see whether one can fit at all.
* The eclipse is on the catalogued star. If it is really on a smaller,
  unresolved star nearby, the sizes and speeds here are wrong.
* Third light (--third-light) only dilutes the eclipse in part A; it does not
  re-split the light between the two stars.
* Part C: without gost.csv the scan law is synthetic and RUWE predictions are
  approximate; without gaia_calibration.json single stars are assumed to have
  RUWE 1.00 +/- 0.05. Treat the upper end of the period range as soft.
* A DEEP DIP in survey visits must show at least three visits clearly
  faint, two of them strongly, all consistent with one another; anything
  less only counts as ordinary evidence. A cloudy night can still pass.
* Part B inherits every caveat of period_limits.py: all eclipses are assumed
  to look like the known one, and the period is assumed constant.
* Limb-darkening values are approximate; their effect is small.
* Part E takes each velocity as the eclipsed star's alone unless rv_epochs.csv
  says otherwise. If the companion is nearly as bright, the two sets of lines
  blend and a survey's automatic velocity can be pulled toward the
  companion's. Look at the spectrum; if it shows two sets of lines, measure
  both and mark the rows A and B, and mark low-resolution values as blend.
  Gaia's median is then treated as a blend too. Blends are modelled as the
  light-weighted average of the two stars' velocities.
  LAMOST gives the date but not the hour of each spectrum; the script assumes a
  time good to about three hours. Velocities from different instruments can
  differ by a few km/s for instrumental reasons; the errors are widened to
  4 km/s (medium resolution) and 7 km/s (low resolution) to cover that.
* Part F takes the companion's light from its mass (light ~ mass^4.5) and its
  size from the low-mass relation. A companion that is fainter for its mass (a
  white dwarf, say) would make no visible eclipse, and the high eccentricities
  part F rules out would come back: rerun with --skip-secondary to see that case.
  A companion eclipse as deep as the known one would be taken for another
  primary eclipse in part B; if no period then fits, that is a possibility.
* A single faint Gaia visit can be an instrumental glitch. Check Gaia's flags
  and the blue and red photometry taken on the same visit before trusting it.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gaia_tools                                              # noqa: E402  (same folder)
import stellar_model                                           # noqa: E402
from stellar_model import load_split                           # noqa: E402

GAP_DAYS = 0.25        # a break longer than this splits a light curve into segments
GUARD = 0.6            # baseline points lie beyond this many eclipse lengths...
REACH = 3.0            # ...and within this many
MIN_IN, MIN_OUT = 4, 8
STEP = 1.0 / 8.0       # window positions are this fraction of the length apart
PHASE_TOL = 1.0 / 16.0  # allowed slip of a predicted time between grid periods
EPOCH_FLOOR = -12.5    # no single predicted eclipse may count for more than 5 sigma
DEPTH_SPREAD = 0.25    # an eclipse's measured depth may differ from the known one by
                       # about this fraction (sampling, timing, colour band)
TIME_SLACK = 0.25      # a predicted eclipse may sit this many lengths early or late
MIN_CLASS = 50         # too few edge windows to calibrate -> they are not used
OUTLIER_FLOOR = 1e-4   # with no eclipse, a depth of any size still has this much
                       # probability per unit depth (rare glitches happen)
CONFIRM_SIGMA = 7.0    # a stand-alone dip this strong counts as a second eclipse
KEEP_MOST = 1 | 2 | 4 | 8 | 32   # for gap-filling data, drop only points taken while the
                                 # spacecraft was re-pointing, in safe mode or firing thrusters
ERR_CUT = 2.0          # drop points whose quoted error exceeds this many medians
GAIA_GROUP = 0.4       # Gaia visits closer together than this (days) are compared
GAIA_DIP_SIGMA = 5.0   # one visit this much fainter than its group is a possible eclipse
MIN_GROUP = 20
TIME_NAMES = ("BJD", "HJD", "HJD_UTC", "JD", "TIME")
FLUX_NAMES = ("TAMFLUX2", "FLUX", "FLUX2", "PDCSAP_FLUX", "SAP_FLUX")
HJD_UTC_TO_BJD_TDB = 65.0 / 86400.0   # leap seconds + 32.184 s, to within a few seconds


class Tee:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


# ----------------------------------------------------------------------------
# Reading light curves
# ----------------------------------------------------------------------------
def read_table(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    from astropy.table import Table

    if suffix == ".tbl":
        return Table.read(path, format="ipac").to_pandas()
    if suffix in (".fits", ".fit"):
        return Table.read(path).to_pandas()
    return pd.read_csv(path, sep=None, engine="python")


def find_column(columns, wanted):
    lookup = {c.lower(): c for c in columns}
    for name in wanted:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def load(path, say):
    """Read one light curve. Returns a table with time t (BJD), flux f, relative error e."""
    raw = read_table(path)
    cols = list(raw.columns)
    tcol, fcol = find_column(cols, TIME_NAMES), find_column(cols, FLUX_NAMES)
    if tcol is None or fcol is None:
        sys.exit(f"{path}: could not find time and flux columns among {', '.join(cols)}")
    time = pd.to_numeric(raw[tcol], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(raw[fcol], errors="coerce").to_numpy(float)
    if np.nanmedian(time) < 2400000:
        sys.exit(f"{path}: times in '{tcol}' are not full Julian dates "
                 f"(median {np.nanmedian(time):.1f}). Add the offset before running.")
    if tcol.upper().startswith("HJD"):
        time = time + HJD_UTC_TO_BJD_TDB
    ecol = find_column(cols, (fcol + "_ERR",))
    err = pd.to_numeric(raw[ecol], errors="coerce").to_numpy(float) if ecol else None

    # SuperWASP image numbers begin with the camera number; cameras differ in scale.
    image = find_column(cols, ("IMAGEID",))
    if image and pd.to_numeric(raw[image], errors="coerce").min() > 1e17:
        group = (pd.to_numeric(raw[image], errors="coerce") // 10**15).to_numpy()
    else:
        group = np.zeros(len(raw))

    good = np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    have_err = err is not None and np.isfinite(err[good]).mean() > 0.9 and np.nanmedian(err[good]) > 0
    if have_err:
        rel = err / np.where(flux > 0, flux, np.nan)
        good &= np.isfinite(rel) & (rel > 0)
        good &= rel < ERR_CUT * np.nanmedian(rel[good])
    else:
        rel = np.full(len(raw), np.nan)
    df = pd.DataFrame({"t": time[good], "f": flux[good], "e": rel[good], "g": group[good]})
    size = df.groupby("g").f.transform("size")
    df = df[size >= MIN_GROUP].sort_values("t")
    df["f"] /= df.groupby("g").f.transform("median")
    removed = []
    if len(df) > 200 and np.median(np.diff(df.t)) < NOISY_DENSE_CADENCE:
        bad, removed = noisy_stretches(df.t.to_numpy(), df.f.to_numpy())
        df = df[~bad].reset_index(drop=True)
    scatter = 1.4826 * np.median(np.abs(np.diff(df.f))) / np.sqrt(2)   # point-to-point
    if not have_err:
        df["e"] = scatter
    keep = (df.f < 1 + 6 * max(scatter, 1e-4) + 0.02) & (df.f > 0.5)
    df = df[keep].reset_index(drop=True)
    say(f"  {Path(path).name}: {len(df)} points, JD {df.t.min():.1f} to {df.t.max():.1f}, "
        f"point scatter {scatter * 100:.2f}%  (time: {tcol}, flux: {fcol})")
    if removed:
        total = sum(b - a for a, b in removed)
        say(f"    left out {total:.2f} days of unreliable data (scatter or brightening far above this "
            f"light curve's normal): " + ", ".join(f"JD {a:.2f}-{b:.2f}" for a, b in removed[:6])
            + (f" and {len(removed) - 6} more" if len(removed) > 6 else ""))
    return df


NOISY_DENSE_CADENCE = 0.02   # only light curves sampled faster than this (days) are screened
NOISY_FACTOR = 4.0           # local scatter this many times the normal level marks a bad stretch
NOISY_WINDOW = 0.25          # days over which the local scatter is measured
NOISY_MARGIN = 0.1           # days also dropped on each side of a bad stretch


def noisy_stretches(t, f, factor=NOISY_FACTOR, window=NOISY_WINDOW, margin=NOISY_MARGIN):
    """Stretches of a densely sampled light curve that cannot be trusted: where the local
    scatter is `factor` times the light curve's normal level, or where the flux over a few hours
    sits well above its 2-day running median (an eclipse only ever dims a star; a short rise is
    scattered light or a systematic; slow drifts in uncorrected flux are left alone).

    The scatter comes from second differences, f[i+1] - 2 f[i] + f[i-1], which cancel any
    straight-line trend, so the steep but smooth sides of a real eclipse do not count as noise.
    A margin is dropped around each bad stretch, since the data next to one (like a dip just
    after a scattered-light spike) are usually affected too.

    Returns (mask of points to drop, list of (start, end) times of the dropped stretches).
    """
    n = len(t)
    cadence = float(np.median(np.diff(t)))
    d2 = np.full(n, np.nan)
    d2[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / np.sqrt(6.0)
    near = np.r_[False, (np.diff(t)[:-1] < 5 * cadence) & (np.diff(t)[1:] < 5 * cadence), False]
    d2[~near] = np.nan                                       # not across data gaps
    a = np.abs(d2)
    normal = 1.4826 * np.nanmedian(a)
    if not np.isfinite(normal) or normal <= 0:
        return np.zeros(n, bool), []
    lo = np.searchsorted(t, t - window / 2)
    hi = np.searchsorted(t, t + window / 2)
    wlo = np.searchsorted(t, t - 1.0)                          # a 2-day baseline for the brightening test
    whi = np.searchsorted(t, t + 1.0)
    local = np.full(n, np.nan)
    level = np.full(n, np.nan)
    base = np.full(n, np.nan)
    for i in range(n):
        seg = a[lo[i]:hi[i]]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 8:
            local[i] = 1.4826 * np.median(seg)
        level[i] = np.median(f[lo[i]:hi[i]])
        base[i] = np.median(f[wlo[i]:whi[i]])
    bad = (local > factor * normal) | (level - base > max(0.02, 10 * normal))
    if not bad.any():
        return bad, []
    # widen by the margin, then collect the stretches
    starts, ends = t[bad] - margin, t[bad] + margin
    order = np.argsort(starts)
    spans = []
    for s, e in zip(starts[order], ends[order]):
        if spans and s <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], e)
        else:
            spans.append([s, e])
    drop = np.zeros(n, bool)
    for s, e in spans:
        drop |= (t >= s) & (t <= e)
    return drop, [(max(s, t[0]), min(e, t[-1])) for s, e in spans]


def download_tess(tic, outdir, say):
    """Fetch every TESS sector for a star and save each as a CSV. Needs the internet."""
    import lightkurve as lk

    tic = str(tic).upper().replace("TIC", "").strip()
    sr = lk.search_lightcurve(f"TIC {tic}", mission="TESS")
    if len(sr) == 0:
        sys.exit(f"No TESS light curves found for TIC {tic}.")
    authors = np.array([str(a) for a in sr.author])
    sectors = np.array([int(str(m).split()[-1]) for m in sr.mission])
    exptime = np.array(sr.exptime.value, dtype=float)
    chosen = {}
    for author in ("SPOC", "TESS-SPOC", "QLP"):               # order of preference
        for i in np.where(authors == author)[0][np.argsort(-exptime[authors == author])]:
            if exptime[i] >= 100:                             # skip 20-second data
                chosen.setdefault(sectors[i], i)
    paths = []
    for sector in sorted(chosen):
        i = chosen[sector]
        lc = sr[i].download().remove_nans()
        out = pd.DataFrame({"BJD": np.asarray(lc.time.value, float) + 2457000.0,
                            "FLUX": np.asarray(lc.flux.value, float)})
        err = np.asarray(lc.flux_err.value, float)
        if np.isfinite(err).mean() > 0.9:
            out["FLUX_ERR"] = err
        path = outdir / f"tess_s{sector:02d}.csv"
        out.to_csv(path, index=False)
        say(f"  downloaded sector {sector} ({authors[i]}) -> {path}")
        paths.append(str(path))
    return paths


def flatten_slowly(t, f, width):
    """Divide out changes slower than `width` days (a running median), one data segment at a time."""
    from scipy.ndimage import median_filter

    out = np.empty_like(f)
    breaks = np.r_[0, np.where(np.diff(t) > GAP_DAYS)[0] + 1, len(t)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        cadence = np.median(np.diff(t[a:b])) if b - a > 2 else width
        size = int(min(max(width / max(cadence, 1e-6), 5), b - a)) // 2 * 2 + 1
        out[a:b] = f[a:b] / median_filter(f[a:b], size=size, mode="nearest")
    return out


NET_TIMEOUT = 180          # seconds to wait on any one archive request before carrying on without it


def within(seconds, fn, *a, **k):
    """Run fn(*a, **k), giving up after `seconds`: raises TimeoutError, and the request is left to
    finish (or not) in the background. Archives sometimes stall without ever answering."""
    import threading
    box = {}

    def work():
        try:
            box["value"] = fn(*a, **k)
        except BaseException as exc:                     # noqa: BLE001  (handed back to the caller)
            box["error"] = exc
    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"no answer within {seconds:.0f} s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def download_fill(tic, outdir, dur, say):
    """Fetch other versions of the sectors already saved, to cover gaps in the main light curves.

    The main light curves are the pipelines' cleaned ones, which leave out every
    stretch the pipeline could not clean (for example when stray light from the
    Earth or Moon reached the camera). Two other versions often have points
    there: the same pipeline's uncorrected flux, and the QLP pipeline's light
    curve. Neither is as clean, but a deep eclipse shows in them all the same.
    Slow changes are divided out so that stray-light ramps do not get in the way.
    """
    import lightkurve as lk

    tic = str(tic).upper().replace("TIC", "").strip()
    have = sorted({int(m.group(1)) for q in outdir.glob("tess_s*.csv")
                   if (m := re.fullmatch(r"tess_s(\d+)", q.stem))})
    wanted = (("SPOC", "raw"), ("TESS-SPOC", "raw"), ("QLP", "qlp"))
    # What MAST offers, remembered from earlier runs (tess_versions.json): when every version it
    # offered for these sectors is already saved, there is nothing to ask MAST about.
    cache = outdir / "tess_versions.json"
    known = json.loads(cache.read_text()) if cache.exists() else None
    if known is not None and set(map(str, have)) <= set(known):
        missing = [(sec, tag) for sec in have for author, tag in wanted
                   if author in known[str(sec)] and not (outdir / f"tess_s{sec:02d}_{tag}.csv").exists()]
        if not missing:
            return []
    say(f"  asking MAST which versions exist (gives up after {NET_TIMEOUT} s)...")
    try:
        sr = within(NET_TIMEOUT, lk.search_lightcurve, f"TIC {tic}", mission="TESS")
    except TimeoutError:
        say("  MAST did not answer; carrying on with the light curves already saved (rerun later, or "
            "drop --fill-gaps)")
        return []
    authors = np.array([str(a) for a in sr.author])
    sectors = np.array([int(str(m).split()[-1]) for m in sr.mission])
    exptime = np.array(sr.exptime.value, dtype=float)
    offered = {str(sec): sorted({str(a) for a, x, e in zip(authors, sectors, exptime) if x == sec and e >= 100})
               for sec in have}
    cache.write_text(json.dumps(offered, indent=1))
    width = max(1.0, 8.0 * dur)
    paths = []
    for sector in have:
        for author, tag in wanted:
            path = outdir / f"tess_s{sector:02d}_{tag}.csv"
            idx = np.where((authors == author) & (sectors == sector) & (exptime >= 100))[0]
            if path.exists() or len(idx) == 0:
                continue
            i = idx[np.argmax(exptime[idx])]
            say(f"  fetching sector {sector} ({author}, {'uncorrected flux' if tag == 'raw' else 'light curve'})...")
            try:
                if tag == "raw":
                    lc = within(NET_TIMEOUT, sr[i].download, quality_bitmask=KEEP_MOST)
                    flux = np.asarray(lc["sap_flux"].value, float)
                else:
                    lc = within(NET_TIMEOUT, sr[i].download)
                    flux = np.asarray(lc.flux.value, float)
                t = np.asarray(lc.time.value, float) + 2457000.0
                ok = np.isfinite(t) & np.isfinite(flux) & (flux > 0)
                order = np.argsort(t[ok])
                t, flux = t[ok][order], flux[ok][order]
                if len(t) < 50:
                    continue
                pd.DataFrame({"BJD": t, "FLUX": flatten_slowly(t, flux, width)}).to_csv(path, index=False)
                say(f"  downloaded sector {sector} ({authors[i]}, "
                    f"{'uncorrected flux' if tag == 'raw' else 'its own light curve'}) -> {path}")
                paths.append(str(path))
            except Exception as exc:
                say(f"  sector {sector} ({authors[i]}, {tag}) could not be fetched: {str(exc)[:100]}")
    return paths


# ----------------------------------------------------------------------------
# Step 1 and 2: the depth series and its calibrated errors
# ----------------------------------------------------------------------------
def depth_series(df, dur):
    """Depth and error of an eclipse-length window at every position the data can test.

    Returns an array with columns: centre, depth, error, edge flag.
    """
    t, f, e = df.t.to_numpy(), df.f.to_numpy(), df.e.to_numpy()
    w = 1.0 / e**2
    out = []
    breaks = np.r_[0, np.where(np.diff(t) > GAP_DAYS)[0] + 1, len(t)]
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b - a < MIN_IN + MIN_OUT:
            continue
        ts, fs, es, ws = t[a:b], f[a:b], e[a:b], w[a:b]
        for c in np.arange(ts[0], ts[-1] + STEP * dur, STEP * dur):
            i0, i1 = np.searchsorted(ts, [c - REACH * dur, c + REACH * dur])
            tt, ff, ee, ww = ts[i0:i1] - c, fs[i0:i1], es[i0:i1], ws[i0:i1]
            inside, far = np.abs(tt) <= dur / 2, np.abs(tt) > GUARD * dur
            if inside.sum() < MIN_IN or far.sum() < MIN_OUT:
                continue
            two_sided = (far & (tt < 0)).sum() >= 3 and (far & (tt > 0)).sum() >= 3
            if two_sided:
                slope, level = np.polyfit(tt[far], ff[far], 1, w=1.0 / ee[far])
            else:
                slope, level = 0.0, float(np.sum(ww[far] * ff[far]) / ww[far].sum())
            pull = (ff[far] - (level + slope * tt[far])) / ee[far]
            excess = max(1.4826 * np.median(np.abs(pull - np.median(pull))), 0.7)
            depth = float(np.sum(ww[inside] * (level + slope * tt[inside] - ff[inside]))
                          / ww[inside].sum())
            err = excess * np.sqrt(1.0 / ww[inside].sum() + 1.0 / ww[far].sum())
            out.append((c, depth, err, 0.0 if two_sided else 1.0))
    return np.array(out) if out else np.empty((0, 4))


def calibrate(series, t0, dur):
    """Stretch the errors until the noise really is as large as they say.

    Brightenings cannot be eclipses, so they show what noise alone does. The
    errors are stretched until three things hold for the brightenings: their
    typical size, their 1-in-100 size and their single most extreme value are
    all no larger than Gaussian errors of that size would produce. Dips are
    then judged on the same scale. Windows at the edge of a data segment are
    calibrated separately because they behave worse.
    """
    from scipy.stats import norm

    factors = []
    for flag in (0.0, 1.0):
        cls = series[:, 3] == flag
        sel = cls & (np.abs(series[:, 0] - t0) > 2 * dur)
        if sel.sum() < MIN_CLASS:
            series[cls, 2] = np.inf                 # cannot be calibrated, so carries no weight
            continue
        z = series[sel, 1] / series[sel, 2]
        bright = -z[z < 0]
        core = 1.4826 * np.median(np.abs(z - np.median(z)))
        factor = max(core, 1.0)
        if len(bright) >= 30:
            n = len(bright)
            typical_max = norm.ppf((1 + 0.5 ** (1.0 / n)) / 2)     # median largest of n
            factor = max(factor, np.quantile(bright, 0.99) / 2.576, bright.max() / typical_max)
        series[cls, 2] *= factor
        factors.append(factor)
    return factors[0] if factors else 1.0


# ----------------------------------------------------------------------------
# Step 4: sum the evidence over every predicted eclipse, for every period
# ----------------------------------------------------------------------------
def period_grid(pmin, pmax, baseline, dur):
    step = PHASE_TOL * dur / baseline                 # fractional period step
    return np.exp(np.arange(np.log(pmin), np.log(pmax), step))


def log_odds(d, err, depth):
    """ln of P(measured depth | eclipse of about `depth`) / P(measured depth | no eclipse).

    Eclipse: the depth scatters around `depth` with the measurement error plus
    a spread of DEPTH_SPREAD * depth, so an eclipse sampled or timed slightly
    differently from the known one is not punished.
    No eclipse: the depth scatters around zero with the measurement error,
    plus a small constant chance of a glitch of any size. Without that floor a
    single wild point would count as overwhelming evidence for an eclipse.
    """
    var = err**2 + (DEPTH_SPREAD * depth) ** 2
    ln_eclipse = -0.5 * (d - depth) ** 2 / var - 0.5 * np.log(2 * np.pi * var)
    ln_quiet = -0.5 * d**2 / err**2 - 0.5 * np.log(2 * np.pi * err**2)
    return ln_eclipse - np.logaddexp(ln_quiet, np.log(OUTLIER_FLOOR))


def best_nearby(centres, values, usable, slack):
    """Largest of `values` among usable windows within `slack` days of each window."""
    out = np.where(usable, values, -np.inf)
    base = out.copy()
    n = len(centres)
    for shift in range(1, int(np.ceil(TIME_SLACK / STEP)) + 1):
        close = centres[shift:] - centres[:-shift] <= slack * 1.001
        out[:-shift] = np.where(close, np.maximum(out[:-shift], base[shift:]), out[:-shift])
        out[shift:] = np.where(close, np.maximum(out[shift:], base[:-shift]), out[shift:])
    assert len(out) == n
    return np.where(usable, out, 0.0)


def accumulate(periods, series, t0, dur, depth, skip=(), decisive=-4.5):
    """Sum the evidence over every predicted eclipse in one dataset, for every period.

    `skip` lists the times of other eclipses that are taken as certain. They are
    left out of the sum, because they are applied separately as a hard rule (the
    period must fit them). Counted here as well, the strong support from one
    such eclipse would cancel the evidence against a period from an eclipse
    that is missing somewhere else.

    Returns (ln LR, mirror ln LR, eclipses observed, eclipses that look like dips, clear misses).
    A clear miss is one predicted eclipse that the data there rule out by
    themselves (at `decisive`, a ln LR). Support from dips elsewhere must not
    cancel such a miss, so the caller rules those periods out whatever the sum.
    The mirror version runs the same sum on the data turned upside down, so
    that brightenings pose as dips. Brightenings cannot be eclipses, so the
    largest mirror value shows how much apparent support noise alone produces.
    """
    lnlr = np.zeros(len(periods), dtype=np.float32)
    mirror = np.zeros(len(periods), dtype=np.float32)
    count = np.zeros(len(periods), dtype=np.int16)
    dips = np.zeros(len(periods), dtype=np.int16)
    missed = np.zeros(len(periods), dtype=bool)
    centres, d, err = series[:, 0], series[:, 1], series[:, 2]
    usable = np.isfinite(err)
    safe = np.where(usable, err, 1.0)
    for_eclipse = np.where(usable, np.maximum(log_odds(d, safe, depth), EPOCH_FLOOR), 0.0)
    for_mirror = np.where(usable, np.maximum(log_odds(-d, safe, depth), EPOCH_FLOOR), 0.0)
    is_dip = usable & (d / safe >= 3) & (d >= 0.5 * depth)
    # Timing slack: take the most eclipse-like window within TIME_SLACK lengths.
    # That keeps a real eclipse from being missed by a few minutes, and makes
    # "ruled out" require that every nearby window is quiet.
    for_eclipse = best_nearby(centres, for_eclipse, usable, TIME_SLACK * dur)
    for_mirror = best_nearby(centres, for_mirror, usable, TIME_SLACK * dur)
    is_dip = best_nearby(centres, is_dip.astype(float), usable, TIME_SLACK * dur) > 0
    tmin, tmax = centres[0], centres[-1]
    reach = STEP * dur * 0.6                           # must land this close to a window

    # whole numbers of orbits between the known eclipse and this dataset
    k_lo = int(np.floor(min((tmin - t0) / periods[0], (tmin - t0) / periods[-1]))) - 1
    k_hi = int(np.ceil(max((tmax - t0) / periods[0], (tmax - t0) / periods[-1]))) + 1
    for k in range(k_lo, k_hi + 1):
        if k == 0:
            continue
        # periods for which T0 + k*P lands inside this dataset
        lo, hi = sorted(((tmin - t0) / k, (tmax - t0) / k))
        i0, i1 = np.searchsorted(periods, [lo, hi])
        if i1 <= i0:
            continue
        t_pred = t0 + k * periods[i0:i1]
        j = np.clip(np.searchsorted(centres, t_pred), 1, len(centres) - 1)
        j = np.where(t_pred - centres[j - 1] < centres[j] - t_pred, j - 1, j)
        ok = ((np.abs(centres[j] - t_pred) <= reach) & (np.abs(t_pred - t0) > 2 * dur)
              & usable[j])
        for other in skip:
            ok &= np.abs(t_pred - other) > 2 * dur
        idx = i0 + np.where(ok)[0]
        jj = j[ok]
        lnlr[idx] += for_eclipse[jj]
        mirror[idx] += for_mirror[jj]
        count[idx] += 1
        dips[idx] += is_dip[jj]
        missed[idx] |= for_eclipse[jj] <= decisive
    return lnlr, mirror, count, dips, missed


def predicted_eclipses(period, sets, t0, dur):
    """Every predicted eclipse of one period that some dataset observed."""
    rows = []
    for s in sets:
        c, d, e = s["series"][:, 0], s["series"][:, 1], s["series"][:, 2]
        k = np.round((c - t0) / period)
        miss = np.abs(c - (t0 + k * period))
        hit = (miss <= STEP * dur * 0.6) & (k != 0) & np.isfinite(e)
        for cycle in np.unique(k[hit]):
            sel = np.where(hit & (k == cycle))[0]
            j = sel[np.argmin(miss[sel])]
            rows.append({"dataset": s["name"], "orbits_from_known": int(cycle), "jd": c[j],
                         "depth_pct": d[j] * 100, "err_pct": e[j] * 100, "sigma": d[j] / e[j]})
    return pd.DataFrame(rows)


def other_eclipses(sets, t0, dur, depth):
    """Stand-alone dips in any dataset, away from the known eclipse.

    A dip is "confirmed" as a second eclipse when it is at least CONFIRM_SIGMA
    on the calibrated scale, at least 2 sigma beyond the strongest brightening
    in its dataset, not at the edge of a data segment, and between half and
    twice the known eclipse's depth.
    """
    rows = []
    for s in sets:
        c, d, e = s["series"][:, 0], s["series"][:, 1], s["series"][:, 2]
        z = np.where((np.abs(c - t0) > 2 * dur) & np.isfinite(e), d / e, 0.0)
        worst = float(-z.min())
        bar = max(5.0, worst + 1.0)                    # beat the strongest brightening
        while z.max() >= bar:
            j = int(np.argmax(z))
            edge = bool(s["series"][j, 3])
            same_kind = 0.5 * depth * s["scale"] <= d[j] <= 2.0 * depth * s["scale"]
            rows.append({"dataset": s["name"], "jd": c[j], "depth_pct": d[j] * 100,
                         "err_pct": e[j] * 100, "sigma": z[j], "edge": edge,
                         "same_kind": bool(same_kind),
                         "confirmed": bool(z[j] >= max(CONFIRM_SIGMA, worst + 2.0)
                                           and not edge and same_kind)})
            z[np.abs(c - c[j]) < 2 * dur] = 0.0
    return pd.DataFrame(rows, columns=["dataset", "jd", "depth_pct", "err_pct", "sigma",
                                       "edge", "same_kind", "confirmed"])


def intervals(periods, mask):
    """Merge runs of True in `mask` into (start, end) period intervals."""
    if not mask.any():
        return np.empty((0, 2))
    edge = np.diff(np.r_[0, mask.astype(np.int8), 0])
    start, stop = np.where(edge == 1)[0], np.where(edge == -1)[0] - 1
    half = np.sqrt(periods[1] / periods[0])
    return np.column_stack([periods[start] / half, periods[stop] * half])


# ----------------------------------------------------------------------------
# Catalogue values (downloaded once, then cached)
# ----------------------------------------------------------------------------
def report_catalogue(say, tic, star, gaia):
    """The catalogue values in one block of summary.txt, so later scripts (make_rnaas_table.py,
    system_view.py) need read nothing else. Missing values are written as n/a."""
    def v(value, spec):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return format(value, spec) if np.isfinite(value) else "n/a"

    g = gaia or {}
    say("\nCATALOGUE VALUES")
    say(f"  TIC {tic}: RA {v(star.get('ra'), '.6f')} deg, Dec {v(star.get('dec'), '.6f')} deg, "
        f"Tmag {v(star.get('tmag'), '.3f')}, Teff {v(star.get('teff'), '.0f')} K (combined light)")
    sid = g.get("source_id")
    say(f"  Gaia DR3 {int(sid) if sid else 'n/a'}: parallax {v(g.get('parallax'), '.4f')} +/- "
        f"{v(g.get('parallax_error'), '.4f')} mas, RUWE {v(g.get('ruwe'), '.3f')}, excess noise "
        f"{v(g.get('astrometric_excess_noise'), '.3f')} mas, G {v(g.get('phot_g_mean_mag'), '.3f')}, "
        f"BP-RP {v(g.get('bp_rp'), '.3f')}")
    say(f"  Gaia DR3 velocity: {v(g.get('radial_velocity'), '+.2f')} +/- {v(g.get('radial_velocity_error'), '.2f')} "
        f"km/s from {v(g.get('rv_nb_transits'), '.0f')} visits, method {v(g.get('rv_method_used'), '.0f')}, "
        f"robust amplitude {v(g.get('rv_amplitude_robust'), '.1f')} km/s")


def cached(path, fetch, say, label, offline):
    if path.exists():
        return json.loads(path.read_text())
    if offline:
        return None
    try:
        value = fetch()
    except Exception as exc:
        say(f"  could not fetch {label}: {str(exc)[:120]}")
        return None
    path.write_text(json.dumps(value, indent=1))
    return value


def fetch_star(tic):
    from astroquery.mast import Catalogs

    row = Catalogs.query_criteria(catalog="Tic", ID=int(tic))[0]

    def num(col):
        return None if np.ma.is_masked(row[col]) else float(row[col])

    return {"tic": int(tic), "ra": num("ra"), "dec": num("dec"), "rad": num("rad"),
            "mass": num("mass"), "teff": num("Teff"), "tmag": num("Tmag")}


def fetch_gaia(ra, dec):
    from astroquery.gaia import Gaia

    q = f"""SELECT source_id, ra, dec, parallax, parallax_error, ruwe,
                   astrometric_excess_noise, astrometric_excess_noise_sig,
                   phot_g_mean_mag, bp_rp, matched_transits, astrometric_n_good_obs_al
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {3.0 / 3600}))"""
    tab = Gaia.launch_job(q).get_results().to_pandas()
    if len(tab) == 0:
        raise RuntimeError("no Gaia source within 3 arcseconds")
    tab = tab.sort_values("phot_g_mean_mag")                  # the brightest is the target
    # The identifier has 19 digits. It must be read from its own whole-number column:
    # taking a whole row first turns it into a decimal number and rounds the last digits.
    out = {k: float(tab[k].iloc[0]) for k in tab.columns if k != "source_id"}
    out["source_id"] = int(tab["source_id"].astype("int64").iloc[0])
    return out


def fetch_gaia_id(ra, dec):
    """The exact Gaia DR3 identifier of the brightest source within 3 arcseconds."""
    from astroquery.gaia import Gaia

    q = f"""SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {3.0 / 3600}))
            ORDER BY phot_g_mean_mag"""
    res = Gaia.launch_job(q).get_results()
    if len(res) == 0:
        raise RuntimeError("no Gaia source within 3 arcseconds")
    return int(res["source_id"][0])


def _as_array(v):
    """A cell holding a list of values: a numpy array, a list, or text such as "(1.0, 2.0)"."""
    if isinstance(v, (np.ndarray, list, tuple)):
        return np.asarray(v)
    if isinstance(v, str) and v.strip()[:1] in "([{" and ("," in v or " " in v.strip()):
        parts = [p for p in re.split(r"[\s,()\[\]{}]+", v) if p]
        try:
            return np.array([float(p) for p in parts])
        except ValueError:
            return np.array(parts, dtype=object)
    return None


def _flatten_epochs(df):
    """One row per transit. The DATAMODEL_GAIA layout puts each source's transits in array-valued
    cells (g_transit_time, g_transit_flux, ...), as arrays or as text; spread them out into rows."""
    if not len(df):
        return df
    first = {c: _as_array(df[c].iloc[0]) for c in df.columns}
    arrays = [c for c, v in first.items() if v is not None and v.ndim == 1 and len(v) > 1]
    if not arrays:
        return df
    n = max(len(first[c]) for c in arrays)
    arrays = [c for c in arrays if len(first[c]) == n]
    cols = {}
    for c in arrays:
        parts = [_as_array(v) for v in df[c]]
        cols[c] = np.concatenate([p for p in parts if p is not None])
    return pd.DataFrame(cols)


def _read_epoch_file(name, data):
    """Parse one file the Gaia data server returned (CSV/ECSV, VOTable or FITS) into a table."""
    import io
    low = name.lower()
    if low.endswith((".xml", ".vot", ".votable")) or data[:5] in (b"<?xml", b"<VOTA"):
        from astropy.table import Table
        return Table.read(io.BytesIO(data), format="votable").to_pandas()
    if low.endswith((".fits", ".fit")) or data[:6] == b"SIMPLE":
        from astropy.table import Table
        return Table.read(io.BytesIO(data), format="fits").to_pandas()
    return pd.read_csv(io.BytesIO(data), comment="#")


def _gaia_datalink(source_id, structure, fmt):
    """Epoch photometry straight from the Gaia archive's data server (no astroquery parsing):
    a list of tables, one per file in the reply."""
    import io
    import urllib.parse
    import urllib.request
    import zipfile
    params = {"RETRIEVAL_TYPE": "EPOCH_PHOTOMETRY", "DATA_STRUCTURE": structure, "FORMAT": fmt,
              "RELEASE": "Gaia DR3", "ID": str(int(source_id)), "VALID_DATA": "false",
              "USE_ZIP_ALWAYS": "true"}
    req = urllib.request.Request("https://gea.esac.esa.int/data-server/data",
                                 data=urllib.parse.urlencode(params).encode(), method="POST")
    with urllib.request.urlopen(req, timeout=120) as reply:
        data = reply.read()
    tables = []
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for name in z.namelist():
                if not name.endswith("/"):
                    tables.append(_read_epoch_file(name, z.read(name)))
    elif data.strip():
        tables.append(_read_epoch_file("reply." + fmt, data))
    return [t for t in tables if len(t)]


def fetch_gaia_photometry(source_id, path):
    """Save Gaia's per-visit G-band photometry as a CSV with full Julian dates."""
    # The archive renamed its layouts in 2026 (release 3.15): INDIVIDUAL became DATAMODEL_STANDARD
    # (one row per transit and band) and RAW became DATAMODEL_GAIA (one row per source, transits in
    # arrays). Each is asked for straight from the data server first, then through astroquery, and
    # the old names last, for older archive mirrors.
    frames, errors = [], []
    attempts = [("direct", "DATAMODEL_STANDARD", "csv"), ("direct", "DATAMODEL_STANDARD", "votable"),
                ("direct", "DATAMODEL_GAIA", "votable"), ("astroquery", "DATAMODEL_STANDARD", "csv"),
                ("astroquery", "DATAMODEL_GAIA", "votable"), ("astroquery", "INDIVIDUAL", "csv")]
    for how, structure, fmt in attempts:
        try:
            if how == "direct":
                found = _gaia_datalink(source_id, structure, fmt)
            else:
                from astroquery.gaia import Gaia
                got = Gaia.load_data(ids=[int(source_id)], data_release="Gaia DR3",
                                     retrieval_type="EPOCH_PHOTOMETRY", data_structure=structure, format=fmt)
                found = []
                for items in got.values():
                    for item in (items if isinstance(items, list) else [items]):
                        tab = item.to_table() if hasattr(item, "to_table") else item
                        if len(tab):
                            found.append(tab.to_pandas())
        except Exception as exc:
            text = str(exc).strip()
            errors.append(f"{how} {structure}/{fmt}: " + (text.splitlines()[-1][:100] if text else type(exc).__name__))
            continue
        if not found:
            errors.append(f"{how} {structure}/{fmt}: empty reply")
            continue
        frames = [_flatten_epochs(t) for t in found]
        break
    if not frames:
        if any(e.startswith("direct") and e.endswith("empty reply") for e in errors):
            raise RuntimeError("Gaia DR3 has none for this star (it published per-visit photometry only for "
                               "stars it classified as variable); carrying on without it")
        raise RuntimeError("Gaia returned no epoch photometry (" + "; ".join(errors[:3]) + ")")
    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).lower() for c in df.columns]
    if "band" in df:
        band = df["band"].astype(str).str.upper().str.strip().str.replace("B'", "").str.replace("'", "")
        df = df[band == "G"]
    if "rejected_by_photometry" in df:
        df = df[~df["rejected_by_photometry"].astype(str).str.lower().isin(("true", "1"))]

    def pick(*names):
        return next((n for n in names if n in df.columns), None)

    tcol = pick("time", "g_transit_time", "obs_time", "transit_time")
    fcol = pick("flux", "g_transit_flux", "g_flux")
    ecol = pick("flux_error", "g_transit_flux_error", "flux_err", "g_flux_error")
    if tcol is None or fcol is None:
        raw = Path(path).with_name("gaia_epoch_photometry_raw.csv")
        df.to_csv(raw, index=False)
        raise RuntimeError(f"unrecognised epoch-photometry columns ({', '.join(df.columns[:12])}); "
                           f"the reply is saved as {raw.name}")
    out = pd.DataFrame({"BJD": pd.to_numeric(df[tcol], errors="coerce") + 2455197.5,   # Gaia's time zero
                        "FLUX": pd.to_numeric(df[fcol], errors="coerce"),
                        "FLUX_ERR": pd.to_numeric(df[ecol], errors="coerce") if ecol else np.nan})
    out = out.dropna(subset=["BJD", "FLUX"]).sort_values("BJD")
    if not len(out):
        raise RuntimeError("Gaia's epoch photometry had no usable G-band points")
    out.to_csv(path, index=False)


# ----------------------------------------------------------------------------
# Part A: the eclipse's shape
# ----------------------------------------------------------------------------
def companion_radius(mass):
    """Radius of a low-mass star or brown dwarf from its mass (solar units, approximate)."""
    m = [0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20]
    r = [0.095, 0.100, 0.12, 0.17, 0.22, 0.30, 0.39, 0.48, 0.58, 0.67, 0.77, 0.87, 1.00, 1.25]
    return np.interp(mass, m, r)


def limb_darkening(teff):
    """Approximate quadratic limb-darkening coefficients in the TESS band."""
    t = [3500, 4000, 5000, 5500, 6000, 6500, 7500]
    return (float(np.interp(teff, t, [0.35, 0.45, 0.42, 0.38, 0.33, 0.28, 0.22])),
            float(np.interp(teff, t, [0.30, 0.25, 0.23, 0.24, 0.25, 0.26, 0.27])))


def blocked_light(k, u1, u2, nz=500, nr=500):
    """Fraction of a limb-darkened star's light hidden by a disc of radius k.

    Returns (separations z, fraction blocked), z in stellar radii from 0 to 1 + k.
    The star is cut into thin rings; for each ring the hidden part of its
    circumference follows from where two circles intersect.
    """
    r = (np.arange(nr) + 0.5) / nr
    mu = np.sqrt(1.0 - r**2)
    weight = (1.0 - u1 * (1 - mu) - u2 * (1 - mu) ** 2) * r
    weight /= weight.sum()
    z = np.linspace(1e-6, 1.0 + k, nz)
    cos_half = (r[None, :] ** 2 + z[:, None] ** 2 - k**2) / (2.0 * r[None, :] * z[:, None])
    hidden = np.arccos(np.clip(cos_half, -1.0, 1.0)) / np.pi
    return z, hidden @ weight


def eclipse_extent(x, fn, sigma, cadence, guess):
    """Half-length of the eclipse (days) as the data show it: the run of binned points,
    outward from the deepest, that lie more than 3 sigma below the baseline."""
    width = max(cadence, 10.0 / 1440)
    key = np.floor(x / width).astype(int)
    bins = np.unique(key)
    depth, err, mid = [], [], []
    for b in bins:
        sel = key == b
        depth.append(float(np.mean(1.0 - fn[sel])))
        err.append(sigma / np.sqrt(sel.sum()))
        mid.append((b + 0.5) * width)
    depth, err, mid = np.array(depth), np.array(err), np.array(mid)
    if len(depth) == 0:
        return guess
    deep = depth > 3 * err
    near = np.abs(mid) < guess
    if not (deep & near).any():
        return guess
    i0 = int(np.argmax(np.where(near, depth, -np.inf)))
    lo = hi = i0
    while lo - 1 >= 0 and deep[lo - 1] and mid[lo] - mid[lo - 1] < 1.5 * width:
        lo -= 1
    while hi + 1 < len(mid) and deep[hi + 1] and mid[hi + 1] - mid[hi] < 1.5 * width:
        hi += 1
    return float(max(abs(mid[lo]), abs(mid[hi])) + width)


def eclipse_points(df, t0, dur):
    """The known eclipse, cleaned up for fitting.

    The eclipse's own length is measured from the data, so that a --duration
    shorter than the real eclipse does not leave its wings in the baseline or cut
    them off the centre (an earlier version used 0.6 x --duration for both, which
    moved the centre by a minute or more when --duration was too short).

    Returns (refined centre, offsets x in days, normalised flux, point scatter, cadence).
    """
    tc, half = t0, 0.6 * dur
    for _ in range(5):
        sel = np.abs(df.t - tc) <= max(2.5 * dur, 3.0 * half)
        x, f = (df.t[sel] - tc).to_numpy(), df.f[sel].to_numpy()
        base = np.abs(x) > 1.1 * half
        if base.sum() < 10:
            base = np.abs(x) > 0.6 * dur
        slope, level = np.polyfit(x[base], f[base], 1)
        fn = f / (level + slope * x)
        sigma = 1.4826 * np.median(np.abs(fn[base] - np.median(fn[base])))
        cadence = float(np.median(np.diff(np.sort(x))))
        half = max(eclipse_extent(x, fn, sigma, cadence, max(half, 0.6 * dur)), 0.3 * dur)
        depth = np.clip(1.0 - fn, 0, None) * (np.abs(x) <= 1.1 * half)
        if depth.sum() > 0:
            tc = tc + float(np.sum(x * depth) / depth.sum())      # centre of the missing light
    return tc, x, fn, float(sigma), cadence


def centre_error(x, fn, sigma, dur):
    """1-sigma error, in days, of an eclipse centre found as the centre of the missing light,
    over the eclipse's measured length."""
    cadence = float(np.median(np.diff(np.sort(x))))
    half = max(eclipse_extent(x, fn, sigma, cadence, 0.6 * dur), 0.3 * dur)
    inside = np.abs(x) <= 1.1 * half
    lost = np.clip(1.0 - fn[inside], 0, None).sum()
    return float(sigma * np.sqrt(np.sum(x[inside] ** 2)) / lost) if lost > 0 else 0.1 * dur


def eclipse_profile(x, fn, dur):
    """Depth as a function of time from mid-eclipse, measured from the data, out to the
    eclipse's measured end (not to 0.6 x --duration, which cut the wings off when --duration
    was shorter than the eclipse)."""
    base = np.abs(x) > 1.2 * dur
    sigma = 1.4826 * np.median(np.abs(fn[base] - np.median(fn[base]))) if base.sum() > 10 else 0.001
    half = max(eclipse_extent(x, fn, sigma, float(np.median(np.diff(np.sort(x)))), 0.6 * dur), 0.3 * dur)
    edges = np.linspace(0, 1.1 * half, 11)
    mid, val = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (np.abs(x) >= a) & (np.abs(x) < b)
        if sel.sum() >= 2:
            mid.append(0.5 * (a + b))
            val.append(max(float(np.median(1.0 - fn[sel])), 0.0))
    mid, val = np.array(mid), np.maximum.accumulate(np.array(val)[::-1])[::-1]   # falls outward
    return lambda offset: np.interp(np.abs(offset), mid, val, right=0.0)


def _chi2_duration_impact(x, xs, fn, weight, b, half, lengths, z_grid, hidden, dilution):
    """Chi-squared for every total eclipse length and impact parameter, one companion size.

    `half` is half the path the companion's centre travels while the discs overlap
    (stellar radii), so the crossing speed for a given length is 2 * half / length.
    """
    out = np.empty((len(lengths), len(b)))
    for i, length in enumerate(lengths):
        speed = 2.0 * half / length
        z = np.sqrt(b[:, None, None] ** 2 + (speed[:, None, None] * xs[None, :, :]) ** 2)
        model = 1.0 - dilution * np.interp(z, z_grid, hidden, right=0.0).mean(axis=2)
        s11 = (model**2).sum(axis=1)                              # best straight-line baseline
        s12 = (x * model**2).sum(axis=1)
        s22 = (x**2 * model**2).sum(axis=1)
        y1, y2 = (fn * model).sum(axis=1), (fn * x * model).sum(axis=1)
        det = s11 * s22 - s12**2
        c0, c1 = (y1 * s22 - y2 * s12) / det, (y2 * s11 - y1 * s12) / det
        resid = fn[None, :] - (c0[:, None] + c1[:, None] * x[None, :]) * model
        out[i] = weight * (resid**2).sum(axis=1)
    return out


def shape_variants(n_m, r1, r2, light2, third_light=0.0, radius_err=0.0):
    """The versions of the system part A averages over, for every companion mass.

    Third light (another star's share of the light, 0 to `third_light`) only dilutes the
    eclipse here; the split of light between the two stars is left as it is (triple_fit.py
    treats a third star fully). The eclipsed star's radius is uncertain by `radius_err`
    (fractional, 1 sigma); a three-point Gauss-Hermite rule (nodes 0, +/-sqrt(3) sigma,
    weights 2/3, 1/6, 1/6) carries that through. The companion's radius scales with it,
    since both come from the same isochrone or catalogue.

    Returns a dict of flat arrays over (variant, mass): mass index, eclipsed-star radius,
    radius ratio, eclipsed star's share of the light, and weight.
    """
    l3s = np.linspace(0.0, third_light, 3) if third_light > 0 else np.zeros(1)
    l3w = np.full(len(l3s), 1.0 / len(l3s))
    if radius_err > 0:
        scales, sw = 1.0 + np.sqrt(3.0) * radius_err * np.array([-1.0, 0.0, 1.0]), np.array([1, 4, 1]) / 6.0
    else:
        scales, sw = np.ones(1), np.ones(1)
    out = {"mass": [], "r1": [], "k": [], "dilute": [], "w": []}
    for l3, wl in zip(l3s, l3w):
        for sc, ws in zip(scales, sw):
            out["mass"].append(np.arange(n_m))
            out["r1"].append(r1 * sc)
            out["k"].append(r2 / r1)
            out["dilute"].append((1.0 - light2) * (1.0 - l3))
            out["w"].append(np.full(n_m, wl * ws))
    return {k: np.concatenate(v) for k, v in out.items()}


def fit_shape(x, fn, sigma, cadence, masses, star, dur, say=None, fineness=0.7, budget=40000,
              variants=None):
    """Likelihood of the eclipse's shape for every crossing speed and companion mass.

    The fit is done in two passes. A coarse pass over eclipse length and impact
    parameter finds where the good fits lie and how sharply the data pin them
    down. A fine pass then covers that region with steps small enough that
    chi-squared changes by less than about 0.5 from one grid point to the next.
    Without that, a well-measured eclipse falls between grid points and the
    answer collapses onto whichever point happens to lie nearest.

    `variants` (from shape_variants) lists versions of the system for every mass
    (third light, the eclipsed star's radius); their likelihoods are averaged. So that
    versions with different stellar radii can be combined, speeds are physical: solar
    radii per day.

    Returns a dict: log_speed (fine grid, solar radii per day), like[speed, mass]
    averaged over impact parameter and variants, graze[speed, mass] (the part of
    `like` from grazing geometries), peak[speed, mass] (best single impact parameter
    and variant), and the best fit.
    """
    u1, u2 = limb_darkening(float(np.median(np.atleast_1d(star["teff"]))))
    n_m = len(masses)
    if variants is None:
        r1 = np.broadcast_to(np.asarray(star["rad"], float), (n_m,))
        r2 = np.asarray(star.get("r2", companion_radius(masses)), float)
        light2 = np.asarray(star.get("light2", (masses / star["mass"]) ** 4.5
                                     / (1 + (masses / star["mass"]) ** 4.5)), float)
        variants = shape_variants(n_m, r1, r2, light2)
    v_mass, v_r1, ks, dilute, v_w = (variants[k] for k in ("mass", "r1", "k", "dilute", "w"))
    smear = np.linspace(-0.5, 0.5, 5) * cadence if cadence > 0.004 else np.zeros(1)
    xs = x[:, None] + smear[None, :]                              # long exposures blur the shape
    weight = 1.0 / sigma**2
    # The light curve's shape depends only on the radius ratio and the dilution; the eclipsed
    # star's radius only turns speeds into solar radii per day. So each distinct shape is fitted
    # once, and the variants that share it differ only in that conversion.
    shapes, u_idx = np.unique(np.round(np.c_[ks, dilute], 12), axis=0, return_inverse=True)
    u_idx = np.ravel(u_idx)
    v_ks, v_dil = ks, dilute
    ks, dilute = shapes[:, 0], shapes[:, 1]
    curves = [blocked_light(k, u1, u2) for k in ks]
    masses = ks                                                   # the loops below run over shapes

    def run(lengths, n_b):
        frac = (np.arange(n_b) + 0.5) / n_b
        cube = np.empty((len(masses), len(lengths), n_b))
        for j, k in enumerate(ks):
            cube[j] = _chi2_duration_impact(x, xs, fn, weight, frac * (1 + k),
                                            (1 + k) * np.sqrt(1 - frac**2), lengths,
                                            curves[j][0], curves[j][1], dilute[j])
        return cube, frac

    # ---- coarse pass ----
    step = 0.03
    coarse = dur * np.exp(np.arange(np.log(0.5), np.log(4.0), step))
    cube, frac = run(coarse, 60)
    j0, i0, b0 = np.unravel_index(np.argmin(cube), cube.shape)

    # How fast chi-squared grows when the length or the impact parameter is nudged.
    def growth(d_length, d_frac):
        f2 = np.array([min(frac[b0] + d_frac, 0.999)])
        base = np.array([frac[b0]])
        k = ks[j0]
        z = lambda fr, length: np.sqrt((fr * (1 + k)) ** 2 + (2 * (1 + k) * np.sqrt(1 - fr**2)
                                                             / length * xs) ** 2)
        m_a = 1 - dilute[j0] * np.interp(z(base[0], coarse[i0]), *curves[j0], right=0.0).mean(axis=1)
        m_b = 1 - dilute[j0] * np.interp(z(f2[0], coarse[i0] * (1 + d_length)), *curves[j0],
                                         right=0.0).mean(axis=1)
        return weight * float(((m_a - m_b) ** 2).sum())

    s_len = growth(0.01, 0.0) / 0.01**2
    s_imp = growth(0.0, 0.005) / 0.005**2
    fine_step = float(np.clip(fineness / np.sqrt(max(s_len, 1e-9)), 4e-4, step))
    n_b = int(np.clip(np.ceil(np.sqrt(max(s_imp, 1e-9)) / fineness), 60, 800))

    # ---- fine pass over the lengths that can fit ----
    floor = cube.min(axis=(0, 2))
    keep = np.where(floor <= cube.min() + s_len * step**2 + 30.0)[0]
    lo, hi = max(keep.min() - 1, 0), min(keep.max() + 1, len(coarse) - 1)
    n_len = int(np.ceil(np.log(coarse[hi] / coarse[lo]) / fine_step)) + 1
    if n_len * n_b > budget:                                      # keep the run time bounded
        scale = np.sqrt(n_len * n_b / budget)
        n_len, n_b = int(n_len / scale), int(n_b / scale)
    lengths = np.exp(np.linspace(np.log(coarse[lo]), np.log(coarse[hi]), max(n_len, 8)))
    fine_step = float(np.log(lengths[1] / lengths[0]))
    if say:
        say(f"    fine grid: {len(lengths)} eclipse lengths from {lengths[0] * 24:.2f} to "
            f"{lengths[-1] * 24:.2f} hours, {n_b} impact parameters")
    cube, frac = run(lengths, n_b)

    best = float(cube.min())
    dof = max(len(x) - 4, 1)
    rescale = max(best / dof, 1.0)                                # never trust a poor best fit
    like = np.exp(-0.5 * (cube - best) / rescale)                 # [mass, length, impact]
    j0, i0, b0 = np.unravel_index(np.argmin(cube), cube.shape)

    # ---- move from eclipse length to crossing speed (solar radii per day), one grid for all ----
    half = (1 + v_ks)[:, None] * np.sqrt(1 - frac**2)[None, :]    # [variant, impact], stellar radii
    used = like.max(axis=1)[u_idx] > 1e-12                        # [variant, impact]
    phys = v_r1[:, None]                                          # stellar radii -> solar radii
    v_lo = (2.0 * half * phys / lengths[-1])[used].min()
    v_hi = (2.0 * half * phys / lengths[0])[used].max()
    log_v = np.arange(np.log(v_lo) - 2 * fine_step, np.log(v_hi) + 2 * fine_step, 0.5 * fine_step)
    out_like = np.zeros((len(log_v), n_m))
    out_graze = np.zeros_like(out_like)
    out_peak = np.zeros_like(out_like)
    peak_b = np.zeros_like(out_like)                              # at the peak: centre distance, stellar radii
    peak_v = np.zeros(out_like.shape, int)                        # ...and which variant
    log_len = np.log(lengths)
    for j, k in enumerate(v_ks):
        grazing = frac * (1 + k) > 1 - k
        jm, ju = int(v_mass[j]), int(u_idx[j])
        for b in np.where(used[j])[0]:
            here = np.interp(log_v, (np.log(2.0 * half[j, b] * v_r1[j]) - log_len)[::-1], like[ju, ::-1, b],
                             left=0.0, right=0.0)
            out_like[:, jm] += v_w[j] * here
            better = here > out_peak[:, jm]
            peak_b[better, jm], peak_v[better, jm] = frac[b] * (1 + k), j
            out_peak[:, jm] = np.maximum(out_peak[:, jm], here)
            if grazing[b]:
                out_graze[:, jm] += v_w[j] * here
    # the best fit, quoted for the variant with the central radius among those with that shape
    mine = np.where(u_idx == j0)[0]
    jv = int(mine[np.argmax(v_w[mine])])
    speed_r1 = float(2.0 * half[jv, b0] / lengths[i0])
    return {"log_speed": log_v, "like": out_like / n_b, "graze": out_graze / n_b,
            "peak": out_peak, "best": best, "dof": dof, "rescale": rescale,
            "mass_index": int(v_mass[jv]), "variant": jv,
            "speed": speed_r1 * float(v_r1[jv]),                     # solar radii per day
            "speed_r1": speed_r1, "k_best": float(v_ks[jv]), "dilute_best": float(v_dil[jv]),
            "r1_best": float(v_r1[jv]), "frac_b": float(frac[b0]), "length": float(lengths[i0]),
            # for every speed and mass, the best shape there: centre distance and variant (k, light, radius)
            "peak_b": peak_b, "peak_variant": peak_v, "v_k": np.asarray(v_ks, float),
            "v_dil": np.asarray(v_dil, float), "v_r1": np.asarray(v_r1, float)}


def spread_over(log_grid, values, lo, hi):
    """Average of `values` (given on the even grid `log_grid`) between lo and hi, elementwise."""
    step = log_grid[1] - log_grid[0]
    edges = np.r_[log_grid - 0.5 * step, log_grid[-1] + 0.5 * step]
    total = np.r_[0.0, np.cumsum(values) * step]
    return (np.interp(hi, edges, total) - np.interp(lo, edges, total)) / np.maximum(hi - lo, 1e-12)


def speed_factor_prior(s_grid, circular, n_e=20):
    """How likely each crossing speed is, relative to a circular orbit's.

    On an eccentric orbit the companion crosses the star faster near closest
    approach and slower far away. The factor is (1 + e sin w) / sqrt(1 - e^2)
    for eccentricity e and orientation w. All e from 0 to 0.95 and all w are
    taken as equally likely, weighted by the chance of eclipsing at all, which
    is higher where the two stars are closer.

    Returns weights[eccentricity bin, speed bin] and the eccentricity bin centres.
    """
    e_edges = np.linspace(0.0, 0.95, n_e + 1)
    e_mid = 0.5 * (e_edges[:-1] + e_edges[1:])
    weights = np.zeros((n_e, len(s_grid)))
    if circular:
        weights[0, np.argmin(np.abs(np.log(s_grid)))] = 1.0
        return weights, e_mid
    log_edges = np.log(s_grid)
    step = log_edges[1] - log_edges[0]
    w = np.linspace(0, 2 * np.pi, 361)[:-1]
    for i, e in enumerate(np.linspace(0.0, 0.95, 20 * n_e + 1)[:-1] + 0.95 / (40 * n_e)):
        factor = (1 + e * np.sin(w)) / np.sqrt(1 - e**2)
        chance = (1 + e * np.sin(w)) / (1 - e**2)
        idx = np.round((np.log(factor) - log_edges[0]) / step).astype(int)
        ok = (idx >= 0) & (idx < len(s_grid))
        np.add.at(weights[min(i // 20, n_e - 1)], idx[ok], chance[ok])
    return weights / (20 * 360), e_mid


# ----------------------------------------------------------------------------
# Part B extra: Gaia's individual brightness measurements
# ----------------------------------------------------------------------------
def load_gaia(path):
    """Read Gaia's per-visit photometry. Nothing is dropped for having a large error:
    a visit made during an eclipse is exactly the kind that tends to have one."""
    raw = pd.read_csv(path)
    cols = list(raw.columns)
    tcol, fcol = find_column(cols, TIME_NAMES), find_column(cols, FLUX_NAMES)
    ecol = find_column(cols, ("FLUX_ERR", "FLUX_ERROR"))
    if tcol is None or fcol is None:
        return pd.DataFrame(columns=["t", "f", "e"])
    df = pd.DataFrame({"t": pd.to_numeric(raw[tcol], errors="coerce"),
                       "f": pd.to_numeric(raw[fcol], errors="coerce"),
                       "e": pd.to_numeric(raw[ecol], errors="coerce") if ecol else 0.0})
    df = df[np.isfinite(df.t) & np.isfinite(df.f) & (df.f > 0)].sort_values("t")
    df["e"] = (df.e / df.f).fillna(0.0)
    df["f"] /= df.f.median()
    return df.reset_index(drop=True)


def gaia_groups(df):
    """Gather Gaia visits made within hours of each other.

    Gaia sees a star twice 1.8 hours apart, and often again six hours later.
    The star's own slow variations barely change over so short a time, so the
    visits in a group can be compared with each other: an eclipse shows up as
    one visit fainter than its companions.

    Returns (groups, noise). Each group is (times, brightness relative to the
    group, variance). The noise is how much visits in a group really differ,
    taken from the data, with the tail of the distribution counted in.
    """
    t = df.t.to_numpy()
    label = np.r_[0, np.cumsum(np.diff(t) > GAIA_GROUP)]
    raw, resid = [], []
    for g in np.unique(label):
        sel = label == g
        n = int(sel.sum())
        if n < 2:
            continue
        y = df.f.to_numpy()[sel]
        y = y / y.mean()
        raw.append((t[sel], y, df.e.to_numpy()[sel]))
        resid.append((y - 1.0) / np.sqrt((n - 1) / n))
    if not raw:
        return [], 0.0
    resid = np.abs(np.concatenate(resid))
    noise = max(1.4826 * np.median(resid), np.quantile(resid, 0.95) / 1.96)
    typical = np.median(np.concatenate([e for _, _, e in raw]))
    groups = [(tt, y, noise**2 + np.maximum(e**2 - typical**2, 0.0)) for tt, y, e in raw]
    return groups, float(noise)


def gaia_dips(groups, peak):
    """Groups in which one visit is much fainter than the rest: possible eclipses.

    Returns a list of (group number, time, drop, error).
    """
    found = []
    for g, (t, y, var) in enumerate(groups):
        for i in range(len(t)):
            rest = np.arange(len(t)) != i
            w = 1.0 / var[rest]
            level = float(np.sum(w * y[rest]) / w.sum())
            drop, err = level - y[i], float(np.sqrt(var[i] + 1.0 / w.sum()))
            if drop / err >= GAIA_DIP_SIGMA and 0.25 * peak <= drop <= 2.0 * peak:
                found.append((g, float(t[i]), float(drop), err))
    return found


def accumulate_gaia(periods, groups, t0, dur, profile, keep=()):
    """Evidence from Gaia's visits, for every period.

    For each period that puts an eclipse on any visit of a group, the eclipse
    profile predicts how much fainter each visit should be than the others.
    The measured differences are compared with that pattern. If all visits of
    a group would be dimmed equally there is nothing to compare, and the group
    says nothing about that period.

    Returns (summed ln LR, {group number: that group's ln LR} for groups in `keep`).
    """
    lnlr = np.zeros(len(periods), dtype=np.float32)
    kept = {}
    reach = 0.6 * dur + TIME_SLACK * dur
    for g, (t, y, var) in enumerate(groups):
        pieces = []
        for ti in t:
            k_a, k_b = (ti - t0) / periods[0], (ti - t0) / periods[-1]
            for k in range(int(np.floor(min(k_a, k_b))) - 1, int(np.ceil(max(k_a, k_b))) + 2):
                if k == 0:
                    continue
                lo, hi = sorted(((ti - t0 - reach) / k, (ti - t0 + reach) / k))
                i0, i1 = np.searchsorted(periods, [lo, hi])
                if i1 > i0:
                    pieces.append(np.arange(i0, i1))
        if not pieces:
            continue
        idx = np.unique(np.concatenate(pieces))
        p = periods[idx]
        offset = (t[:, None] - t0 + 0.5 * p[None, :]) % p[None, :] - 0.5 * p[None, :]
        w = (1.0 / var)[:, None]
        dim = 1.0 - y[:, None]
        dim = dim - (w * dim).sum(axis=0) / w.sum(axis=0)          # measured, relative to the group
        best = np.full(len(idx), -np.inf)
        # The eclipse may be a little early or late: keep the most eclipse-like reading.
        for shift in (-TIME_SLACK * dur, -0.5 * TIME_SLACK * dur, 0.0,
                      0.5 * TIME_SLACK * dur, TIME_SLACK * dur):
            pred = profile(offset + shift)
            contrast = pred.max(axis=0) - pred.min(axis=0)
            pattern = pred - (w * pred).sum(axis=0) / w.sum(axis=0)
            power = (w * pattern**2).sum(axis=0)
            ok = (contrast > 0) & (power > 0)
            safe = np.where(ok, power, 1.0)
            scale = (w * dim * pattern).sum(axis=0) / safe          # 1 if the eclipse is there
            c = np.where(ok, contrast, 1.0)
            value = np.where(ok, log_odds(scale * c, c / np.sqrt(safe), c), 0.0)
            best = np.maximum(best, value)
        best = np.maximum(best, EPOCH_FLOOR)
        lnlr[idx] += best.astype(np.float32)
        if g in keep:
            one = np.zeros(len(periods), dtype=np.float32)
            one[idx] = best
            kept[g] = one
    return lnlr, kept


def accumulate_visits(periods, df, t0, dur, profile, skip=(), decisive=-4.5):
    """Evidence from a survey that sees the star a few times a night (ATLAS, ASAS-SN, ZTF).

    Such data are too sparse for the sliding window, so each night is judged on
    its own: for every period that puts an eclipse near that night's visits, the
    eclipse profile predicts how much fainter each visit should be than the
    star's normal brightness (1, after normalising), and the visits are compared
    with that. A night counts for no more than EPOCH_FLOOR against a period,
    since visits taken minutes apart share the same night's conditions.

    Returns (ln LR, clear misses, nights with a deep dip, typical error per visit).
    """
    t, y = df.t.to_numpy(float), df.f.to_numpy(float)
    scatter = 1.4826 * np.median(np.abs(y - np.median(y)))
    err = np.maximum(df.e.to_numpy(float), scatter)
    lnlr = np.zeros(len(periods), dtype=np.float32)
    missed = np.zeros(len(periods), dtype=bool)
    peak = float(profile(0.0))
    reach = 0.6 * dur + TIME_SLACK * dur
    breaks = np.r_[0, np.where(np.diff(t) > 0.3)[0] + 1, len(t)]
    dips = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        tt, yy, ee = t[a:b], y[a:b], err[a:b]
        if any(abs(tt.mean() - x) < 2 * dur for x in (t0, *skip)):
            continue                                   # the known eclipses are rules, not evidence
        deep = ((1 - yy) / ee >= 5) & (1 - yy >= 0.3 * peak) & (1 - yy <= 1.5 * peak)
        fainter = ((1 - yy) / ee >= 3) & (1 - yy <= 1.5 * peak)
        if deep.sum() >= 2 and fainter.sum() >= 3:
            # A real eclipse dims every visit taken within minutes by about the same amount. Faint
            # visits next to normal ones taken minutes earlier or later are bad frames, not an eclipse.
            steady = True
            for i in np.where(deep)[0]:
                close = np.abs(tt - tt[i]) <= 0.1 * dur
                allowed = 4 * np.hypot(ee, ee[i]) + peak * np.abs(tt - tt[i]) / (0.5 * dur)
                if np.any(close & (np.abs(yy - yy[i]) > allowed)):
                    steady = False
            dips.append((tt, yy, ee, steady))
        pieces = []
        for ti in (tt.min(), tt.max()):
            k_a, k_b = (ti - t0) / periods[0], (ti - t0) / periods[-1]
            for k in range(int(np.floor(min(k_a, k_b))) - 1, int(np.ceil(max(k_a, k_b))) + 2):
                if k == 0:
                    continue
                lo, hi = sorted(((ti - t0 - reach) / k, (ti - t0 + reach) / k))
                i0, i1 = np.searchsorted(periods, [lo, hi])
                if i1 > i0:
                    pieces.append(np.arange(i0, i1))
        if not pieces:
            continue
        idx = np.unique(np.concatenate(pieces))
        p = periods[idx]
        offset = (tt[:, None] - t0 + 0.5 * p[None, :]) % p[None, :] - 0.5 * p[None, :]
        w = (1.0 / ee**2)[:, None]
        dim = (1.0 - yy)[:, None]
        best = np.full(len(idx), -np.inf)
        for shift in (-TIME_SLACK * dur, -0.5 * TIME_SLACK * dur, 0.0, 0.5 * TIME_SLACK * dur, TIME_SLACK * dur):
            pred = profile(offset + shift)
            power = (w * pred**2).sum(axis=0)
            c = pred.max(axis=0)
            ok = (c > 0.2 * peak) & (power > 0)
            safe = np.where(ok, power, 1.0)
            scale = (w * dim * pred).sum(axis=0) / safe            # 1 if the eclipse is there
            cc = np.where(ok, c, 1.0)
            value = np.where(ok, log_odds(scale * cc, cc / np.sqrt(safe), cc), 0.0)
            best = np.maximum(best, value)
        best = np.maximum(best, EPOCH_FLOOR)
        lnlr[idx] += best.astype(np.float32)
        missed[idx] |= best <= decisive
    return lnlr, missed, dips, float(np.median(err))


def explains(periods, night, t0, dur, profile):
    """True for the periods that put an eclipse on a night's faint visits."""
    tt, yy, ee = night[:3]
    faint = tt[np.argmax((1 - yy) / ee)]
    offset = (faint - t0 + 0.5 * periods) % periods - 0.5 * periods
    peak = float(profile(0.0))
    good = np.zeros(len(periods), dtype=bool)
    for shift in (-TIME_SLACK * dur, 0.0, TIME_SLACK * dur):
        good |= profile(offset + shift) >= 0.2 * peak
    return good


def chance_support(periods, other, gap, tol, trials=2000, seed=1):
    """The best support any candidate period would collect if the Gaia dip had come at
    some other time. `other` is the evidence from everything except the dip (minus
    infinity where a period is ruled out), `gap` the time between the two eclipses and
    `tol` how well it is known. Returns the best value for each of `trials` random times.
    """
    rng = np.random.default_rng(seed)
    best = np.empty(trials)
    for i in range(trials):
        shifted = gap + rng.choice((-1, 1)) * rng.uniform(2.0, 150.0)
        n = np.arange(max(int(np.ceil(shifted / periods[-1])), 1), int(shifted / periods[0]) + 1)
        a = np.searchsorted(periods, (shifted - tol) / n)
        b = np.searchsorted(periods, (shifted + tol) / n)
        best[i] = max((other[lo:hi].max() for lo, hi in zip(a, b) if hi > lo), default=-np.inf)
    return best


# ----------------------------------------------------------------------------
# Part C: Gaia astrometry
# ----------------------------------------------------------------------------
def wobble_mas(period, m2, m1, parallax):
    """Size of the star's photocentre orbit on the sky, in milliarcseconds."""
    a_au = ((m1 + m2) * (period / 365.25) ** 2) ** (1 / 3)
    light = (m2 / m1) ** 4.5                                     # companion's share of the light
    return a_au * parallax * np.abs(m2 / (m1 + m2) - light / (1 + light))


def ruwe_kappa(model, periods, n_orient=32, seed=5):
    """Extra chi-squared Gaia's five-parameter fit is left with, per (mas)^2 of photocentre
    semi-major axis, for an edge-on (eclipsing) orbit at each period: the average over sky
    orientation, eccentricity (0-0.8), periastron angle and phase. Orbits much longer than
    Gaia's 34 months are mostly absorbed into the proper motion, and this falls."""
    rng = np.random.default_rng(seed)
    big_om, om = rng.uniform(0, 2 * np.pi, n_orient), rng.uniform(0, 2 * np.pi, n_orient)
    ecc, ph = rng.uniform(0, 0.8, n_orient), rng.uniform(0, 1, n_orient)
    out = np.empty(len(periods))
    for i, period in enumerate(periods):
        lam = [model.lam(model.along_scan(*gaia_tools.wobble_track(
            model.t, period, ecc[j], om[j], gaia_tools.DR3_REF + ph[j] * period, np.pi / 2, big_om[j], 1.0)))
            for j in range(n_orient)]
        out[i] = float(np.mean(lam))
    return out


def gaia_likelihood(period, m2, m1, gaia, soften):
    """How well a period and mass agree with the little excess scatter Gaia recorded."""
    seen = gaia["astrometric_excess_noise"] ** 2
    sig = max(gaia["astrometric_excess_noise_sig"], 0.1)
    spread = soften * max(seen / sig, 2e-4)                       # uncertainty of the squared noise
    faded = np.where(period > 1000.0, (1000.0 / period) ** 2, 1.0)   # orbits longer than the data
    added = (wobble_mas(period, m2, m1, gaia["parallax"]) * faded) ** 2 / 4.0
    return np.exp(-0.5 * (np.clip(added - seen, 0, None) / spread) ** 2), spread


# ----------------------------------------------------------------------------
# Part E: measured velocities
# ----------------------------------------------------------------------------
def fetch_lamost(ra, dec):
    """Velocities of a star from the LAMOST survey, one per night and instrument, via VizieR."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier

    found = Vizier(columns=["**"], row_limit=500).query_region(
        SkyCoord(ra * u.deg, dec * u.deg), radius=3 * u.arcsec,
        catalog=["V/146", "V/149", "V/153", "V/164", "V/156", "V/162"])
    return lamost_nights({name: table.to_pandas() for name, table in zip(found.keys(), found)})


def lamost_nights(tables):
    """Reduce LAMOST catalogue rows (a dict of table name -> rows) to one velocity per night
    and instrument, keeping the newest release of each.

    LAMOST gives the date of each spectrum but not the hour. It observes from
    China, so the hour is known to within a few: the local date number runs
    eight hours ahead of the universal one, and comparing the two narrows it.
    """
    import re

    nights = {}                                            # (spectrum number, mode) -> (release, row)
    for name, df in tables.items():
        cols = {c.lower(): c for c in df.columns}
        if "obsid" not in cols or "mjd" not in cols:
            continue
        release = int((re.findall(r"dr(\d+)", name.lower()) or ["0"])[0])
        medium = "mobsid" in cols
        wanted = ("rvbr1", "rvb1", "rvr1", "rvlasp1") if medium else ("rv",)
        if not any(c in cols for c in wanted):
            continue
        for obsid, grp in df.groupby(cols["obsid"]):
            values = None
            for c in wanted:                               # first velocity column that has numbers
                if c in cols:
                    v = pd.to_numeric(grp[cols[c]], errors="coerce").dropna().to_numpy(float)
                    v = v[np.abs(v) < 2000]
                    if len(v):
                        values = v
                        break
            if values is None:
                continue
            spread = 0.0
            if medium and "rvb1" in cols and "rvr1" in cols:   # do the two halves of the spectrum agree?
                diff = (pd.to_numeric(grp[cols["rvb1"]], errors="coerce")
                        - pd.to_numeric(grp[cols["rvr1"]], errors="coerce")).abs()
                spread = 0.5 * float(diff.median()) if diff.notna().any() else 0.0
            mjd = float(pd.to_numeric(grp[cols["mjd"]], errors="coerce").iloc[0])
            exact = None
            if "lmjm" in cols:
                # medium-resolution catalogues give the exposure's start in local (Beijing, UTC+8)
                # modified Julian minutes; add half an exposure when its length is listed
                lmjm = pd.to_numeric(grp[cols["lmjm"]], errors="coerce").dropna()
                if len(lmjm):
                    exp_s = (float(pd.to_numeric(grp[cols["exptime"]], errors="coerce").iloc[0])
                             if "exptime" in cols else 1200.0)
                    exact = float(lmjm.median()) / 1440.0 - 8.0 / 24.0 + 0.5 * exp_s / 86400.0
            later = "lmjd" in cols and float(pd.to_numeric(grp[cols["lmjd"]], errors="coerce").iloc[0]) > mjd
            hour = (0.78 if later else 0.56) if "lmjd" in cols else 0.70
            when = exact if (exact is not None and abs(exact - mjd) < 1.5) else mjd + hour
            key = (int(obsid), medium)
            if key not in nights or release >= nights[key][0]:
                nights[key] = (release, {
                    "JD": when + 2400000.5, "RV": float(np.median(values)),
                    "RV_ERR": max(4.0 if medium else 7.0, float(np.std(values)), spread),
                    "SOURCE": f"LAMOST {'medium' if medium else 'low'} resolution, spectrum {int(obsid)}"
                              + ("" if exact is not None and abs(exact - mjd) < 1.5 else ", hour assumed")})
    return sorted((row for _, row in nights.values()), key=lambda r: r["JD"])


def fetch_gaia_rv(ra, dec):
    """Gaia's summary of the star's velocity: the median of its visits and how much they spread."""
    from astroquery.gaia import Gaia

    q = f"""SELECT radial_velocity, radial_velocity_error, rv_nb_transits, rv_amplitude_robust,
                   rv_method_used, grvs_mag, phot_g_mean_mag FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {3.0 / 3600}))
            ORDER BY phot_g_mean_mag"""
    res = Gaia.launch_job(q).get_results().to_pandas()
    if len(res) == 0:
        return {}
    out = {}
    for name in ("radial_velocity", "radial_velocity_error", "rv_nb_transits", "rv_amplitude_robust",
                 "rv_method_used", "grvs_mag"):
        value = pd.to_numeric(res[name], errors="coerce").iloc[0]
        out[name] = None if not np.isfinite(value) else float(value)
    return out


def solve_kepler(mean, e):
    """Eccentric anomaly for a mean anomaly and eccentricity (arrays that broadcast)."""
    mean, e = np.broadcast_arrays(np.mod(mean, 2 * np.pi), e)
    ecc = np.where(e < 0.8, mean, np.pi)
    for _ in range(40):
        ecc = ecc - (ecc - e * np.sin(ecc) - mean) / (1.0 - e * np.cos(ecc))
    return ecc


def velocity_shape(mean, e, omega):
    """The star's line-of-sight velocity at a mean anomaly, in units of the swing a circular
    orbit would give, and apart from a constant. Positive is away from us."""
    ecc = solve_kepler(mean, e)
    true = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(ecc / 2), np.sqrt(1 - e) * np.cos(ecc / 2))
    return np.cos(true + omega) + e * np.cos(omega)


def velocity_constraint(periods, usable, which, per_bin, masses, m1, s_grid, e_mid, w_es,
                        spread3, base2, epochs, median, t0, cap=600, gaia_rv=None, light_rv2=None,
                        max_cells=3e7):
    """How well each orbit reproduces the measured velocities.

    For a trial period the eclipse time fixes where in the orbit each velocity
    was measured. The eccentricity and the crossing speed fix the orbit's
    orientation up to a mirror image, and the companion's mass fixes the size
    of the swing, so the velocities are predicted with nothing left to adjust
    except the system's overall velocity, which is fitted.

    Each epoch is (time, velocity, error, which), where `which` says whose
    velocity it is:
        "1"      the star that is eclipsed
        "2"      the companion (it moves the opposite way, faster if lighter)
        "A", "B" the two stars of a double-lined spectrum when it is not known
                 which of them is the eclipsed one; both ways round are tried
        "BLEND"  both stars' lines merged into one, as a low-resolution spectrum
                 or a survey's single-star pipeline gives: the light-weighted
                 average of the two velocities
    `median` is (velocity, error, which) for Gaia's median velocity, compared
    with the median the orbit would give.

    `gaia_rv` (a gaia_tools.GaiaRvModel), when given, replaces that comparison
    wherever the number of trial periods allows: every orbit is evaluated at Gaia's
    actual transit times, each transit's velocity is the blend of both stars' lines
    or, once the lines separate, the brighter star's alone, and both the median and
    the robust amplitude (rv_amplitude_robust) Gaia reported are compared with what
    the orbit gives. `light_rv2` is the companion's share of the light in the RVS
    band, per companion mass.

    Returns (chance[period bin, mass, speed factor, eccentricity, mirror image]
    averaged over the periods of each bin, support for each trial period, weight
    of the two mirror images, whether any bin had to be sampled, how many bins used
    the transit-level Gaia model). Mirror image 0 has closest approach before the
    eclipse, 1 after it.
    """
    n_p, n_m, n_s = spread3.shape
    n_e = len(e_mid)
    out = np.zeros((n_p, n_m, n_s, n_e, 2))                    # last axis: the two mirror images
    support = np.full(len(periods), np.nan)
    mirror = np.zeros(2)
    sampled = False
    step = np.log(periods[1] / periods[0])
    sub = np.exp(step * np.array([-1 / 3, 0.0, 1 / 3]))        # three periods inside each grid step
    root = np.sqrt(1.0 - e_mid**2)
    phases = (np.arange(240) + 0.5) / 240 * 2 * np.pi
    data = [(row[0] - t0, row[1], 1.0 / row[2] ** 2, str(row[3]).upper()) for row in epochs]
    med_kind = None
    if median:
        med_kind = str(median[2]).upper()
        err = gaia_rv.median_err if gaia_rv is not None else median[1]
        data.append((None, median[0], 1.0 / err**2, "GAIA"))
    sw = sum(d[2] for d in data)
    sv = sum(d[2] * d[1] for d in data)
    svv = sum(d[2] * d[1] ** 2 for d in data)
    kinds = sorted({d[3] for d in data if d[3] != "GAIA"} | ({med_kind} if med_kind else set()))
    # if a double-lined spectrum does not say which star is which, try both ways round
    ways = [{"A": "1", "B": "2"}, {"A": "2", "B": "1"}] if ("A" in kinds or "B" in kinds) else [{}]
    if light_rv2 is None:
        light = (masses / m1) ** 4.5                           # companion's light relative to the star's
        share = light / (1.0 + light)
    else:
        share = np.asarray(light_rv2, float)
    factor = {"1": np.ones(len(masses)), "2": -m1 / masses,   # how each kind of velocity scales
              "BLEND": (1.0 - share) - share * m1 / masses}
    dominant = np.where(share > 0.5, -m1 / masses, 1.0)       # the brighter star's velocity, scaled
    n_transit_bins = 0

    for b in range(n_p):
        idx = np.where((which == b) & usable)[0]
        cells = base2[b][:, None] * spread3[b]                 # [mass, speed factor]
        if len(idx) == 0 or cells.max() <= 0:
            continue
        scale = 1.0
        if len(idx) > cap:
            idx = idx[np.linspace(0, len(idx) - 1, cap).astype(int)]
            scale, sampled = ((which == b) & usable).sum() / cap, True
        s_use = np.where(cells.max(axis=0) > 1e-7 * cells.max())[0]
        s = s_grid[s_use]
        allowed = (w_es[:, s_use] > 0).T                       # [speed factor, eccentricity]
        x = np.clip((s[:, None] * root[None, :] - 1.0) / e_mid[None, :], -1.0, 1.0)
        omega = np.stack([np.arcsin(x), np.pi - np.arcsin(x)], axis=-1)      # two mirror images
        e3 = e_mid[None, :, None]
        f_ecl = np.pi / 2 - omega                              # where the eclipse happens
        ecc_ecl = 2.0 * np.arctan(np.sqrt((1 - e3) / (1 + e3)) * np.tan(f_ecl / 2))
        m_ecl = ecc_ecl - e3 * np.sin(ecc_ecl)

        p = (periods[idx][:, None] * sub[None, :]).ravel()
        shape = (len(p),) + omega.shape
        transits = (gaia_rv is not None and median is not None
                    and len(gaia_rv.t) * len(p) * omega.size <= max_cells)
        n_transit_bins += int(transits)
        sums = {k: [np.zeros(shape), np.zeros(shape), np.zeros(shape)] for k in kinds}   # g, g*g, v*g
        for ti, vi, wi, kind in data:
            if kind == "GAIA":
                if transits:
                    continue                                   # handled per mass below
                kind = med_kind
                g = (np.mean if (gaia_rv is not None and gaia_rv.stat == "mean") else np.median)(
                    velocity_shape(phases[:, None, None, None], e3[None], omega[None]), axis=0)[None]
            else:
                g = velocity_shape(m_ecl[None] + 2 * np.pi * ti / p[:, None, None, None], e3[None], omega[None])
            sums[kind][0] += wi * g
            sums[kind][1] += wi * g * g
            sums[kind][2] += wi * vi * g
        if transits:                                           # the star's velocity shape at Gaia's transits
            tg = gaia_rv.t - t0
            g_gaia = velocity_shape(m_ecl[None, None] + 2 * np.pi * tg[:, None, None, None, None]
                                    / p[None, :, None, None, None], e3[None, None], omega[None, None])
            w_med = 1.0 / gaia_rv.median_err**2
        c_a = svv - sv**2 / sw

        circ_swing = 212.91 * masses * (m1 + masses) ** (-2 / 3)          # km/s at a one-day period
        weight_e = w_es[:, s_use].T                                       # [speed factor, eccentricity]
        norm = (cells[:, s_use][:, :, None] * weight_e[None]).sum()
        here = np.zeros(len(idx))
        for jm in range(n_m):
            if cells[jm, s_use].max() <= 0:
                continue
            swing = circ_swing[jm] * p[:, None, None, None] ** (-1 / 3) / root[None, None, :, None]
            if transits:
                # what Gaia measures at each transit, relative to the system velocity, in units of swing
                apart = np.abs(swing[None] * g_gaia) * (1.0 + m1[jm] / masses[jm])
                w_dom = 1.0 / (1.0 + np.exp(-(apart - gaia_rv.resolve) / gaia_rv.width))
                g_eff = g_gaia * ((1 - w_dom) * factor["BLEND"][jm] + w_dom * dominant[jm])
                g_med = gaia_rv.summarise(g_eff, axis=0)
                amp = np.sqrt((swing * np.ptp(g_eff, axis=0)) ** 2 + gaia_rv.noise_range**2)
                amp_like = (np.exp(-0.5 * ((gaia_rv.amp_obs - amp) / gaia_rv.amp_err) ** 2)
                            if gaia_rv.amp_obs is not None else 1.0)
            like = 0.0
            for way in ways:
                sg = sgg = svg = 0.0
                for kind in kinds:
                    f = factor[way.get(kind, kind)][jm]
                    sg = sg + f * sums[kind][0]
                    sgg = sgg + f * f * sums[kind][1]
                    svg = svg + f * sums[kind][2]
                if transits:
                    sg = sg + w_med * g_med
                    sgg = sgg + w_med * g_med**2
                    svg = svg + w_med * median[0] * g_med
                c_b, c_c = svg - sv * sg / sw, sgg - sg**2 / sw
                like = like + np.exp(-0.5 * np.clip(c_a - 2 * swing * c_b + swing**2 * c_c, 0, 1400)) / len(ways)
            if transits:
                like = like * amp_like
            like = like * allowed[None, :, :, None]
            like = like.reshape(len(idx), len(sub), *like.shape[1:]).mean(axis=1)   # [period, s, e, mirror]
            weights = cells[jm, s_use][:, None] * weight_e                # [s, e]
            mirror += (like * weights[None, :, :, None]).sum(axis=(0, 1, 2))
            both = like.mean(axis=3)
            out[b, jm, s_use] = like.sum(axis=0) * scale / per_bin[b]
            here += (both * weights[None]).sum(axis=(1, 2))
        support[idx] = here / max(norm, 1e-300)
    return out, support, mirror, sampled, n_transit_bins


# ----------------------------------------------------------------------------
# Part F: the companion's own eclipse, when it passes behind the star
# ----------------------------------------------------------------------------
def companion_temperature(mass):
    """Rough surface temperature of an ordinary star of a given mass (solar masses)."""
    m = [0.08, 0.10, 0.16, 0.23, 0.44, 0.57, 0.70, 0.82, 0.88, 1.00, 1.08, 1.18, 1.33]
    t = [2500, 2800, 3050, 3200, 3550, 3850, 4440, 5040, 5270, 5770, 5930, 6150, 6550]
    return float(np.interp(mass, m, t))


def anomaly_from_true(true, e):
    """Mean anomaly (0 to 2 pi) at a true anomaly."""
    ecc = 2.0 * np.arctan2(np.sqrt(1 - e) * np.sin(true / 2), np.sqrt(1 + e) * np.cos(true / 2))
    return np.mod(ecc - e * np.sin(ecc), 2 * np.pi)


def prepare_curves(sets, frames, visits):
    """The light curves Part F looks through: dense ones (in the order of `sets`, so a
    gap-filling version comes after the one it fills) and sparse survey visits."""
    dense, sparse = [], []
    for s in sets:
        if "series" not in s or s["name"] not in frames:
            continue
        df = frames[s["name"]].sort_values("t")
        t = df.t.to_numpy()
        dense.append({"name": s["name"], "t": t, "x": t - np.median(t), "f": df.f.to_numpy(),
                      "w": 1.0 / df.e.to_numpy() ** 2})
    for name, df in visits:
        df = df.sort_values("t")
        e = df.e.to_numpy()
        pull = (1.0 - df.f.to_numpy()) / e
        scale = max(1.4826 * float(np.median(np.abs(pull - np.median(pull)))), 1.0)
        sparse.append({"name": name, "t": df.t.to_numpy(), "f": df.f.to_numpy(),
                       "w": 1.0 / (scale * e) ** 2})
    return dense, sparse


def _range_sums(t, columns, lo, hi):
    """Sums of each column over the points with lo <= t < hi (arrays of ranges)."""
    i0, i1 = np.searchsorted(t, lo), np.searchsorted(t, hi)
    out = []
    for c in columns:
        total = np.r_[0.0, np.cumsum(c)]
        out.append(total[i1] - total[i0])
    return out


def box_depths(curve, centres, width, keep, t0, sparse=False):
    """Depth of a box `width` long at each centre, its error, and whether it can be used.

    Dense data: the level outside the box (GUARD to REACH widths away, a straight
    line when there are points on both sides) minus the mean inside, with errors
    later stretched by `calibrate`. Sparse visits: one minus the mean inside, at
    least two visits needed. `keep` is zero for points that lie in a primary eclipse.
    """
    t, f, w = curve["t"], curve["f"], curve["w"] * keep
    one = (w > 0).astype(float)
    if sparse:
        n, sw, swf = _range_sums(t, [one, w, w * f], centres - width / 2, centres + width / 2)
        ok = n >= 2
        sw = np.where(ok, sw, 1.0)
        return 1.0 - swf / sw, 1.0 / np.sqrt(sw), ok, np.zeros(len(centres))
    x = curve["x"]
    cols = [one, w, w * x, w * x * x, w * f, w * x * f]
    inner = _range_sums(t, cols, centres - width / 2, centres + width / 2)
    left = _range_sums(t, cols, centres - REACH * width, centres - GUARD * width)
    right = _range_sums(t, cols, centres + GUARD * width, centres + REACH * width)
    n_in, n_l, n_r = inner[0], left[0], right[0]
    s, sx, sxx, sf, sxf = [a + b for a, b in zip(left[1:], right[1:])]
    ok = (n_in >= MIN_IN) & (n_l + n_r >= MIN_OUT)
    two = ok & (n_l >= 3) & (n_r >= 3)
    s = np.where(ok, s, 1.0)
    det = s * sxx - sx**2
    slope = np.where(two & (det > 0), (s * sxf - sx * sf) / np.where(det > 0, det, 1.0), 0.0)
    level = (sf - slope * sx) / s
    w_in = np.where(ok, inner[1], 1.0)
    depth = level + slope * inner[2] / w_in - inner[4] / w_in
    err = np.sqrt(1.0 / w_in + 1.0 / s)
    return depth, err, ok, np.where(two, 0.0, 1.0)


def secondary_tables(period, dense, sparse, widths, t0, dur, d_grid):
    """Evidence for or against a companion's eclipse at every place in the orbit.

    For each box width, the orbit is cut into slots STEP widths long. For every
    slot, the evidence (ln odds, floored per orbit as in part B) for an eclipse of
    each depth in d_grid is summed over every orbit the data cover. Points inside
    the primary eclipses of this period are left out of everything, and slots that
    would overlap a primary eclipse carry no evidence.

    Returns a list, one per width, of (slot length in days, table[slot, depth],
    number of orbits in which each slot was observed).
    """
    out = []
    for width in widths:
        n_slot = int(np.ceil(period / (STEP * width)))
        slot = period / n_slot
        phase = (np.arange(n_slot) + 0.5) * slot
        table = np.zeros((n_slot, len(d_grid)))
        seen = np.zeros(n_slot)
        taken_all = np.zeros(0, np.int64)
        for curve, is_sparse in [(c, False) for c in dense] + [(c, True) for c in sparse]:
            t = curve["t"]
            n_lo, n_hi = int(np.floor((t[0] - t0) / period)) - 1, int(np.ceil((t[-1] - t0) / period)) + 1
            cycles = np.arange(n_lo, n_hi + 1)
            prim = t0 + cycles * period
            j = np.clip(np.searchsorted(prim, t), 1, len(prim) - 1)
            near = np.minimum(np.abs(t - prim[j - 1]), np.abs(t - prim[j]))
            keep = (near > 0.6 * dur).astype(float)
            centres = (prim[:, None] + phase[None, :]).ravel()
            depth, err, ok, flag = box_depths(curve, centres, width, keep, t0, is_sparse)
            if not is_sparse:
                series = np.column_stack([centres[ok], depth[ok], err[ok], flag[ok]])
                calibrate(series, t0, width)
                err = err.copy()
                err[ok] = series[:, 2]
                ok &= np.isfinite(err) & (flag == 0)       # boxes beside a gap: ramps and edges, not used
                # the same stretch of sky time is counted once: an earlier version wins
                key = np.floor((centres - t0) / slot).astype(np.int64)
                ok &= ~np.isin(key, taken_all)
                taken_all = np.union1d(taken_all, key[ok])
            idx = np.where(ok)[0]
            if len(idx) == 0:
                continue
            lo = np.clip(log_odds(depth[idx, None], err[idx, None], d_grid[None, :]), EPOCH_FLOOR, SUPPORT_CAP)
            where = idx % n_slot
            np.add.at(table, where, lo)
            np.add.at(seen, where, 1)
        clash = np.minimum(phase, period - phase) < width / 2 + 0.6 * dur
        table[clash] = 0.0
        seen[clash] = 0
        out.append((slot, table, seen))
    return out


def secondary_predictions(period, m_idx, s_vals, e_vals, masses, k, light2, star_m, star_r, length,
                          ):
    """Where the companion's eclipse falls and what it looks like, for every orbit.

    Arrays are [mass, speed factor, eccentricity]; the two mirror images share
    everything except the timing, which is mirrored (phase -> period - phase).
    Returns impact parameter of the secondary, its speed (stellar radii per day),
    its length, the phase after the primary for mirror image 0, closest approach
    (days before the primary eclipse, mirror image 0), and whether the
    eccentricity can give this speed at all.
    """
    m = masses[m_idx][:, None, None]
    kk = k[m_idx][:, None, None]
    s = s_vals[None, :, None]
    e = e_vals[None, None, :]
    if np.ndim(star_m):                                   # a separate eclipsed star for each companion mass
        star_m = np.asarray(star_m)[m_idx][:, None, None]
        star_r = np.asarray(star_r)[m_idx][:, None, None]
    a_over_r = ((star_m + m) * (period / 365.25) ** 2) ** (1 / 3) * 215.032 / star_r
    v_circ = 2 * np.pi * a_over_r / period
    root = np.sqrt(1 - e**2)
    x = (s * root - 1.0) / e
    valid = np.abs(x) <= 1.0
    sin_w = np.clip(x, -1.0, 1.0)
    half_chord = s * v_circ * length / 2
    b_pri = np.sqrt(np.clip((1 + kk) ** 2 - half_chord**2, 0.0, None))
    ratio = (1 + e * sin_w) / (1 - e * sin_w)
    b_sec = b_pri * ratio
    v_sec = v_circ * (1 - e * sin_w) / root
    long = 2 * np.sqrt(np.clip((1 + kk) ** 2 - b_sec**2, 0.0, None)) / v_sec
    omega = np.arcsin(sin_w)
    m_p = anomaly_from_true(np.pi / 2 - omega, e)
    m_s = anomaly_from_true(3 * np.pi / 2 - omega, e)
    phase = np.mod(m_s - m_p, 2 * np.pi) / (2 * np.pi) * period
    before = m_p / (2 * np.pi) * period                  # closest approach this long before the eclipse
    return tuple(np.array(a) for a in np.broadcast_arrays(b_sec, v_sec, long, phase, before, valid))


SUPPORT_CAP = 3.0        # in part F one orbit's dip can support a companion eclipse by at most
                         # this (ln odds, 20 to 1): a lone dip of the right timing may be a glitch,
                         # while a missing one at a well-observed time is not, so it counts in full
MAX_RUNS_F = 60          # Part F runs only when no more period ranges than this remain open
VISIBLE = 0.005          # a companion eclipse this deep (0.5%) counts as plainly visible


def secondary_constraint(periods, open_mask, which, open_runs, masses, k, star, length, dur, t0,
                         s_grid, e_mid, w_es, before_f, sets, frames, visits, say):
    """Chance of the light curves for every orbit, given where its companion eclipse falls.

    When the companion passes behind the star it disappears in part or in full,
    and the system dims by the share of light it gives. The depth follows from
    its mass (light, size), the size of the star and how far apart the two discs
    pass; the time and length from the eccentricity and orientation. Each orbit
    is tested like a period in part B: the light curves at every predicted
    companion eclipse are compared with an eclipse of the predicted depth and
    with no eclipse.

    Returns chance[period bin, mass, speed factor, eccentricity, mirror image]
    (1 where nothing could be tested) and a dict for the report.
    """
    n_p, n_m, n_s, n_e, _ = before_f.shape
    dense, sparse = prepare_curves(sets, frames, visits)
    if "light2" in star:                                      # from the isochrone fit (or the M^4.5 split)
        light2 = np.asarray(star["light2"], float)
    else:
        light = (masses / star["mass"]) ** 4.5
        light2 = light / (1 + light)
    teff2 = star.get("teff2", [companion_temperature(m) for m in masses])
    comp = [blocked_light(1.0 / kk, *limb_darkening(t2), nz=400, nr=300)
            for kk, t2 in zip(k, teff2)]                      # in units of the companion's radius
    n_sub = 3
    e_edges = np.linspace(0.0, 0.95, n_e + 1)
    e_vals = (e_edges[:-1, None] + (np.arange(n_sub) + 0.5) / n_sub * (e_edges[1] - e_edges[0])).ravel()
    grow = 1.25
    widths = dur * np.exp(np.arange(np.log(0.25), np.log(16.0), np.log(grow)))
    d_grid = np.exp(np.linspace(np.log(2e-4), np.log(0.8), 48))
    times = [c["t"] for c in dense + sparse]
    t_min, t_max = min(t[0] for t in times), max(t[-1] for t in times)

    samples = []
    for p_lo, p_hi in open_runs:
        c0, c1 = np.searchsorted(periods, [p_lo, p_hi])
        members = np.arange(c0, c1)[open_mask[c0:c1]]
        if len(members) == 0:
            continue
        cycles = max(abs(t_max - t0), abs(t_min - t0)) / periods[members].mean()
        n_take = int(np.clip(np.ceil((periods[members[-1]] - periods[members[0]]) * cycles / (0.25 * dur)), 1, 6))
        for i in members[np.linspace(0, len(members) - 1, n_take).round().astype(int)]:
            samples.append((float(periods[i]), int(which[i]), len(members) / n_take))
    count_b = np.zeros(n_p)
    for _, b, share in samples:
        count_b[b] += share

    acc = np.zeros(before_f.shape)
    cnt = np.zeros(before_f.shape[:3])
    top = before_f.max()
    rep = {"wa": 0.0, "wb": 0.0, "vis_a": 0.0, "vis_b": 0.0, "none_a": 0.0, "faint_a": 0.0,
           "unseen_a": 0.0, "branch": np.zeros(2), "best": None,
           "offset_hist": np.zeros(400), "depth_hist": np.zeros(60), "n_orbits": 0}
    depth_edges = np.exp(np.linspace(np.log(1e-4), np.log(1.0), 61))
    u = np.linspace(-0.5, 0.5, 15)
    for period, b, share in samples:
        cells = before_f[b].sum(axis=(2, 3))
        if cells.max() <= 1e-9 * top:
            continue
        m_use = np.where(cells.max(axis=1) > 1e-6 * cells.max())[0]
        s_use = np.where(cells.max(axis=0) > 1e-6 * cells.max())[0]
        b_sec, v_sec, long, phase, before, valid = secondary_predictions(
            period, m_use, s_grid[s_use], e_vals, masses, k, light2, star["mass"], star["rad"], length)
        has = long > 0
        j = np.clip(np.round(np.log(np.where(has, long, widths[0]) / widths[0]) / np.log(grow)).astype(int),
                    0, len(widths) - 1)
        needed = np.unique(j[has])
        tables = secondary_tables(period, dense, sparse, widths[needed], t0, dur, d_grid) if len(needed) else []
        peak = np.zeros(b_sec.shape)
        expected = np.zeros(b_sec.shape)
        for a, jm in enumerate(m_use):
            zc, hc = comp[jm]
            peak[a] = light2[jm] * np.interp(b_sec[a] / k[jm], zc, hc, right=0.0)
            z = np.hypot(b_sec[a][..., None], v_sec[a][..., None] * widths[j[a]][..., None] * u) / k[jm]
            expected[a] = light2[jm] * np.interp(z, zc, hc, right=0.0).mean(axis=-1)
        peak[~has] = 0.0
        expected[~has] = 0.0
        ll = np.zeros(b_sec.shape + (2,))
        seen = np.zeros(b_sec.shape + (2,), bool)
        d_ext = np.r_[0.0, d_grid]
        for q, jw in enumerate(needed):
            slot, table, looked = tables[q]
            sel = has & (j == jw)
            d = np.clip(expected[sel], 0.0, d_grid[-1])
            pos = np.clip(np.searchsorted(d_ext, d, side="right"), 1, len(d_ext) - 1)
            frac = (d - d_ext[pos - 1]) / (d_ext[pos] - d_ext[pos - 1])
            for br, ph in enumerate((phase[sel], period - phase[sel])):
                idx = np.clip((ph / slot).astype(int), 0, len(table) - 1)
                left = np.where(pos >= 2, table[idx, np.maximum(pos - 2, 0)], 0.0)
                ll[..., br][sel] = left * (1 - frac) + table[idx, pos - 1] * frac
                seen[..., br][sel] = looked[idx] > 0
        like = np.exp(np.clip(ll, -700.0, 300.0))
        shape5 = (len(m_use), len(s_use), n_e, n_sub)
        vw = valid.reshape(shape5).astype(float)
        vw = np.where(vw.sum(axis=3, keepdims=True) > 0, vw, 1.0)
        vw = vw / vw.sum(axis=3, keepdims=True)                         # [m, s, e, sub]
        like5 = like.reshape(shape5 + (2,))
        block = np.ix_(m_use, s_use)
        acc[b][block] += share * (like5 * vw[..., None]).sum(axis=3)
        cnt[b][block] += share

        # for the report: weight of each orbit before and after this test
        prior_here = before_f[b][block] * (share / count_b[b])          # [m, s, e, mirror]
        wb = prior_here[:, :, :, None, :] * vw[..., None]               # [m, s, e, sub, mirror]
        wa = wb * like5
        pk = peak.reshape(shape5)[..., None] * np.ones(2)
        sn = seen.reshape(shape5 + (2,))
        offset = np.stack([phase, period - phase], axis=-1).reshape(shape5 + (2,))   # days after the primary
        offset = np.where(offset > period / 2, offset - period, offset)
        offset = np.where(offset < -period / 2, offset + period, offset)
        rep["wa"] += wa.sum()
        rep["wb"] += wb.sum()
        vis = (pk >= VISIBLE) & sn
        rep["vis_a"] += wa[vis].sum()
        rep["vis_b"] += wb[vis].sum()
        rep["none_a"] += wa[pk <= 0].sum()
        rep["faint_a"] += wa[(pk > 0) & (pk < VISIBLE)].sum()
        rep["unseen_a"] += wa[(pk >= VISIBLE) & ~sn].sum()
        rep["branch"] += wa.sum(axis=(0, 1, 2, 3))
        real = pk > 0
        rep["offset_hist"] += np.histogram(offset[real] / period, bins=400, range=(-0.5, 0.5),
                                           weights=wa[real])[0]
        rep["depth_hist"] += np.histogram(np.clip(pk[real], 1e-4, 0.999), bins=depth_edges, weights=wa[real])[0]
        rep["n_orbits"] += int(wb.size)
        ll5 = ll.reshape(shape5 + (2,))
        plausible = wb > 1e-6 * top * share / count_b[b]
        if plausible.any():
            i = np.unravel_index(np.argmax(np.where(plausible, ll5, -np.inf)), ll5.shape)
            if rep["best"] is None or ll5[i] > rep["best"]["ll"]:
                width = widths[j.reshape(shape5)[i[:4]]]
                rep["best"] = {"ll": float(ll5[i]), "offset": float(offset[i]), "depth": float(pk[i]),
                               "period": period, "width": float(width),
                               "expected": float(expected.reshape(shape5)[i[:4]])}
        worst = np.where(plausible & (pk >= VISIBLE), ll5, np.inf)
        if np.isfinite(worst).any():
            rep["worst_ll"] = min(rep.get("worst_ll", np.inf), float(worst.min()))
    chance = np.where(cnt[..., None, None] > 0, acc / np.maximum(cnt, 1e-300)[..., None, None], 1.0)
    rep["period"] = float(np.average([s[0] for s in samples], weights=[s[2] for s in samples])) if samples else 0.0
    rep["depth_edges"] = depth_edges
    rep["widths"] = (widths[0], widths[-1])
    rep["samples"] = len(samples)
    rep["datasets"] = [c["name"] for c in dense + sparse]
    best = rep["best"]
    if best:                                              # what each light curve shows at its times
        rows = []
        for curve, is_sparse in [(c, False) for c in dense] + [(c, True) for c in sparse]:
            t = curve["t"]
            cycles = np.arange(int(np.floor((t[0] - t0) / best["period"])) - 1,
                               int(np.ceil((t[-1] - t0) / best["period"])) + 2)
            centres = t0 + cycles * best["period"] + best["offset"]
            depth, err, ok, flag = box_depths(curve, centres, best["width"], np.ones(len(t)), t0, is_sparse)
            for c, d, e_, o, f_ in zip(centres, depth, err, ok, flag):
                if o:
                    rows.append((float(c), curve["name"], float(d), float(e_), bool(f_)))
        best["epochs"] = sorted(rows)
    return chance, rep


def report_secondary(say, rep, before, after, e_mid):
    """Print what part F found."""
    if rep["wb"] <= 0:
        say("    no orbit could be tested")
        return
    say(f"    {rep['n_orbits']:,} orbit shapes and orientations tested, against "
        + ", ".join(rep["datasets"]))
    say(f"    Before this test, {100 * rep['vis_b'] / rep['wb']:.0f}% of the probability lay on orbits whose companion "
        f"eclipse would be deeper than {100 * VISIBLE:.1f}%")
    say("    and fall where the light curves have data. "
        + (f"After it: {100 * rep['vis_a'] / rep['wa']:.0f}%." if rep["wa"] > 0 else "None of them survives."))
    best = rep["best"]
    if best and best["ll"] >= 0.5 * SUPPORT_CAP:
        say(f"    POSSIBLE COMPANION ECLIPSE: some orbits put it on a dip, {abs(best['offset']):.2f} days "
            f"{'after' if best['offset'] > 0 else 'before'} the known eclipse")
        say(f"    and about {100 * best['depth']:.1f}% deep (ln odds {best['ll']:+.0f}). Look at the light curve "
            "there before trusting it;")
        say("    noise, a ramp after a data gap or a bad stretch can do this too. What each light curve shows")
        say(f"    in a {best['width'] * 24:.1f}-hour box there (expected about {100 * best['expected']:.1f}% "
            "deep averaged over the box; errors before stretching):")
        for when, name, d, e_, edge in best.get("epochs", [])[:12]:
            say(f"      JD {when:.3f} (BTJD {when - 2457000:.3f})  {name:<14} {100 * d:+6.2f}% +/- {100 * e_:.2f}%"
                + ("   beside a gap: not used" if edge else ""))
    if rep["wa"] <= 0:
        return
    wa = rep["wa"]
    say("    Where the companion's own eclipse should be, given everything:")
    say(f"      no eclipse at all (it passes clear of the star):   {100 * rep['none_a'] / wa:5.1f}%")
    say(f"      shallower than {100 * VISIBLE:.1f}%, too faint to see:            {100 * rep['faint_a'] / wa:5.1f}%")
    say(f"      visible, but falls where there are no data:        {100 * rep['unseen_a'] / wa:5.1f}%")
    say(f"      visible where there are data:                      {100 * rep['vis_a'] / wa:5.1f}%")
    hist = rep["offset_hist"]
    if hist.sum() > 0:
        centres = (np.arange(400) + 0.5) / 400 - 0.5
        folded = np.abs(centres) * rep["period"]
        q = interval(folded, hist, (0.16, 0.5, 0.84))
        after_share = hist[centres > 0].sum() / hist.sum()
        side = "after" if after_share >= 0.5 else "before"
        sure_side = max(after_share, 1 - after_share)
        say(f"    If it happens: about {q[1]:.2f} days {side} the known eclipse (68%: {q[0]:.2f} to {q[2]:.2f} days"
            + (")" if sure_side >= 0.995 else
               f"; {100 * sure_side:.0f}% {side}, otherwise as far {'before' if side == 'after' else 'after'})"))
        edges = rep["depth_edges"]
        dq = interval(np.sqrt(edges[:-1] * edges[1:]), rep["depth_hist"], (0.16, 0.5, 0.84))
        say(f"    and {100 * dq[1]:.2f}% deep (68%: {100 * dq[0]:.2f} to {100 * dq[2]:.2f}%).")
    e_b = before.sum(axis=(0, 1, 2, 4))
    e_a = after.sum(axis=(0, 1, 2, 4))
    if e_b.sum() > 0 and e_a.sum() > 0:
        ib, ia = interval(e_mid, e_b, (0.16, 0.5, 0.84)), interval(e_mid, e_a, (0.16, 0.5, 0.84))
        say(f"    Eccentricity before this test {ib[1]:.2f} (68%: {ib[0]:.2f} to {ib[2]:.2f}); "
            f"after {ia[1]:.2f} (68%: {ia[0]:.2f} to {ia[2]:.2f})")


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def interval(values, weights, levels=(0.025, 0.16, 0.5, 0.84, 0.975)):
    """Quantiles of a distribution given on a grid, interpolated between grid points.

    Each grid value carries its weight spread across its own cell, so the cumulative
    probability is taken at the cell middles and interpolated linearly. (An earlier
    version returned the grid value where the cumulative probability first crossed
    each level, so on a coarse grid the median and a range limit could coincide.)
    """
    values, weights = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    if w.sum() <= 0:
        return [float("nan")] * len(levels)
    keep = w > 0
    first, last = np.argmax(keep), len(w) - 1 - np.argmax(keep[::-1])
    v, w = v[max(first - 1, 0):last + 2], w[max(first - 1, 0):last + 2]      # one empty cell either side
    cum = np.cumsum(w) / w.sum()
    mid = cum - 0.5 * w / w.sum()
    return [float(np.interp(q, mid, v)) for q in levels]


def at_edge(values, weights, grid_lo, grid_hi, tol=0.02):
    """Whether more than `tol` of the probability sits in the first or last cell of the allowed range."""
    values, weights = np.asarray(values, float).ravel(), np.asarray(weights, float).ravel()
    total = weights.sum()
    lo = weights[np.isclose(values, grid_lo)].sum() / total
    hi = weights[np.isclose(values, grid_hi)].sum() / total
    return lo > tol, hi > tol


def describe(say, label, post, periods, masses, star, radii=None):
    p_w, m_w = post.sum(axis=1), post.sum(axis=0)
    p = interval(periods, p_w)
    m = interval(masses, m_w)
    r = ([float(np.interp(v, masses, radii)) for v in m] if radii is not None
         else [float(companion_radius(v)) for v in m])
    say(f"  {label}")
    say(f"      period        {p[2]:6.1f} days   (68%: {p[1]:.1f} to {p[3]:.1f};  95%: {p[0]:.1f} to {p[4]:.1f})")
    lo_edge, hi_edge = at_edge(masses, m_w, masses[0], masses[-1])
    say(f"      mass          {m[2]:6.2f} Msun   (68%: {m[1]:.2f} to {m[3]:.2f};  95%: {m[0]:.2f} to {m[4]:.2f})"
        + ("   [piles up at the largest mass allowed]" if hi_edge else "")
        + ("   [piles up at the smallest mass allowed]" if lo_edge else ""))
    say(f"      radius        {r[2]:6.2f} Rsun   (68%: {r[1]:.2f} to {r[3]:.2f};  95%: {r[0]:.2f} to {r[4]:.2f})"
        f"   = {r[2] * 9.731:.1f} Jupiter radii")
    return p, m


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Joint limits on the orbit and size of an eclipsing companion.")
    p.add_argument("--tic", default="284759415")
    p.add_argument("--t0", type=float, default=3441.265, help="known eclipse time (BTJD or JD)")
    p.add_argument("--duration", type=float, default=2.5, help="eclipse length, hours")
    p.add_argument("--also", type=float, nargs="*", default=[], metavar="TIME",
                   help="times of other eclipses of the same kind that you are sure of (BTJD or JD); "
                        "the period must fit them whether or not the script finds them itself")
    p.add_argument("--fill-gaps", action="store_true",
                   help="also download other versions of each TESS sector (uncorrected flux, QLP) and use "
                        "them wherever the main light curve has no points")
    p.add_argument("--wasp", default="wasp_lc.csv", help="SuperWASP file, if present")
    p.add_argument("--extra", nargs="*", default=[],
                   help="other light curves (full JD): dense ones such as KELT, or survey visits such as "
                        "ATLAS (atlas_check.py writes them in the right form)")
    p.add_argument("--ignore-dips", action="store_true",
                   help="do not require the period to explain deep dips found in survey visits")
    p.add_argument("--rv", default=None,
                   help="velocities file with columns JD, RV, RV_ERR, SOURCE "
                        "(default: rv_epochs.csv in the output folder, filled from LAMOST)")
    p.add_argument("--min-period", type=float, default=10.0)
    p.add_argument("--max-period", type=float, default=1000.0)
    p.add_argument("--sigma", type=float, default=3.0, help="confidence for ruling a period out")
    p.add_argument("--gaia-soften", type=float, default=2.0,
                   help="make the Gaia limit this many times more lenient (default 2)")
    p.add_argument("--circular", action="store_true", help="allow only circular orbits")
    p.add_argument("--no-blend-correct", action="store_true",
                   help="take the catalogue mass and radius as the eclipsed star's own (the old behaviour) instead of "
                        "as describing the light of both stars together")
    p.add_argument("--iso", default=None,
                   help="MIST .iso.cmd isochrone for the stars (default: one in the output folder); 'auto' "
                        "downloads MIST's UBVRIplus set once and takes the file nearest --feh. Used only "
                        "while the folder has no stellar_split.json")
    p.add_argument("--feh", type=float, default=0.0, help="[Fe/H] for --iso auto (default 0)")
    p.add_argument("--nstars", type=int, default=2, choices=(2, 3),
                   help="stars in the isochrone fit to the photometry (default 2)")
    p.add_argument("--no-isochrone", action="store_true",
                   help="ignore stellar_split.json and share the catalogue's light as mass^4.5")
    p.add_argument("--eclipsed-brighter", action="store_true",
                   help="(mass^4.5 split only) require the eclipsed star to be the brighter one, as before")
    p.add_argument("--radius-err", type=float, default=None,
                   help="fractional 1-sigma uncertainty of the eclipsed star's radius, averaged over in part A "
                        "(default: from stellar_split.json, else 0.07)")
    p.add_argument("--flux-ratio", nargs=3, metavar=("BAND", "RATIO", "ERR"), default=None,
                   help="measured flux ratio of the two stars, fainter/brighter, in a band (G, BP, RP, J, H, K "
                        "or T), e.g. from a double-lined spectrum: --flux-ratio RP 0.75 0.12")
    p.add_argument("--third-light", type=float, default=0.0,
                   help="let other unresolved stars give up to this share of the light (0-0.5; default 0)")
    p.add_argument("--gost", default=None,
                   help="Gaia scan law: a GOST CSV export or gaiamock healpix FITS (default: gost*.csv in outdir)")
    p.add_argument("--no-gaia-transits", action="store_true",
                   help="compare only Gaia's median velocity, averaged over the orbit (the old behaviour)")
    p.add_argument("--gaia-amplitude", choices=["auto", "use", "ignore"], default="auto",
                   help="Gaia's robust RV amplitude: 'auto' (default) ignores it when the spectra are double-lined "
                        "(rows marked A/B/2 in rv_epochs.csv) and uses only the median, modelled as a blend")
    p.add_argument("--rvs-resolve", type=float, default=35.0,
                   help="separation (km/s) above which Gaia's velocity follows the brighter star's lines")
    p.add_argument("--skip-secondary", action="store_true",
                   help="do not test for the companion's own eclipse (part F)")
    p.add_argument("--offline", action="store_true", help="use only files already downloaded")
    p.add_argument("--outdir", default="joint_output")
    return p.parse_args()


def main():
    args = parse_args()
    say = Tee()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dur = args.duration / 24.0
    t0 = args.t0 + 2457000.0 if args.t0 < 100000 else args.t0
    tic = str(args.tic).upper().replace("TIC", "").strip()

    # ---- gather everything ----------------------------------------------------------
    say("STAR")
    star = cached(outdir / "star.json", lambda: fetch_star(tic), say, "the catalogue entry", args.offline)
    if not star or not star.get("rad") or not star.get("mass"):
        sys.exit("Need the star's radius and mass. Put them in joint_output/star.json as "
                 '{"rad": ..., "mass": ..., "teff": ..., "ra": ..., "dec": ...}.')
    say(f"  TIC {tic}: radius {star['rad']:.2f} Rsun, mass {star['mass']:.2f} Msun, "
        f"{star['teff']:.0f} K")
    gaia = cached(outdir / "gaia.json", lambda: fetch_gaia(star["ra"], star["dec"]), say,
                  "Gaia values", args.offline)
    if gaia:
        say(f"  Gaia: distance {1000 / gaia['parallax']:.0f} pc, excess noise "
            f"{gaia['astrometric_excess_noise']:.3f} mas (significance "
            f"{gaia['astrometric_excess_noise_sig']:.2f}), RUWE {gaia['ruwe']:.2f}")
    else:
        say("  No Gaia values: the astrometric limit (part C) is skipped.")

    tess_files = sorted(str(p) for p in outdir.glob("tess_s*.csv"))
    if not tess_files and not args.offline:
        say("\nDownloading TESS light curves")
        try:
            tess_files = download_tess(tic, outdir, say)
        except Exception as exc:
            say(f"  TESS download failed: {str(exc)[:120]}")
    if args.fill_gaps and tess_files and not args.offline:
        say("\nDownloading other versions of the same sectors, to cover gaps")
        try:
            if not download_fill(tic, outdir, dur, say):
                say("  nothing new to download")
        except Exception as exc:
            say(f"  failed: {str(exc)[:120]}")
        tess_files = sorted(str(p) for p in outdir.glob("tess_s*.csv"))
    gaia_phot = outdir / "gaia_epoch_photometry.csv"
    if not gaia_phot.exists() and Path("gaia_epoch_photometry.csv").exists():
        gaia_phot = Path("gaia_epoch_photometry.csv")          # saved next to wasp_lc.csv
    none_note = outdir / "gaia_epoch_photometry_none.txt"     # Gaia published none: do not ask again
    if gaia and not gaia_phot.exists() and not none_note.exists() and not args.offline:
        say(f"\nAsking the Gaia archive for visit-by-visit photometry (gives up after {NET_TIMEOUT} s)...")
        try:
            exact = within(NET_TIMEOUT, fetch_gaia_id, star["ra"], star["dec"])   # a saved identifier may be rounded
            if exact != gaia.get("source_id"):
                gaia["source_id"] = exact
                (outdir / "gaia.json").write_text(json.dumps(gaia, indent=1))
            within(2 * NET_TIMEOUT, fetch_gaia_photometry, exact, gaia_phot)
        except Exception as exc:
            say(f"  Gaia photometry not available: {str(exc)[:120]}")
            if "has none for this star" in str(exc):
                none_note.write_text(f"{exc}\n(delete this file to ask the Gaia archive again)\n")
    elif none_note.exists():
        say("\nGaia published no visit-by-visit photometry for this star (as found earlier; "
            f"delete {none_note.name} to ask again)")
    gdf = load_gaia(gaia_phot) if gaia_phot.exists() else None
    if gdf is not None and len(gdf) < 4:
        gdf = None

    # measured velocities: LAMOST spectra, anything added by hand, and Gaia's summary
    rv_path = Path(args.rv) if args.rv else outdir / "rv_epochs.csv"
    if not rv_path.exists() and not args.offline:
        try:
            rows = fetch_lamost(star["ra"], star["dec"])
            pd.DataFrame(rows, columns=["JD", "RV", "RV_ERR", "SOURCE"]).to_csv(rv_path, index=False)
        except Exception as exc:
            say(f"  LAMOST velocities not available: {str(exc)[:120]}")
    rv_epochs = []
    if rv_path.exists():
        table = pd.read_csv(rv_path)
        for row in table.to_dict("records"):
            when = float(row["JD"]) + (2400000.5 if float(row["JD"]) < 2400000 else 0.0)   # accept MJD too
            kind = str(row.get("STAR", "1")).strip().upper()
            kind = {"NAN": "1", "": "1", "1.0": "1", "2.0": "2"}.get(kind, kind)
            if kind not in ("1", "2", "A", "B", "BLEND"):
                sys.exit(f"{rv_path}: STAR must be 1, 2, A, B or blend (got '{row.get('STAR')}')")
            rv_epochs.append((when, float(row["RV"]), max(float(row["RV_ERR"]), 0.5), kind,
                              str(row.get("SOURCE", ""))))
    double_lined = any(r[3] in ("A", "B", "2") for r in rv_epochs)
    if gaia and "radial_velocity" not in gaia and not args.offline:
        try:
            gaia.update(fetch_gaia_rv(star["ra"], star["dec"]))
            (outdir / "gaia.json").write_text(json.dumps(gaia, indent=1))
        except Exception as exc:
            say(f"  Gaia velocity not available: {str(exc)[:120]}")
    rv_median = None
    if gaia and gaia.get("radial_velocity") is not None and (gaia.get("rv_nb_transits") or 0) >= 8:
        # With two sets of lines, Gaia's single-star measurement is a blend of both stars.
        rv_median = (gaia["radial_velocity"], 5.0, "BLEND" if double_lined else "1")
    if rv_epochs or rv_median:
        say("\nMEASURED VELOCITIES")
        whose = {"1": "the star", "2": "the companion", "A": "star A", "B": "star B", "BLEND": "both stars blended"}
        for when, value, err, kind, source in rv_epochs:
            say(f"  JD {when:.2f}: {value:+7.1f} +/- {err:.1f} km/s   {whose[kind]}   ({source})")
        if any(r[3] in ("A", "B") for r in rv_epochs):
            say("  Stars A and B are the two sets of lines; either could be the one that is eclipsed.")
        if rv_median:
            how = ("median of" if gaia.get("rv_method_used") in (None, 1) and gaia.get("rv_amplitude_robust") is not None
                   else "combined spectrum of")
            say(f"  Gaia, {how} {gaia['rv_nb_transits']:.0f} visits (2014-2017): {rv_median[0]:+.1f} km/s"
                + (" (a blend of both stars)" if double_lined else "")
                + (f"; visits range over {gaia['rv_amplitude_robust']:.0f} km/s"
                   if gaia.get("rv_amplitude_robust") else ""))

    # Gaia at the level of individual transits: scan law, comparison stars, quality flags
    ruwe_model = gaia_rv = None
    if gaia:
        say("\nGAIA, TRANSIT BY TRANSIT")
        if not args.offline:
            for name, fetch in (("gaia_diagnostics.json", lambda: gaia_tools.diagnostics(star["ra"], star["dec"])),
                                ("gaia_calibration.json", None)):
                path = outdir / name
                if path.exists():
                    continue
                try:
                    if fetch is not None:
                        path.write_text(json.dumps(fetch(), indent=1))
                    else:
                        t = (gaia_tools.load_diagnostics(outdir) or {}).get("target") or {}
                        if t.get("bp_rp") is not None:
                            path.write_text(json.dumps(gaia_tools.calibrate(
                                star["ra"], star["dec"], t["phot_g_mean_mag"], t["bp_rp"], gaia.get("ruwe")),
                                indent=1))
                except Exception as exc:
                    say(f"  could not fetch {name}: {str(exc)[:120]}")
        diag = gaia_tools.load_diagnostics(outdir)
        if diag and diag.get("target"):
            for key in ("astrometric_n_good_obs_al", "rv_amplitude_robust", "rv_nb_transits", "bp_rp",
                        "rv_method_used", "grvs_mag"):
                if gaia.get(key) is None and diag["target"].get(key) is not None:
                    gaia[key] = diag["target"][key]
            gaia_tools.report_diagnostics(diag, say)
        law = gaia_tools.scan_law_for(outdir, star["ra"], star["dec"], gaia.get("astrometric_n_good_obs_al"),
                                      args.gost, say, fetch=not args.offline)
        if gaia.get("ruwe") is not None and gaia.get("phot_g_mean_mag") is not None:
            ruwe_model = gaia_tools.RuweModel(law, gaia["phot_g_mean_mag"], gaia["ruwe"],
                                              gaia_tools.load_calibration(outdir, say))
            cal = gaia_tools.load_calibration(outdir, lambda *a: None)
            if cal.get("f_at_or_above_observed") is not None:
                say(f"  {100 * cal['f_at_or_above_observed']:.1f}% of comparison stars have RUWE as high as "
                    f"this star's {gaia['ruwe']:.2f}")
        if rv_median and not args.no_gaia_transits:
            gaia_rv = gaia_tools.GaiaRvModel(law, gaia, resolve=args.rvs_resolve)
            if args.gaia_amplitude == "ignore" or (args.gaia_amplitude == "auto" and double_lined):
                gaia_rv.median_only()
                say(f"  Gaia velocity, {gaia_rv.describe()}")
            else:
                say(f"  Gaia velocity, {gaia_rv.describe()}; lines count as separate above "
                    f"~{gaia_rv.resolve:.0f} km/s")

    report_catalogue(say, tic, star, gaia)

    files = tess_files + ([args.wasp] if Path(args.wasp).exists() else []) + list(args.extra)
    if not tess_files:
        sys.exit("No TESS light curves. The eclipse's shape (part A) needs the sector that contains it.")

    say("\nLIGHT CURVES")
    sets, frames, visits = [], {}, []
    for path in files:
        df = load(path, say)
        series = depth_series(df, dur)
        if len(series) < 20:
            if len(df) >= 30:
                visits.append((Path(path).stem, df))
                say(f"    too sparse for the sliding window; used night by night ({len(df)} visits)")
            else:
                say("    too little data to test anything; skipped")
            continue
        stretch = calibrate(series, t0, dur)
        ok = np.isfinite(series[:, 2])
        say(f"    {ok.sum()} usable window positions; typical depth uncertainty "
            f"{np.median(series[ok, 2]) * 100:.2f}% (errors stretched x{stretch:.2f})")
        name = Path(path).stem
        sets.append({"name": name, "series": series, "scale": 1.0, "fill": False})
        frames[name] = df
    # A second version of a sector counts only where the versions before it have no usable
    # window, so that no stretch of sky time is counted twice.
    for number, s in enumerate(sets):
        version = re.fullmatch(r"(tess_s\d+)_\w+", s["name"])
        if not version:
            continue
        s["fill"] = True
        centres = s["series"][:, 0]
        covered = np.zeros(len(centres), bool)
        for earlier in sets[:number]:
            if earlier["name"] == version.group(1) or earlier["name"].startswith(version.group(1) + "_"):
                c = earlier["series"][np.isfinite(earlier["series"][:, 2]), 0]
                if len(c):
                    j = np.clip(np.searchsorted(c, centres), 1, len(c) - 1) if len(c) > 1 else np.zeros(len(centres), int)
                    near = np.minimum(np.abs(centres - c[j - 1]), np.abs(centres - c[j])) if len(c) > 1 \
                        else np.abs(centres - c[0])
                    covered |= near <= STEP * dur
        usable = np.isfinite(s["series"][:, 2])
        s["series"][covered, 2] = np.nan
        say(f"    {s['name']} is used only where {version.group(1)} has nothing to test with: "
            f"{int((usable & ~covered).sum()) * STEP * dur:.1f} days of extra coverage")

    # ---- the known eclipse ---------------------------------------------------------------
    depth, anchor = None, None
    for s in sets:
        near = (np.abs(s["series"][:, 0] - t0) <= dur / 2) & np.isfinite(s["series"][:, 2])
        if near.any():
            j = np.where(near)[0][np.argmax(s["series"][near, 1] / s["series"][near, 2])]
            if s["series"][j, 1] / s["series"][j, 2] > 4:
                depth, anchor = s["series"][j, 1], s["name"]
                break
    if depth is None:
        sys.exit("None of the light curves contains the known eclipse at --t0.")
    tc, x, fn, sigma, cadence = eclipse_points(frames[anchor], t0, dur)
    profile = eclipse_profile(x, fn, dur)
    say(f"\nKNOWN ECLIPSE in {anchor}: centre JD {tc:.4f}, deepest about "
        f"{float(profile(0.0)) * 100:.1f}%, {depth * 100:.2f}% averaged over {args.duration:g} hours")
    t0 = tc
    say(f"    mid-eclipse uncertainty {centre_error(x, fn, sigma, dur) * 1440:.1f} minutes (1 sigma, from the "
        "centre of the missing light; a model fit can place the middle a minute or two differently)")

    # ---- part B: periods the photometry rules out ------------------------------------------
    baseline = max(np.abs(s["series"][:, 0] - t0).max() for s in sets)
    if gdf is not None:
        baseline = max(baseline, float(np.abs(gdf.t - t0).max()))
    for _, df in visits:
        baseline = max(baseline, float(np.abs(df.t - t0).max()))
    periods = period_grid(args.min_period, args.max_period, baseline, dur)
    say(f"\nPART B: testing {len(periods):,} periods against every light curve")
    # Other eclipses that are certain: found in the data, or given with --also.
    found = other_eclipses([s for s in sets if "series" in s], t0, dur, depth)
    sure = []                                           # (time, timing error, where it came from)
    fill_names = {s["name"] for s in sets if s.get("fill")}
    maybe = []                                          # dips seen only in the gap-filling versions
    for r in (found[found.confirmed] if len(found) else found).itertuples():
        if r.dataset in fill_names:
            say(f"    POSSIBLE FURTHER ECLIPSE at JD {r.jd:.4f} in {r.dataset} ({r.depth_pct:.1f}% deep, "
                f"{r.sigma:.1f} sigma). This version of the data is less clean, so it is")
            say(f"    counted as support only. Look at it; if it is real, run again with --also {r.jd - 2457000:.3f}")
            maybe.append(r.jd)
            continue
        sure.append([r.jd, 0.1 * dur, f"found in {r.dataset} ({r.sigma:.1f} sigma)", r.dataset])
    for given in args.also:
        given = given + 2457000.0 if given < 100000 else given
        if abs(given - t0) < 2 * dur or any(abs(given - e[0]) < 2 * dur for e in sure):
            continue                                    # the known eclipse, or one already found
        holder = [n for n, df in frames.items() if ((df.t - given).abs() < 0.5 * dur).any()]
        sure.append([given, 0.1 * dur, "given with --also", holder[0] if holder else None])
    t0_error = centre_error(x, fn, sigma, dur)
    for entry in sure:
        if entry[3] is not None:                        # pin the centre down from the points themselves
            try:
                c2, x2, f2, s2, _ = eclipse_points(frames[entry[3]], entry[0], dur)
                entry[0], entry[1] = c2, centre_error(x2, f2, s2, dur)
            except Exception:
                pass
    skip = [e[0] for e in sure]
    total = np.zeros(len(periods))
    missed = np.zeros(len(periods), dtype=bool)         # some predicted eclipse is plainly absent
    for s in sets:
        s["lnlr"], _, _, _, miss_here = accumulate(periods, s["series"], t0, dur, depth, skip,
                                                   -0.5 * args.sigma**2)
        total += s["lnlr"]
        missed |= miss_here
    deep_nights = []
    for name, df in visits:
        v_lnlr, v_missed, dips, typical = accumulate_visits(periods, df, t0, dur, profile, skip, -0.5 * args.sigma**2)
        total += v_lnlr
        missed |= v_missed
        sets.append({"name": name, "lnlr": v_lnlr})
        say(f"    {name}: {len(df)} survey visits judged night by night (typical error {100 * typical:.1f}% per visit)")
        for night in dips:
            tt, yy, ee, steady = night
            when = pd.Timestamp(float(tt.mean()) - 2440587.5, unit="D").strftime("%Y-%m-%d")
            if not steady:
                say(f"    faint visits in {name} on {when} sit next to normal ones taken minutes apart: "
                    "bad frames, not an eclipse; treated as noise")
                continue
            say(f"    DEEP DIP in {name} on {when} (JD {tt.mean():.3f}): {int((((1 - yy) / ee) >= 5).sum())} visits, "
                f"up to {100 * (1 - yy.min()):.1f}% faint")
            deep_nights.append((name, when, night))
    seen, match, kept, by_group = [], None, {}, {}     # possible eclipses among Gaia's visits
    if gdf is not None:
        groups, noise = gaia_groups(gdf)
        if groups:
            by_group = {}
            for dip in gaia_dips(groups, float(profile(0.0))):
                if dip[0] not in by_group or dip[2] > by_group[dip[0]][2]:
                    by_group[dip[0]] = dip
            seen = list(by_group.values())
            g_lnlr, kept = accumulate_gaia(periods, groups, t0, dur, profile, keep=set(by_group))
            total += g_lnlr
            say(f"    Gaia photometry: {len(gdf)} visits, JD {gdf.t.min():.1f} to {gdf.t.max():.1f}; "
                f"{sum(len(grp[0]) for grp in groups)} of them in {len(groups)} groups made within hours. "
                f"Visits in a group differ by {noise * 100:.2f}%.")
            for dip in seen:
                say(f"    POSSIBLE SECOND ECLIPSE in Gaia: the visit at JD {dip[1]:.4f} is "
                    f"{dip[2] * 100:.1f}% fainter than the rest of its group ({dip[2] / dip[3]:.1f} sigma).")
            if seen:
                match = np.ones(len(periods), dtype=bool)
                for number in by_group:
                    match &= kept[number] >= kept[number].max() - 0.5 * args.sigma**2
            sets.append({"name": "gaia_photometry", "lnlr": g_lnlr})
    soft = total <= -0.5 * args.sigma**2
    ruled_out = soft | missed
    for name, when, night in deep_nights:
        if args.ignore_dips:
            continue
        before = ruled_out.copy()
        ruled_out |= ~explains(periods, night, t0, dur, profile)
        say(f"    the period must put an eclipse on the dip of {when} in {name} "
            "(--ignore-dips to treat it as noise instead)")
        if ruled_out.all() and not before.all():
            say(f"    NO PERIOD LEFT once the dip of {when} is required: either that dip is bad data, or one")
            say("    of the eclipses taken as certain is wrong. Look at it, and rerun with --ignore-dips to")
            say("    see the answer without the survey dips as rules.")
    for when, error, origin, _ in sure:
        orbits = (when - t0) / periods
        miss = np.abs(orbits - np.round(orbits)) * periods
        slack = max(0.1 * dur, 5.0 * np.hypot(error, t0_error))
        ruled_out |= (miss > slack) | (np.round(orbits) == 0)
        say(f"    second eclipse at JD {when:.4f} (+/- {error * 1440:.1f} min), {origin}: the period must fit it "
            f"to within {slack * 1440:.0f} minutes")
    if sure:
        say("    (that eclipse is not also counted as support: a period that puts any other")
        say("    eclipse on data showing none is ruled out, however well it fits the two seen)")
    say(f"    {100 * ruled_out.mean():.1f}% of all trial periods are ruled out"
        + (f" ({int((~ruled_out).sum()):,} remain)" if ruled_out.mean() > 0.999 else ""))
    pd.DataFrame(intervals(periods, ~ruled_out), columns=["from_days", "to_days"]).to_csv(
        outdir / "allowed_periods.csv", index=False)

    # ---- the coarse map ----------------------------------------------------------------------
    n_p, n_m = 140, 70
    edges = np.exp(np.linspace(np.log(args.min_period), np.log(args.max_period), n_p + 1))
    p_mid = np.sqrt(edges[:-1] * edges[1:])
    which = np.clip(np.digitize(periods, edges) - 1, 0, n_p - 1)
    open_frac = np.bincount(which, weights=~ruled_out, minlength=n_p) / \
        np.maximum(np.bincount(which, minlength=n_p), 1)
    per_bin = np.maximum(np.bincount(which, minlength=n_p), 1)
    fits_gaia = None
    if match is not None:
        # If the dip is taken as real it may not also be used to excuse missing eclipses
        # elsewhere, so the rest of the data must leave the period open by itself.
        other = total - sum(kept[number] for number in by_group)
        fits_gaia = match & ~((other <= -0.5 * args.sigma**2) | (ruled_out & ~soft))
    if not (args.no_blend_correct or args.no_isochrone) and not (outdir / "stellar_split.json").exists():
        iso = args.iso or next((str(p) for p in sorted(outdir.glob("*.iso.cmd"))), None)
        if iso:
            say("\nISOCHRONE FIT (stellar_model.py): two coeval stars fitted to the system's photometry")
            try:
                sm_args = stellar_model.parser().parse_args(
                    [str(outdir), "--iso", iso, "--feh", str(args.feh), "--nstars", str(args.nstars)])
                stellar_model.run(sm_args, say=lambda text="": say("  " + text.replace("\n", "\n  ")))
            except Exception as exc:
                say(f"  isochrone fit failed ({str(exc)[:160]}); sharing the light as mass^4.5 instead")
        else:
            say("\n(No isochrone: give --iso auto --feh <[Fe/H]>, or put a MIST .iso.cmd file in the output "
                "folder, for stars whose radii and light come from one isochrone)")
    split = None if (args.no_blend_correct or args.no_isochrone) else load_split(outdir)
    if args.no_blend_correct:
        masses = np.exp(np.linspace(np.log(0.05), np.log(min(star["mass"], 1.2)), n_m))
        m1, r1 = np.full(n_m, star["mass"]), np.full(n_m, star["rad"])
        r2, teff1, teff2 = companion_radius(masses), np.full(n_m, star["teff"]), \
            np.array([companion_temperature(m) for m in masses])
        light2 = lightG2 = lightRV2 = (masses / m1) ** 4.5 / (1 + (masses / m1) ** 4.5)
        radius_err = 0.07 if args.radius_err is None else args.radius_err
        split_note = "the catalogue's mass and radius taken as the eclipsed star's own"
    elif split is not None:
        # One isochrone at the age fitted to the system's photometry (stellar_model.py): every star's
        # radius, temperature and light follow from its mass, and the eclipsed star's mass is whatever
        # makes the two stars' TESS-band light add up to what is observed. Either star may be the
        # brighter one.
        iso, f_tot = split["iso"], split["fT_obs"][0]
        if split.get("nstars") == 3 and "m_c" in split.get("masses", {}):
            f_tot = f_tot - float(iso.flux("T", split["masses"]["m_c"][0]))     # the third star's light
        lo = max(0.08, iso.m_min)
        hi = float(iso.mass_for_flux("T", 0.9 * f_tot))
        if not np.isfinite(hi):
            hi = iso.m_max
        masses = np.exp(np.linspace(np.log(lo), np.log(min(hi, 1.6)), n_m))
        m1 = iso.mass_for_flux("T", f_tot - iso.flux("T", masses))
        if np.isnan(m1).any():
            keep = np.isfinite(m1)
            masses = np.exp(np.linspace(np.log(masses[keep].min()), np.log(masses[keep].max()), n_m))
            m1 = iso.mass_for_flux("T", f_tot - iso.flux("T", masses))
        r1, r2 = iso("radius", m1), iso("radius", masses)
        teff1, teff2 = iso("teff", m1), iso("teff", masses)
        pair = lambda band: iso.flux(band, masses) / (iso.flux(band, masses) + iso.flux(band, m1))
        light2, lightG2, lightRV2 = pair("T"), pair("G"), pair("RP")
        rd = split["derived"].get("r_a", [1, 1, 1])
        radius_err = args.radius_err if args.radius_err is not None else \
            float(np.clip((rd[2] - rd[1]) / (2 * rd[0]), 0.02, 0.2))
        split_note = (f"isochrone {split['isochrone_file']} at log age {split['log_age'][0]:.2f} "
                      f"(stellar_split.json)")
    else:
        # The catalogue's mass and radius come from the light of both stars together. A companion of
        # mass m takes its share (light ~ mass^4.5), the eclipsed star keeps the rest: its mass follows
        # from its own light, and its radius from that light at the catalogue temperature (L ~ R^2).
        # Unless --eclipsed-brighter, the companion may supply up to 90% of the light: a grazing
        # eclipse can belong to either star.
        top = min(star["mass"] * (0.5 if args.eclipsed_brighter else 0.9) ** (1 / 4.5), 1.4)
        masses = np.exp(np.linspace(np.log(0.05), np.log(top), n_m))
        m1 = (star["mass"] ** 4.5 - masses**4.5) ** (1 / 4.5)
        r1 = star["rad"] * (m1 / star["mass"]) ** 2.25
        r2, teff1 = companion_radius(masses), np.full(n_m, star["teff"])
        teff2 = np.array([companion_temperature(m) for m in masses])
        light2 = lightG2 = lightRV2 = (masses / m1) ** 4.5 / (1 + (masses / m1) ** 4.5)
        radius_err = 0.07 if args.radius_err is None else args.radius_err
        split_note = ("the catalogue's light shared out as mass^4.5 (run stellar_model.py for an "
                      "isochrone-based split)")
    say(f"\nSTARS: {split_note}; eclipsed star's radius uncertain by {100 * radius_err:.0f}%"
        + (f"; third light up to {100 * args.third_light:.0f}% allowed" if args.third_light > 0 else ""))

    # A measured flux ratio of the two stars (from a double-lined spectrum), fainter / brighter in
    # one band, constrains the companion's mass directly. Either star may be the fainter one.
    fr_like = np.ones(n_m)
    if args.flux_ratio:
        band, fr_obs, fr_err = args.flux_ratio[0].upper(), float(args.flux_ratio[1]), float(args.flux_ratio[2])
        if split is not None and band in stellar_model.BAND_COLUMNS:
            f2, f1 = iso.flux(band, masses), iso.flux(band, m1)
            how = f"isochrone {band}-band fluxes"
        else:
            f2, f1 = masses**4.5, m1**4.5
            how = "light ~ mass^4.5 (no band dependence without an isochrone)"
        fr_model = np.minimum(f1, f2) / np.maximum(f1, f2)
        fr_like = np.exp(-0.5 * ((fr_model - fr_obs) / fr_err) ** 2)
        ok = fr_like > 0.01
        say(f"  measured flux ratio {fr_obs:.2f} +/- {fr_err:.2f} (fainter/brighter, {band}; {how}): "
            + (f"companion masses {masses[ok].min():.2f}-{masses[ok].max():.2f} Msun fit it within 3 sigma"
               if ok.any() else "NO companion mass on the grid reproduces it - check the band and value"))
        if not ok.any():
            fr_like = np.ones(n_m)
    prim = dict(star, mass=m1, rad=r1, r2=r2, teff=teff1, teff2=teff2, light2=light2)   # per companion mass
    k = r2 / r1
    a_over_r = ((m1[None, :] + masses[None, :]) * (p_mid[:, None] / 365.25) ** 2) ** (1 / 3) \
        * 215.032 / r1[None, :]

    say("\nPART A: fitting the eclipse's shape for every crossing speed and companion mass")
    variants = shape_variants(n_m, r1, r2, light2, args.third_light, radius_err)
    fit = fit_shape(x, fn, sigma, cadence, masses, prim, dur, say, variants=variants)
    best, dof, rescale, j = fit["best"], fit["dof"], fit["rescale"], fit["mass_index"]
    log_speeds = fit["log_speed"]
    say(f"    best fit: chi-squared {best:.0f} for {dof} points "
        f"({'errors enlarged to match' if rescale > 1.05 else 'consistent with the noise'}); "
        f"crossing speed {fit['speed']:.1f} solar radii per day")
    # the best-fitting shape itself (as drawn in eclipse_fit.png), so that figures made later from
    # summary.txt draw this eclipse rather than one rebuilt from medians taken one at a time
    say(f"    BEST-FIT ECLIPSE: radius ratio {fit['k_best']:.4f}, impact parameter "
        f"{fit['frac_b'] * (1 + fit['k_best']):.4f} stellar radii, crossing speed {fit['speed_r1']:.4f} "
        f"stellar radii per day, eclipsed star's share of the light {fit['dilute_best']:.4f}, "
        f"length {24 * fit['length']:.3f} hours")

    # Speed of a circular orbit at each period and mass (solar radii per day), and how much
    # faster or slower than that the companion may be going on an eccentric one.
    v_circ = 2.0 * np.pi * a_over_r * r1[None, :] / p_mid[:, None]
    s_grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 71))
    w_es, e_mid = speed_factor_prior(s_grid, args.circular)
    w_s = w_es.sum(axis=0)
    # The likelihood is far narrower in speed than these bins, so each bin gets
    # its average over the speeds the bin covers, never a single sample.
    half_s = 0.5 * np.log(s_grid[1] / s_grid[0])
    half_p = np.log(edges[1:] / edges[:-1]) / 6.0                  # speed goes as period^(-1/3)
    spread3 = np.zeros((n_p, n_m, len(s_grid)))                    # [period, mass, speed factor]
    graze3 = np.zeros_like(spread3)
    for jm in range(n_m):
        centre = np.log(v_circ[:, jm])[:, None] + np.log(s_grid)[None, :]
        width = half_p[:, None] if args.circular else half_s
        spread3[:, jm, :] = spread_over(log_speeds, fit["like"][:, jm], centre - width, centre + width)
        graze3[:, jm, :] = spread_over(log_speeds, fit["graze"][:, jm], centre - width, centre + width) * w_s
    shape3 = spread3 * w_s

    prior = masses[None, :] * (1.0 + k[None, :]) / a_over_r * fr_like[None, :]   # flat in mass and log-period,
    base3 = prior[:, :, None] * shape3                             # before any period is ruled out
    stages = [("eclipse shape and geometry only", prior[:, :, None] * shape3)]   # times the
    stages.append(("plus missing eclipses (TESS, SuperWASP, Gaia photometry)",  # chance of eclipsing
                   stages[-1][1] * open_frac[:, None, None]))
    ruwe_pred = None
    if gaia and ruwe_model is not None:
        # The pair's own photocentre orbit, projected on Gaia's scans, through DR3's five-parameter fit.
        # It can only add scatter, so it counts against orbits that predict more than Gaia saw; an
        # observed RUWE *above* the prediction is left to other causes (a third star; see triple_fit.py).
        kappa = ruwe_kappa(ruwe_model, p_mid)
        a0 = ((m1[None, :] + masses[None, :]) * (p_mid[:, None] / 365.25) ** 2) ** (1 / 3) * gaia["parallax"] \
            * np.abs(masses[None, :] / (m1[None, :] + masses[None, :]) - lightG2[None, :])
        lam = kappa[:, None] * a0**2
        ruwe_pred = ruwe_model.s * np.sqrt(1.0 + lam / ruwe_model.nu)
        spread_r = np.hypot(ruwe_model.s**2 * np.sqrt(2 * (ruwe_model.nu + 2 * lam)) / ruwe_model.nu
                            / (2 * ruwe_pred), ruwe_model.core) * args.gaia_soften
        over = np.clip(ruwe_pred - ruwe_model.obs, 0, None)
        g_like = np.exp(-0.5 * (over / spread_r) ** 2)
        say(f"\nPART C: Gaia RUWE {ruwe_model.obs:.2f} ({ruwe_model.describe()})")
        say(f"  orbits predicting RUWE above that (by more than {args.gaia_soften:g}x the expected scatter) "
            "lose probability")
        stages.append(("plus Gaia astrometry", stages[-1][1] * g_like[:, :, None]))
        base3 = base3 * g_like[:, :, None]
    elif gaia:
        g_like, spread = gaia_likelihood(p_mid[:, None], masses[None, :], m1[None, :], gaia,
                                         args.gaia_soften)
        limit = 2.0 * np.sqrt(gaia["astrometric_excess_noise"] ** 2 + 2 * spread)
        say(f"\nPART C: Gaia would have noticed a wobble above about {limit:.2f} mas")
        stages.append(("plus Gaia astrometry", stages[-1][1] * g_like[:, :, None]))
        base3 = base3 * g_like[:, :, None]

    post_e, rv_note, rv_branches = None, [], None
    if len({(round(r[0], 3), r[3]) for r in rv_epochs}) + (1 if rv_median else 0) >= 2:
        say("\nPART E: comparing every orbit with the measured velocities")
        base2 = prior * (g_like if gaia else 1.0)
        chance5, support, mirror, sampled, n_tb = velocity_constraint(
            periods, ~ruled_out, which, per_bin, masses, m1, s_grid, e_mid, w_es,
            spread3, base2, rv_epochs, rv_median, t0, gaia_rv=gaia_rv, light_rv2=lightRV2)
        if gaia_rv is not None and rv_median:
            n_open = int((np.bincount(which[~ruled_out], minlength=n_p) > 0).sum())
            where = ("for every open period" if n_tb >= n_open else
                     f"in {n_tb} of {n_open} period bins (elsewhere only the median, averaged over the orbit)"
                     if n_tb else "nowhere: too many periods open (only the median, averaged over the orbit)")
            say(f"    Gaia's velocities modelled transit by transit ({gaia_rv.n} transits; "
                + ("median and robust amplitude" if gaia_rv.amp_obs is not None else gaia_rv.stat)
                + f") {where}")
        chance4 = chance5.mean(axis=4)
        post4 =base2[:, :, None, None] * spread3[:, :, :, None] * w_es.T[None, None, :, :] * chance4
        if post4.sum() > 0:
            stages.append(("plus measured velocities", post4.sum(axis=3)))
            post_e = post4.sum(axis=(0, 1, 2))
            rv_branches = chance5
            if not sampled:
                good = np.nan_to_num(support, nan=0.0)
                keep = (~ruled_out) & (good >= good.max() * np.exp(-0.5 * args.sigma**2))
                pd.DataFrame(intervals(periods, keep), columns=["from_days", "to_days"]).to_csv(
                    outdir / "allowed_periods_with_velocities.csv", index=False)
                say(f"    of the periods the photometry leaves open, {100 * keep.sum() / max((~ruled_out).sum(), 1):.0f}% "
                    "also fit the velocities")
                say("    (listed in allowed_periods_with_velocities.csv)")
            else:
                say("    (too many trial periods to list which ones fit; the map below still uses them all)")
        else:
            say("    no orbit of the kind fitted here reproduces the velocities; they are left out below")
    elif rv_epochs:
        say("\nPART E: one velocity cannot constrain the orbit, since the system's own velocity is unknown.")
        say("  It is kept as a reference: a later spectrum that differs by tens of km/s confirms the companion.")
    # ---- part F: the companion's own eclipse ---------------------------------------------------
    base2 = prior * (g_like if gaia else 1.0)
    if rv_branches is not None:
        before_f = base2[:, :, None, None, None] * spread3[..., None, None] * w_es.T[None, None, :, :, None] \
            * rv_branches
    else:
        before_f = (base2 * open_frac[:, None])[:, :, None, None, None] * spread3[..., None, None] \
            * w_es.T[None, None, :, :, None] * np.ones(2)
    post5 = before_f                                    # [period, mass, speed factor, eccentricity, mirror]
    sec_report = None
    open_runs = intervals(periods, ~ruled_out)
    if args.skip_secondary or len(open_runs) == 0 or before_f.sum() <= 0:
        pass
    elif len(open_runs) > MAX_RUNS_F:
        say(f"\nPART F: the companion's own eclipse is not tested while {len(open_runs):,} separate period "
            "ranges remain open")
        say("  (it needs the period pinned down, as by a second eclipse).")
    else:
        say("\nPART F: looking for the companion's own eclipse, when it passes behind the star")
        chance_f, sec_report = secondary_constraint(
            periods, ~ruled_out, which, open_runs, masses, k, prim, fit["length"], max(dur, fit["length"]), t0,
            s_grid, e_mid, w_es, before_f, sets, frames, visits, say)
        post5 = before_f * chance_f
        report_secondary(say, sec_report, before_f, post5, e_mid)
        if post5.sum() > 0:
            stages.append(("plus the companion's own eclipse", post5.sum(axis=(3, 4))))
            post_e = post5.sum(axis=(0, 1, 2, 4))
        else:
            say("    no orbit survives this test; it is left out below. Either the companion is fainter")
            say("    than a main-sequence star of its mass, or the eclipses do not come from one companion.")
            post5 = before_f
    stages[-1] = (stages[-1][0] + " (all evidence)", stages[-1][1])

    say("\nRESULTS" + (" (circular orbits only)" if args.circular else ""))
    say("  (medians, with 68% and 95% ranges. Periods are on a coarse map"
        + ("; the exact period is under CANDIDATE PERIODS" if sure else "") + ".)")
    for label, post3 in stages:
        if post3.sum() <= 0:
            say(f"  {label}: nothing survives.")
            say("\nNo orbit of this kind fits. Either the orbit is eccentric (drop --circular),")
            say("or the eclipse is not on this star, or it is not an eclipse by a companion.")
            (outdir / "summary.txt").write_text("\n".join(say.lines) + "\n")
            return
        describe(say, label, post3.sum(axis=2) / post3.sum(), p_mid, masses, star, r2)
    post3 = stages[-1][1] / stages[-1][1].sum()
    post = post3.sum(axis=2)
    if not args.no_blend_correct:
        w_m = post.sum(axis=0)
        mi, ri = interval(m1, w_m), interval(r1, w_m)
        share = interval(light2, w_m)
        r2i = interval(r2, w_m)
        say(f"  the eclipsed star, once the companion's share of the TESS-band light ({100 * share[2]:.0f}%, "
            f"68%: {100 * share[1]:.0f} to {100 * share[3]:.0f}) is taken out:")
        say(f"      MAIN STAR: mass {mi[2]:.3f} Msun, radius {ri[2]:.3f} Rsun   (68%: {mi[1]:.2f} to {mi[3]:.2f} Msun, "
            f"{ri[1]:.2f} to {ri[3]:.2f} Rsun; catalogue {star['mass']:.2f} Msun, {star['rad']:.2f} Rsun)")
        say(f"      companion radius {r2i[2]:.3f} Rsun (68%: {r2i[1]:.2f} to {r2i[3]:.2f})")
        t1i, t2i = interval(teff1, w_m), interval(teff2, w_m)
        say(f"      temperatures: eclipsed star {t1i[2]:.0f} K (68%: {t1i[1]:.0f} to {t1i[3]:.0f}), companion "
            f"{t2i[2]:.0f} K (68%: {t2i[1]:.0f} to {t2i[3]:.0f}); companion's TESS-band light share "
            f"{share[2]:.4f}")
        if args.flux_ratio:
            fri = interval(fr_model, w_m)
            say(f"      flux ratio fainter/brighter in {band}: {fri[2]:.2f} (68%: {fri[1]:.2f} to {fri[3]:.2f}); "
                f"measured {fr_obs:.2f} +/- {fr_err:.2f}")
        brighter = float(w_m[light2 > 0.5].sum() / max(w_m.sum(), 1e-300))
        if brighter > 0.05:
            say(f"      the companion is the brighter star in {100 * brighter:.0f}% of the probability: the eclipse "
                "may be of the fainter star; spectra would tell")
    if ruwe_pred is not None:
        r_int = interval(ruwe_pred.ravel(), post.ravel())
        say(f"  RUWE this binary alone would give: {r_int[2]:.3f} (68%: {r_int[1]:.3f} to {r_int[3]:.3f}); "
            f"Gaia measured {ruwe_model.obs:.2f}"
            + (". The excess needs another cause - see triple_fit.py." if ruwe_model.obs > r_int[4] + 3 * ruwe_model.core
               else "."))

    # ---- eccentricity ------------------------------------------------------------------------------
    share = w_es / np.maximum(w_es.sum(axis=0, keepdims=True), 1e-300)

    def eccentricity(p3, e_weights=None):
        post_s = p3.sum(axis=(0, 1))
        s_int = interval(s_grid, post_s)
        say(f"  The companion crossed {s_int[2]:.1f} times faster than a circular orbit would "
            f"(68%: {s_int[1]:.1f} to {s_int[3]:.1f}).")
        if not args.circular:
            e_weights = share @ post_s if e_weights is None else e_weights
            e_int = interval(e_mid, e_weights)
            say(f"  Eccentricity {e_int[2]:.2f} (68%: {e_int[1]:.2f} to {e_int[3]:.2f}; "
                f"95% of the probability lies above {interval(e_mid, e_weights, (0.05,))[0]:.2f}).")

    say("\nECCENTRICITY")
    eccentricity(post3, post_e)
    told = rv_branches is not None or sec_report is not None
    if told and not args.circular and post5.sum() > 0:
        side = post5.sum(axis=(0, 1, 2, 3))
        after = side[1] / side.sum()
        told = max(after, 1 - after) >= 0.8
        by = "the velocities" + (" and the companion's own eclipse" if sec_report else "") if rv_branches is not None \
            else "the companion's own eclipse"
        if told:
            say(f"  Closest approach comes {'after' if after >= 0.5 else 'before'} the eclipse "
                f"(probability {100 * max(after, 1 - after):.0f}%, from {by}).")
        else:
            which_ones = ("Neither the velocities nor the companion's own eclipse shows"
                          if rv_branches is not None and sec_report else
                          "The velocities do not show" if rv_branches is not None else
                          "The companion's own eclipse does not show")
            say(f"  {which_ones} whether closest approach comes before or after the eclipse.")
    rep = None
    if not args.circular and post5.sum() > 0:
        # one orbit that respects everything, for drawing: the most probable cell
        cube = post5.sum(axis=(0, 1))                                  # [speed factor, eccentricity, mirror]
        if told:
            cube = cube * (np.arange(2) == np.argmax(cube.sum(axis=(0, 1))))   # on the favoured side
        # the median eccentricity, and the most probable speed and side for it (the overall
        # peak tends to sit on the edge where the eclipse is exactly at closest approach)
        e_marg = cube.sum(axis=(0, 2))
        i_e = int(np.searchsorted(np.cumsum(e_marg) / e_marg.sum(), 0.5))
        if cube[:, i_e].max() < 0.1 * cube.max():
            i_s, i_e, i_b = np.unravel_index(np.argmax(cube), cube.shape)
        else:
            i_s, i_b = np.unravel_index(np.argmax(cube[:, i_e]), cube[:, i_e].shape)
        s_best = s_grid[i_s]
        e_best = max(e_mid[i_e], abs(s_best**2 - 1) / (s_best**2 + 1) + 1e-4)
        rep = {"s": float(s_best), "e": float(e_best), "cell": (i_s, i_e, i_b), "side": i_b if told else None}
        sin_w = np.clip((s_best * np.sqrt(1 - e_best**2) - 1) / e_best, -1, 1)
        omega = np.arcsin(sin_w) if i_b == 0 else np.pi - np.arcsin(sin_w)
        p_best = float(interval(p_mid, post.sum(axis=1), (0.5,))[0])
        lead = anomaly_from_true(np.pi / 2 - omega, e_best) / (2 * np.pi) * p_best
        lead = lead if lead <= p_best / 2 else lead - p_best
        if told:
            say(f"  Representative orbit (the median eccentricity, at its most probable speed): eccentricity {e_best:.2f}, crossing {s_best:.2f} times faster than "
                f"circular, closest approach {abs(lead):.2f} days {'before' if lead >= 0 else 'after'} the eclipse.")
        else:
            say(f"  Representative orbit (the median eccentricity, at its most probable speed): eccentricity {e_best:.2f}, crossing {s_best:.2f} times faster than "
                f"circular, closest approach {abs(lead):.2f} days before or after the eclipse (unknown which).")
    if not args.circular:
        # how badly does a circular orbit do, among periods that are still open?
        circ = np.array([np.interp(np.log(v_circ[:, jm]), log_speeds, fit["peak"][:, jm], left=0, right=0)
                         for jm in range(n_m)]).T * (open_frac[:, None] > 0)
        worst = -2.0 * np.log(max(circ.max(), 1e-300))
        say(f"  A circular orbit at any period still open fits worse by chi-squared "
            f"{'more than 1000' if worst > 1000 else f'{worst:.0f}'}"
            + (" - effectively ruled out." if worst > 25 else
               " - still acceptable." if worst < 9 else " - disfavoured."))

    # ---- derived quantities ---------------------------------------------------------------------
    part = np.divide(graze3, shape3, out=np.zeros_like(shape3), where=shape3 > 0)
    grazing = float((post3 * part).sum())
    rv = 212.91 * masses[None, :] * (m1[None, :] + masses[None, :]) ** (-2 / 3) \
        * p_mid[:, None] ** (-1 / 3)
    rv_int = interval(rv.ravel(), post.ravel())
    catch = float((post * np.minimum(1.0, 27.0 / p_mid[:, None])).sum())
    say("\nWHAT THIS PREDICTS")
    say(f"  Chance the eclipse is grazing (only part of the companion crosses the star): "
        f"{100 * grazing:.0f}%")
    say(f"  Velocity swing of the star: at least {rv_int[2]:.0f} km/s "
        f"(68%: {rv_int[1]:.0f} to {rv_int[3]:.0f}; larger if eccentric)")
    if gaia:
        w = ((m1[None, :] + masses[None, :]) * (p_mid[:, None] / 365.25) ** 2) ** (1 / 3) * gaia["parallax"] \
            * np.abs(masses[None, :] / (m1[None, :] + masses[None, :]) - lightG2[None, :])
        w_int = interval(w.ravel(), post.ravel())
        say(f"  Wobble on the sky: {w_int[2]:.2f} mas (68%: {w_int[1]:.2f} to {w_int[3]:.2f})")
    say(f"  Chance that one 27-day TESS sector catches an eclipse: {100 * catch:.0f}%")
    say("  The period ranges above are envelopes. Inside them, only the "
        + ("periods listed below remain open." if sure else "intervals in"))
    if not sure:
        say("  allowed_periods.csv remain open.")

    # ---- two eclipses pin the period to their separation divided by a whole number ------------------
    if sure:
        gap, gap_error = abs(sure[0][0] - t0), float(np.hypot(sure[0][1], t0_error))
        open_now = ~ruled_out
        left = np.bincount(which, weights=open_now, minlength=n_p)
        density = post.sum(axis=1) / np.maximum(left, 1)            # probability per open trial period
        by_number = {}
        for p_lo, p_hi in intervals(periods, open_now):
            c0, c1 = np.searchsorted(periods, [p_lo, p_hi])
            members = np.arange(c0, c1)[open_now[c0:c1]]
            if len(members):
                number = int(round(gap / periods[members].mean()))
                by_number[number] = by_number.get(number, 0.0) + float(density[which[members]].sum())
        cand = pd.DataFrame([{"orbits_between": n, "period_days": gap / n, "plus_minus": gap_error / n,
                              "probability": v} for n, v in by_number.items()],
                            columns=["orbits_between", "period_days", "plus_minus", "probability"])
        cand = cand.sort_values("probability", ascending=False)
        cand.to_csv(outdir / "candidate_periods.csv", index=False)
        n_lo, n_hi = int(np.ceil(gap / args.max_period)), int(np.floor(gap / args.min_period))
        say("\nCANDIDATE PERIODS")
        say(f"  The eclipse at --t0 and the one at JD {sure[0][0]:.4f} are {gap:.4f} days apart (give or take "
            f"{max(gap_error * 1440, 0.1):.1f} minutes),")
        say("  so the period is that gap divided by a whole number.")
        if len(sure) > 1:
            say(f"  ({len(sure)} other eclipses are known in all; the periods listed fit every one of them.)")
        say(f"  Whole numbers {n_lo} to {n_hi} give periods between --min-period and --max-period. "
            f"{len(cand)} of them remain{'s' if len(cand) == 1 else ''};")
        say("  each of the others would put another eclipse on data that show none.")
        if len(cand):
            runs, numbers = [], sorted(cand.orbits_between)
            for n in numbers:
                if runs and n == runs[-1][1] + 1:
                    runs[-1][1] = n
                else:
                    runs.append([n, n])
            say("  Still open: divide by " + ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)
                + f"   (periods {gap / numbers[-1]:.2f} to {gap / numbers[0]:.2f} days)")
            basis = ["the eclipse's shape", "the chance of alignment"] + (["Gaia's astrometry"] if gaia else []) \
                + (["the velocities"] if rv_branches is not None else []) \
                + (["the companion's own eclipse"] if sec_report is not None else [])
            say("  The most probable, given " + ", ".join(basis[:-1]) + " and " + basis[-1] + ":")
            say("      divide by   period (days)   probability")
            for r in cand.head(10).itertuples():
                say(f"      {r.orbits_between:9d}   {r.period_days:13.4f}   {100 * r.probability:10.1f}%")
            say("  (all in candidate_periods.csv)")
            # Where TESS will look next, from tess-point's pointing tables, for the most probable period.
            # (triple_fit.py gives better predictions: it fits the ATLAS eclipses too.)
            now_jd = pd.Timestamp.now(tz="UTC").to_julian_date()
            future = [(s_, mid) for s_, mid in gaia_tools.tess_sectors(star["ra"], star["dec"], tic) if mid > now_jd - 14]
            top = cand.iloc[0]
            if future:
                say(f"  TESS sectors still to come for this star, with eclipses predicted from the gap / "
                    f"{int(top.orbits_between)} period ({top.period_days:.5f} +/- {top.plus_minus:.5f} d):")
                for s_, mid in future:
                    n0, n1 = np.ceil((mid - 13.7 - t0) / top.period_days), np.floor((mid + 13.7 - t0) / top.period_days)
                    whens = [(t0 + c * top.period_days, np.hypot(t0_error, abs(c) * top.plus_minus))
                             for c in np.arange(n0, n1 + 1)]
                    date = pd.Timestamp(mid - 2440587.5, unit="D").strftime("%Y %b")
                    say(f"      sector {s_} (around {date}): "
                        + (", ".join(f"BJD {w:.4f} +/- {e:.4f}" for w, e in whens) if whens
                           else "no eclipse falls in the sector"))
            elif tic:
                say("  (pip install tess-point to list the TESS sectors still to come, with predicted eclipses)")
            # Why is each one open? Either TESS never looked when it predicts another eclipse,
            # or it looked but the light curves have a hole there.
            spans = {}
            for one in sets:
                version = re.fullmatch(r"(tess_s\d+)(_\w+)?", one["name"])
                if version and one["name"] in frames:
                    first, last = float(frames[one["name"]].t.min()), float(frames[one["name"]].t.max())
                    old = spans.get(version.group(1), (first, last))
                    spans[version.group(1)] = (min(old[0], first), max(old[1], last))
            hidden, backed = {}, []
            for n in numbers:
                spots = []
                for base, (first, last) in sorted(spans.items()):
                    for cycle in range(int(np.ceil((first - t0) * n / gap)), int(np.floor((last - t0) * n / gap)) + 1):
                        when = t0 + cycle * gap / n
                        if any(abs(when - other) <= 2 * dur for other in maybe):
                            backed.append(n)
                        elif abs(when - t0) > 2 * dur and all(abs(when - e[0]) > 2 * dur for e in sure):
                            spots.append((base, when))
                hidden[n] = spots
            backed = sorted(set(backed))
            clear = [n for n in numbers if not hidden[n] and n not in backed]
            holes = [n for n in numbers if hidden[n] and n not in backed]

            def as_runs(values):
                out = []
                for n in values:
                    if out and n == out[-1][-1] + 1:
                        out[-1].append(n)
                    else:
                        out.append([n])
                return out

            def label(run):
                return f"{run[0]}" if len(run) == 1 else f"{run[0]}-{run[-1]}"

            say("")
            if backed:
                say("  Divide by " + ", ".join(label(r) for r in as_runs(backed)) + ": puts an eclipse on a possible "
                    "further eclipse reported above.")
                say("  If that dip is real, the others below are wrong; give its time with --also to apply it.")
            if clear:
                say("  Divide by " + ", ".join(label(r) for r in as_runs(clear)) + ": no further eclipse is predicted "
                    "inside any TESS sector's")
                say("  time span, so TESS can neither exclude nor support these.")
            if holes:
                say("  The rest are open only because every further eclipse they predict falls where the")
                say("  light curves have nothing to test with (a gap, or the first or last hours beside one):")
                for run in as_runs(holes):
                    parts = []
                    for base in sorted(spans):
                        times = [when for n in run for b, when in hidden[n] if b == base]
                        if times:
                            times, groups = sorted(times), []
                            for when in times:                         # stretches more than a day apart
                                if groups and when - groups[-1][-1] <= 1.0:
                                    groups[-1].append(when)
                                else:
                                    groups.append([when])
                            parts.append(f"{base} JD " + " and ".join(
                                f"{g[0]:.2f}" + (f" to {g[-1]:.2f}" if g[-1] - g[0] > 0.01 else "") for g in groups))
                    say(f"      divide by {label(run):<8}" + ";  ".join(parts))
                if clear:
                    say(f"  If those gaps hide nothing, the shortest possible period is {gap / max(clear):.2f} days "
                        f"(divide by {max(clear)}).")
                if not any(one.get("fill") for one in sets):
                    say("  Other versions of the same sectors may have points in those gaps. To fetch and use")
                    say("  them, run again with --fill-gaps (needs the internet).")

    # ---- if the dip Gaia saw is a second eclipse --------------------------------------------------
    if fits_gaia is not None:
        say(f"\nIF THE GAIA DIP (JD {', '.join(f'{dip[1]:.4f}' for dip in seen)}) IS A SECOND ECLIPSE")
        say("  Then the period is the time between the two eclipses divided by a whole number,")
        say("  and only such periods that are still open count.")
        left = np.bincount(which, weights=fits_gaia, minlength=n_p)
        cond3 = base3 * (left / per_bin)[:, None, None]
        if cond3.sum() <= 0:
            say("  No period that is still open puts an eclipse on that visit. Either the dip is")
            say("  not an eclipse, or the eclipses do not all look like the known one.")
        else:
            cond3 = cond3 / cond3.sum()
            describe(say, "orbit and companion", cond3.sum(axis=2), p_mid, masses, star, r2)
            eccentricity(cond3)
            density = cond3.sum(axis=(1, 2)) / np.maximum(left, 1)      # per trial period
            rows = []
            for p_lo, p_hi in intervals(periods, fits_gaia):
                c0, c1 = np.searchsorted(periods, [p_lo, p_hi])
                members = np.arange(c0, c1)[fits_gaia[c0:c1]]
                if len(members) == 0:
                    continue
                centre = float(periods[members].mean())
                rows.append({"orbits_between": int(round(abs(t0 - seen[0][1]) / centre)),
                             "period_days": centre,
                             "plus_minus": float(max(periods[members[-1]] - periods[members[0]],
                                                     periods[members[0]] * (periods[1] / periods[0] - 1)) / 2),
                             "probability": float(density[which[members]].sum()),
                             "other_evidence": float(other[members].max())})
            cand = pd.DataFrame(rows).sort_values("probability", ascending=False)
            cand.to_csv(outdir / "gaia_candidate_periods.csv", index=False)
            say(f"  {len(cand)} separate periods remain (all in gaia_candidate_periods.csv). The most probable:")
            for r in cand.head(8).itertuples():
                say(f"      {r.period_days:10.4f} days (+/- {r.plus_minus:.4f}): {r.orbits_between} orbits "
                    f"between the eclipses, probability {100 * r.probability:.1f}%, "
                    f"other data {r.other_evidence:+.1f}")
            supported = cand.sort_values("other_evidence", ascending=False).iloc[0]
            gap = abs(t0 - seen[0][1])
            tol = float(np.median(cand.plus_minus * cand.orbits_between))
            null = chance_support(periods, np.where(ruled_out | (other <= -0.5 * args.sigma**2), -np.inf, other), gap, tol)
            by_chance = float((null >= supported.other_evidence).mean())
            say(f"  Best supported by the other light curves: {supported.period_days:.4f} days "
                f"({supported.other_evidence:+.1f}).")
            say("  'Other data' is the summed evidence from every other measurement: positive when")
            say("  dips appear where the period predicts them. With this many candidates one of")
            say("  them always collects some. Had the Gaia dip come at a random other time, the")
            say(f"  best candidate would have done as well {100 * by_chance:.0f}% of the time"
                + (" - so this is no more than a hint." if by_chance > 0.01 else
                   " - more than chance. This period, or a whole multiple or fraction of it, is favoured."))

    # ---- one system, for drawing -------------------------------------------------------------------
    # Every number for the figures from a single system that all the evidence favours: the
    # representative orbit's eccentricity, speed and side, the most probable companion mass for that
    # orbit, the most probable period, and that system's own best eclipse shape (impact parameter and
    # variant at its crossing speed). Unlike medians taken one at a time, these belong together; unlike
    # the best-fitting eclipse alone, they also respect the velocities, RUWE and missing eclipses.
    if rep is not None and post5.sum() > 0:
        # chosen jointly, not one marginal at a time: at the median eccentricity, the most probable
        # (period, mass, speed, side) cell; then, at the exact period, the speed within the posterior's
        # support that gives that mass its best eclipse
        i_e = rep["cell"][1]
        sub = post5[:, :, :, i_e, :].copy()                             # [period, mass, speed, side]
        if rep["side"] is not None:
            sub[..., 1 - rep["side"]] = 0.0
        if sure and len(cand):
            p_sys = float(cand.iloc[0].period_days)
            near = np.abs(np.log(p_mid / p_sys))
            keep = near <= near.min() + 1e-12
            sub[~keep] = 0.0
        if sub.sum() <= 0:
            sub = post5.sum(axis=3)
        ip, jm, i_s, i_b = np.unravel_index(int(np.argmax(sub)), sub.shape)
        if not (sure and len(cand)):
            p_sys = float(p_mid[ip])
        a_sys = ((m1[jm] + masses[jm]) * (p_sys / 365.25) ** 2) ** (1 / 3) * 215.032     # solar radii
        v_c = 2 * np.pi * a_sys / p_sys                                                  # circular, Rsun/day
        row = sub[ip, jm, :, i_b]
        ok = np.where(row >= 0.1 * row.max())[0]
        shape_at = np.interp(np.log(s_grid[ok] * v_c), fit["log_speed"], fit["peak"][:, jm], left=0, right=0)
        i_s = int(ok[np.argmax(shape_at)])
        rep = dict(rep, s=float(s_grid[i_s]),
                   e=float(max(e_mid[i_e], abs(s_grid[i_s] ** 2 - 1) / (s_grid[i_s] ** 2 + 1) + 1e-4)))
        v_sys = rep["s"] * v_c                                                           # Rsun per day
        iv = int(np.argmin(np.abs(fit["log_speed"] - np.log(v_sys))))
        if fit["peak"][iv, jm] > 0:
            jv = int(fit["peak_variant"][iv, jm])
            kk, bb = float(fit["v_k"][jv]), float(fit["peak_b"][iv, jm])
            r1_sys = float(fit["v_r1"][jv])
            v_r = v_sys / r1_sys                                                         # stellar radii per day
            hours = 2 * np.sqrt(max((1 + kk) ** 2 - bb ** 2, 0.0)) / v_r * 24
            worse = -2 * np.log(float(fit["peak"][iv, jm])) * rescale
            e_s, s_s = rep["e"], rep["s"]
            sin_w = np.clip((s_s * np.sqrt(1 - e_s ** 2) - 1) / e_s, -1, 1)
            omega = np.arcsin(sin_w) if i_b == 0 else np.pi - np.arcsin(sin_w)
            lead = anomaly_from_true(np.pi / 2 - omega, e_s) / (2 * np.pi) * p_sys
            lead = lead if lead <= p_sys / 2 else lead - p_sys
            side = ("before or after" if rep["side"] is None else "before" if lead >= 0 else "after")
            t1s = float(np.atleast_1d(teff1)[jm] if np.ndim(teff1) else teff1)
            t2s = float(np.atleast_1d(teff2)[jm])
            say("\nMOST PROBABLE SYSTEM (one system that all the evidence favours; system_view.py draws it)")
            say(f"  orbit: period {p_sys:.5f} d, eccentricity {e_s:.3f}, crossing {s_s:.3f} times faster than "
                f"circular, closest approach {abs(lead):.2f} days {side} the eclipse")
            say(f"  main star: mass {m1[jm]:.3f} Msun, radius {r1_sys:.3f} Rsun, {t1s:.0f} K; companion: mass "
                f"{masses[jm]:.3f} Msun, radius {kk * r1_sys:.3f} Rsun, {t2s:.0f} K")
            say(f"  its eclipse: radius ratio {kk:.4f}, impact parameter {bb:.4f} stellar radii, crossing speed "
                f"{v_r:.4f} stellar radii per day, eclipsed star's share of the light {float(fit['v_dil'][jv]):.4f}, "
                f"length {hours:.3f} hours; fits the eclipse worse than the best fit by chi-squared {worse:.1f}")

    # ---- files and pictures ------------------------------------------------------------------------
    grid = pd.DataFrame(post, index=np.round(p_mid, 4), columns=np.round(masses, 4))
    grid.index.name = "period_days \\ mass_msun"
    grid.to_csv(outdir / "posterior_grid.csv")

    fig = plt.figure(figsize=(9.5, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1.2], height_ratios=[1.2, 4], hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    ax.pcolormesh(p_mid, masses, post.T, cmap="Blues", shading="auto")
    flat = np.sort(post.ravel())[::-1]
    cum = np.cumsum(flat)
    levels = sorted({flat[np.searchsorted(cum, q)] for q in (0.95, 0.68)})
    if len(levels) == 2:
        ax.contour(p_mid, masses, post.T, levels=levels, colors=["0.3", "k"], linewidths=[1, 1.6])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital period (days)")
    ax.set_ylabel("Companion mass (solar masses)")
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    top.step(p_mid, post.sum(axis=1), where="mid", color="k")
    top.set_yticks([])
    plt.setp(top.get_xticklabels(), visible=False)
    top.set_title(f"TIC {tic}: where the probability lies (contours hold 68% and 95%)", fontsize=10)
    side = fig.add_subplot(gs[1, 1], sharey=ax)
    side.step(post.sum(axis=0), masses, where="mid", color="k")
    side.set_xticks([])
    plt.setp(side.get_yticklabels(), visible=False)
    fig.savefig(outdir / "posterior.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    u1, u2 = limb_darkening(float(np.median(teff1)))
    kk = fit["k_best"]
    zg, hid = blocked_light(kk, u1, u2)
    xx = np.linspace(x.min(), x.max(), 600)
    speed = fit["speed_r1"]
    model = 1 - np.interp(np.hypot(fit["frac_b"] * (1 + kk), speed * xx), zg, hid, right=0.0) \
        * fit["dilute_best"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(x * 24, fn, ".", color="0.6", ms=3)
    ax.plot(xx * 24, model, color="crimson", lw=1.6)
    ax.set_xlabel("Hours from mid-eclipse")
    ax.set_ylabel("Brightness")
    ax.set_title(f"Best-fitting model: {masses[j]:.2f} Msun companion crossing at "
                 f"{fit['speed']:.0f} solar radii per day, {fit['frac_b'] * (1 + kk):.2f} stellar radii from centre",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "eclipse_fit.png", dpi=110)
    plt.close(fig)

    (outdir / "summary.txt").write_text("\n".join(say.lines) + "\n")
    print(f"\nSaved to {outdir}/")


if __name__ == "__main__":
    main()
