#!/usr/bin/env python
"""
eclipse_gallery.py - one figure showing how well an orbit fits every observed eclipse.

Top: one small panel per eclipse caught by TESS or ATLAS, in time order, each
showing the measurements against the eclipse the orbit predicts, with the
measured mid-eclipse time (and how early or late it came) and depth.
Bottom: the whole TESS and ATLAS record on one timeline, with every predicted
eclipse marked and lines running up to the panels.

Usage (on the output folder of a joint_orbit.py run, the same one system_movie.py uses):

    python eclipse_gallery.py joint_output
    python eclipse_gallery.py joint_output --period 23.67589 --per-row 6

The orbit comes from summary.txt (via system_view.py), the period from
candidate_periods.csv unless --period is given, the eclipse time from
summary.txt unless --t0 is given. system_view.py and system_movie.py must sit
next to this script.

Output, in the folder: eclipse_gallery.png (and .pdf), eclipse_table.csv.

How the numbers are measured
----------------------------
Each panel's eclipse is fitted with the orbit's own eclipse shape, allowed to
shift in time and to scale in depth (a weighted least-squares fit over a grid
of shifts). The quoted errors come from where chi-squared rises by one, after
the errors are enlarged if the fit is worse than they say. When too little of
the eclipse is covered to pin the time down (a few ATLAS visits, say), the
time is held at the prediction and only the depth is fitted; the panel then
says "time not measured".
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import system_view as sv            # noqa: E402
import system_movie as sm           # noqa: E402

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
COLOURS = {"TESS": "#2a78d6", "ATLAS o": "#eb6834", "ATLAS c": "#1baf7a"}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_all(folder):
    """Every light curve, with errors: list of dicts (name, kind, label, t, f, e)."""
    out = []
    for path in sorted(folder.glob("tess_s*.csv")) + sorted(folder.glob("atlas_*.csv")):
        if path.name.startswith("atlas_check"):
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        t = df[cols.get("bjd", df.columns[0])].to_numpy(float)
        f = df[cols.get("flux", df.columns[1])].to_numpy(float)
        e = df[cols["flux_err"]].to_numpy(float) if "flux_err" in cols else np.full(len(t), np.nan)
        t = t + (2457000.0 if np.nanmedian(t) < 100000 else 0.0)
        good = np.isfinite(t) & np.isfinite(f) & (f > 0)
        t, f, e = t[good], f[good], e[good]
        if len(t) < 10:
            continue
        med = np.median(f)
        f, e = f / med, e / med
        if not np.isfinite(e).all() or (e <= 0).any():          # no usable errors: use the scatter
            e = np.full(len(f), 1.4826 * np.median(np.abs(np.diff(f))) / np.sqrt(2))
        kind = "TESS" if path.name.startswith("tess") else "ATLAS"
        if kind == "TESS":
            sector = re.match(r"tess_s(\d+)", path.stem).group(1)
            version = path.stem.split("_", 2)[2] if path.stem.count("_") >= 2 else ""
            label = f"TESS S{int(sector)}" + (f" ({version.upper()})" if version else "")
        else:
            label = "ATLAS " + path.stem.split("_")[-1]
        order = np.argsort(t)
        t, f, e = t[order], f[order], e[order]
        jitter = 0.0
        if kind == "ATLAS":
            e, jitter = atlas_noise(t, f, e)
        out.append({"name": path.stem, "kind": kind, "label": label, "t": t, "f": f, "e": e, "jitter": jitter})
    return out


def atlas_noise(t, f, e, gap=0.05):
    """ATLAS errors as the data show them: the scatter of visits within a session (a set of visits
    less than an hour apart), and the extra zero-point offset each session shares (its 'jitter'),
    measured from the scatter of session means. Eclipses are too rare to bias the medians."""
    sess = session_ids(t, gap)
    n = np.bincount(sess)
    mean = np.bincount(sess, f) / n
    multi = n[sess] >= 2
    within = ((f - mean[sess]) * np.sqrt(n[sess] / np.maximum(n[sess] - 1, 1)))[multi]
    if len(within) > 20:
        e = e * max(1.4826 * np.median(np.abs(within)) / np.median(e), 0.5)
    good = n >= 2
    if good.sum() < 6:
        return e, 0.0
    sm = mean[good]
    spread = 1.4826 * np.median(np.abs(sm - np.median(sm)))
    white = np.median(e) / np.sqrt(np.median(n[good]))
    return e, float(np.sqrt(max(spread**2 - white**2, 0.0)))


def session_ids(t, gap=0.05):
    """Number the runs of visits less than `gap` days apart (t sorted)."""
    return np.concatenate([[0], np.cumsum(np.diff(t) > gap)]).astype(int)


def correlated(e, groups, jitter):
    """A function q(a, b) = a' C^-1 b for C = diag(e^2) plus jitter^2 shared within each group."""
    w = 1.0 / e**2
    if groups is None:
        return lambda a, b: float(np.sum(w * a * b))
    g = np.unique(groups, return_inverse=True)[1]
    j2 = np.bincount(g, jitter**2) / np.bincount(g)          # one jitter per group
    sw = np.bincount(g, w)
    shrink = j2 / (1.0 + j2 * sw)

    def q(a, b):
        return float(np.sum(w * a * b) - np.sum(shrink * np.bincount(g, w * a) * np.bincount(g, w * b)))
    return q


def main_versions(curves):
    """One TESS curve per sector (the main version where there is one), and every ATLAS filter."""
    keep, sectors = [], {}
    for c in curves:
        if c["kind"] == "TESS":
            sector = re.match(r"tess_s(\d+)", c["name"]).group(1)
            if sector not in sectors or c["name"].count("_") < sectors[sector]["name"].count("_"):
                sectors[sector] = c
        else:
            keep.append(c)
    return sorted(sectors.values(), key=lambda c: c["t"][0]) + keep


# ----------------------------------------------------------------------------
# Measuring one eclipse
# ----------------------------------------------------------------------------
def measure(dt, f, e, profile, half, fit_time, groups=None, jitter=None):
    """Fit 1 - s * profile(dt - tau). Returns tau, tau_err, s, s_err, timed (bool).
    With groups and jitter (ATLAS), visits in one group share a zero-point offset of that size."""
    q = correlated(e, groups, jitter if jitter is not None else np.zeros(len(e)))

    def fit(tau):
        d = profile(dt - tau)
        sdd = q(d, d)
        if sdd <= 0:
            return np.inf, 0.0, np.inf
        s = q(d, 1 - f) / sdd
        r = 1 - f - s * d
        return q(r, r), float(s), float(1 / np.sqrt(sdd))

    taus = np.linspace(-half, half, 401) if fit_time else np.zeros(1)
    chi = np.array([fit(tau)[0] for tau in taus])
    i = int(np.argmin(chi))
    chi_min, s, s_err = fit(taus[i])
    dof = max(len(f) - (2 if fit_time else 1), 1)
    stretch = max(chi_min / dof, 1.0)
    s_err *= np.sqrt(stretch)
    if not fit_time:
        return 0.0, np.nan, s, s_err, False
    inside = np.where(chi <= chi_min + stretch)[0]
    lo, hi = taus[inside.min()], taus[inside.max()]
    step = taus[1] - taus[0]
    tau_err = 0.5 * (hi - lo)
    if 0 < i < len(taus) - 1:                                   # parabola through the lowest three points
        curve = (chi[i - 1] - 2 * chi[i] + chi[i + 1]) / step**2
        if curve > 0:
            tau_err = max(tau_err, np.sqrt(2 * stretch / curve))
    in_eclipse = np.abs(dt - taus[i]) < half
    timed = (inside.min() > 0 and inside.max() < len(taus) - 1 and in_eclipse.sum() >= 4
             and (dt[in_eclipse] < taus[i]).any() and (dt[in_eclipse] > taus[i]).any())
    if not timed:
        return measure(dt, f, e, profile, half, False, groups, jitter)
    return float(taus[i]), float(tau_err), s, s_err, True


def stack_parts(parts, curves):
    """Concatenate (label, kind, dt, f, e) parts; for ATLAS also session groups and jitter per visit."""
    dt = np.concatenate([q[2] for q in parts])
    f = np.concatenate([q[3] for q in parts])
    e = np.concatenate([q[4] for q in parts])
    if parts[0][1] == "TESS":
        return dt, f, e, None, None
    jit = {c["label"]: c.get("jitter", 0.0) for c in curves}
    groups, jitter = [], []
    for n, q in enumerate(parts):
        order = np.argsort(q[2])
        ids = np.empty(len(q[2]), int)
        ids[order] = session_ids(q[2][order])
        groups.append(ids + 100000 * n)
        jitter.append(np.full(len(q[2]), jit.get(q[0], 0.0)))
    return dt, f, e, np.concatenate(groups), np.concatenate(jitter)


def local_level(dt, f, e, half):
    """Divide out a straight line fitted to the points outside the eclipse (TESS)."""
    out = np.abs(dt) > 1.3 * half
    if out.sum() < 10:
        return f, e
    slope, level = np.polyfit(dt[out], f[out], 1, w=1 / e[out])
    base = level + slope * dt
    return f / base, e / base


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Every observed eclipse against the orbit, on one figure.")
    p.add_argument("folder", help="output folder of a joint_orbit.py run")
    p.add_argument("--period", type=float, default=None, help="orbital period, days (default: candidate_periods.csv)")
    p.add_argument("--t0", type=float, default=None, help="a mid-eclipse time, BTJD or JD (default: summary.txt)")
    p.add_argument("--per-row", type=int, default=8, help="eclipse panels per row (default 8)")
    p.add_argument("--hours", type=float, default=None, help="half-width of each panel, hours (default: 3 eclipse lengths)")
    p.add_argument("--out", default=None, help="output file (default: eclipse_gallery.png in the folder)")
    args = p.parse_args()

    folder = Path(args.folder)
    run = sv.read_run(folder)
    text = (folder / "summary.txt").read_text()
    if args.t0 is not None:
        t0 = args.t0 + 2457000.0 if args.t0 < 100000 else args.t0
    else:
        found = re.findall(r"KNOWN ECLIPSE in \S+: centre JD ([\d.]+)", text)
        if not found:
            sys.exit("No known eclipse in summary.txt; give --t0.")
        t0 = float(found[-1])
    period = args.period
    if period is None and (folder / "candidate_periods.csv").exists():
        cand = pd.read_csv(folder / "candidate_periods.csv").sort_values("probability", ascending=False)
        period = float(cand.period_days.iloc[0])
        if len(cand) > 1:
            print(f"Using the most probable of {len(cand)} candidate periods: {period:.5f} days")
    if period is None:
        sys.exit("Give --period: there is no candidate_periods.csv in the folder.")
    run["period"][0] = period
    geo = sv.orbit_geometry(run)
    orbit = sm.Orbit(run, geo, period, t0)

    # the predicted eclipse's shape and length (the dip at t0 only, not a companion eclipse nearby)
    tt = np.linspace(-1.0, 1.0, 8001)
    model = orbit.brightness(t0 + tt)
    # On an eccentric orbit the deepest point of the eclipse is not exactly at conjunction. t0 is a
    # measured mid-eclipse, so centre the model on its own middle (the centre of its missing light).
    near = np.abs(tt) < 0.5
    lost = np.clip(1.0 - model, 0, None) * near
    if lost.sum() > 0:
        shift = float(np.sum(tt * lost) / lost.sum())
        model = orbit.brightness(t0 + shift + tt)
    mid = len(tt) // 2
    dipping = model < 0.999
    if not dipping[mid]:
        sys.exit("The orbit predicts no eclipse at t0; check summary.txt and --t0.")
    lo = mid - int(np.argmin(dipping[mid::-1])) + 1
    hi = mid + int(np.argmin(dipping[mid:])) - 1
    half = 0.5 * (tt[hi] - tt[lo])
    model_depth = 1.0 - float(model[mid])
    window = args.hours / 24 if args.hours else max(3 * half, 0.25)

    def profile(dt):
        return 1.0 - np.interp(dt, tt, model, left=1.0, right=1.0)

    curves = read_all(folder)
    if not curves:
        sys.exit(f"No tess_s*.csv or atlas_*.csv light curves in {folder}.")

    # ---- every predicted eclipse that some light curve covers ------------------------------------
    first = min(c["t"][0] for c in curves)
    last = max(c["t"][-1] for c in curves)
    cycles = range(int(np.floor((first - t0) / period)), int(np.ceil((last - t0) / period)) + 1)
    events = []
    for k in cycles:
        when = t0 + k * period
        tess = [c for c in curves if c["kind"] == "TESS" and (np.abs(c["t"] - when) < half).sum() >= 3]
        atlas = [c for c in curves if c["kind"] == "ATLAS" and (np.abs(c["t"] - when) < half).any()]
        if tess:
            most = max((np.abs(c["t"] - when) < half).sum() for c in tess)
            good = [c for c in tess if (np.abs(c["t"] - when) < half).sum() >= 0.6 * most]
            best = min(good, key=lambda c: c["name"].count("_"))    # the cleaned light curve if it will do
            sel = np.abs(best["t"] - when) < window
            dt = best["t"][sel] - when
            f, e = local_level(dt, best["f"][sel], best["e"][sel], half)
            parts = [(best["label"], "TESS", dt, f, e)]
        elif sum((np.abs(c["t"] - when) < half).sum() for c in atlas) >= 2:
            parts = []
            for c in atlas:
                sel = np.abs(c["t"] - when) < window
                parts.append((c["label"], c["label"], c["t"][sel] - when, c["f"][sel], c["e"][sel]))
        else:
            continue
        dt, f, e, groups, jitter = stack_parts(parts, curves)
        tau, tau_err, s, s_err, timed = measure(dt, f, e, profile, half, True, groups, jitter)
        events.append({"k": k, "predicted": when, "parts": parts, "tau": tau, "tau_err": tau_err,
                       "depth": s * model_depth, "depth_err": s_err * model_depth, "timed": timed,
                       "source": parts[0][0] if parts[0][1] == "TESS" else "ATLAS"})
    if not events:
        sys.exit("None of the light curves covers a predicted eclipse.")

    table = pd.DataFrame([{
        "orbit": ev["k"], "source": ev["source"],
        "predicted_bjd": ev["predicted"],
        "measured_bjd": ev["predicted"] + ev["tau"] if ev["timed"] else np.nan,
        "time_error_min": ev["tau_err"] * 1440 if ev["timed"] else np.nan,
        "o_minus_c_min": ev["tau"] * 1440 if ev["timed"] else np.nan,
        "depth_pct": 100 * ev["depth"], "depth_error_pct": 100 * ev["depth_err"],
        "model_depth_pct": 100 * model_depth,
        "points": int(sum(len(q[2]) for q in ev["parts"]))} for ev in events])
    table.to_csv(folder / "eclipse_table.csv", index=False)

    # ---- the figure ----------------------------------------------------------------------------------
    n = len(events)
    per_row = max(1, min(args.per_row, n))
    rows = int(np.ceil(n / per_row))
    width = max(11.0, 2.15 * per_row + 0.8)
    height = 3.3 * rows + 3.3
    fig = plt.figure(figsize=(width, height), facecolor=SURFACE)
    gs = fig.add_gridspec(rows + 1, per_row, height_ratios=[2.6] * rows + [2.4],
                          hspace=0.9 if rows > 1 else 0.62, wspace=0.12,
                          left=0.06, right=0.985, top=1 - 1.55 / height, bottom=0.6 / height)

    # a shared brightness scale: the model's dip plus room for the noisier data
    spread = np.concatenate([q[3] for ev in events for q in ev["parts"]])
    low = min(1 - 1.35 * model_depth, np.nanpercentile(spread, 1) - 0.005)
    high = max(1 + 0.25 * model_depth, np.nanpercentile(spread, 99) + 0.005)

    hours = np.linspace(-window, window, 1200)
    axes = []
    for i, ev in enumerate(events):
        ax = fig.add_subplot(gs[i // per_row, i % per_row])
        axes.append(ax)
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7, length=2)
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
        ax.plot(hours * 24, 1 - profile(hours), color=INK, lw=1.3, zorder=3, label="orbit predicts")
        for label, key, dt, f, e in ev["parts"]:
            colour = COLOURS.get(key, COLOURS["TESS"])
            if key == "TESS" and len(dt) > 120:              # raw points faint, 15-minute bins on top
                ax.plot(dt * 24, f, ".", ms=1.5, color=colour, alpha=0.25, zorder=1)
                edges = np.arange(-window, window + 1e-9, 15 / 1440)
                idx = np.digitize(dt, edges)
                bt = [dt[idx == j].mean() for j in np.unique(idx) if (idx == j).sum() >= 3]
                bf = [f[idx == j].mean() for j in np.unique(idx) if (idx == j).sum() >= 3]
                ax.plot(np.array(bt) * 24, bf, "o", ms=3.2, color=colour, mec=SURFACE, mew=0.5, zorder=4)
            else:
                ax.errorbar(dt * 24, f, yerr=e, fmt="o", ms=3.6, color=colour, ecolor=colour, elinewidth=0.6,
                            mec=SURFACE, mew=0.5, alpha=0.9, zorder=4)
        if ev["timed"]:
            ax.axvline(ev["tau"] * 24, color=INK2, lw=0.7, ls=(0, (2, 2)), zorder=2)
        ax.set_xlim(-window * 24, window * 24)
        ax.set_ylim(low, high)
        if i % per_row:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Brightness", color=INK2, fontsize=8)
        ax.set_xlabel("Hours from prediction", color=INK2, fontsize=7)
        date = pd.Timestamp(ev["predicted"] - 2440587.5, unit="D").strftime("%Y-%m-%d")
        if ev["timed"]:
            err_min = ev["tau_err"] * 1440
            digits = 0 if err_min >= 1 else 1
            t_line = (f"mid BJD {ev['predicted'] + ev['tau']:.4f} ± {err_min:.{digits}f} min\n"
                      f"O−C {ev['tau'] * 1440:+.{digits}f} min")
        else:
            t_line = f"predicted BJD {ev['predicted']:.4f}\ntime not measured (too few points)"
        d_line = f"depth {100 * ev['depth']:.1f} ± {100 * ev['depth_err']:.1f}%"
        ax.annotate(t_line + "\n" + d_line, xy=(0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", fontsize=6.8, color=INK2, va="bottom", ha="left", linespacing=1.3)
        ax.annotate(f"{i + 1}  {date}  ·  {ev['source']}", xy=(0, 1), xycoords="axes fraction", xytext=(0, 38),
                    textcoords="offset points", fontsize=8.5, color=INK, va="bottom", ha="left")

    # ---- the whole record -----------------------------------------------------------------------------
    axt = fig.add_subplot(gs[rows, :])
    axt.set_facecolor(SURFACE)
    for side in ("top", "right"):
        axt.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axt.spines[side].set_color(GRID)
    axt.tick_params(colors=INK2, labelsize=8, length=2)

    def year(t):
        return 2000.0 + (np.asarray(t) - 2451544.5) / 365.25

    shown = set()
    for c in main_versions(curves):
        key = "TESS" if c["kind"] == "TESS" else c["label"]
        colour = COLOURS.get(key, COLOURS["TESS"])
        legend = key if key not in shown else None
        shown.add(key)
        if c["kind"] == "TESS":                                     # hour bins keep the plot light
            edges = np.arange(c["t"][0], c["t"][-1] + 1 / 24, 1 / 24)
            idx = np.digitize(c["t"], edges)
            counts = np.bincount(idx)
            sums = np.bincount(idx, weights=c["f"])
            ok = counts > 0
            tb = np.bincount(idx, weights=c["t"])[ok] / counts[ok]
            axt.plot(year(tb), sums[ok] / counts[ok], ".", ms=1.6, color=colour, label=legend, zorder=3)
        else:
            axt.plot(year(c["t"]), c["f"], ".", ms=2.2, color=colour, alpha=0.55, label=legend, zorder=2)
    span = (year(first) - 0.05, year(last) + 0.05)
    axt.set_xlim(*span)
    axt.set_ylim(low, high)
    axt.set_ylabel("Brightness", color=INK2, fontsize=8)
    axt.set_xlabel("Year", color=INK2, fontsize=8)
    every = [t0 + k * period for k in cycles]
    axt.vlines(year(every), high - 0.06 * (high - low), high, color=GRID, lw=0.7, zorder=1)
    for i, ev in enumerate(events):
        x = year(ev["predicted"])
        axt.vlines(x, high - 0.09 * (high - low), high, color=INK, lw=1.0, zorder=4)
        if rows > 1:
            axt.text(x, high, f"{i + 1}", color=INK, fontsize=7, ha="center", va="bottom")
        else:
            con = ConnectionPatch(xyA=(0.5, -0.27), coordsA=axes[i].transAxes,
                                  xyB=(x, high), coordsB=axt.transData,
                                  color=MUTED, lw=0.6, alpha=0.8, zorder=0)
            con.set_in_layout(False)
            fig.add_artist(con)
    leg = axt.legend(loc="lower left", fontsize=7.5, frameon=False, markerscale=4, ncol=len(shown))
    for t in leg.get_texts():
        t.set_color(INK2)
    axt.text(0.995, 0.04, "short ticks: every predicted eclipse;  tall ticks: the eclipses shown above",
             transform=axt.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)

    # the eclipse's length as the table gives it: first to last contact of the eclipse joint_orbit.py
    # fitted (summary.txt); without that line, where the drawn model is more than 0.1% deep
    fitted = re.findall(r"BEST-FIT ECLIPSE: .*?length ([\d.]+) hours", text)
    length_h = float(fitted[-1]) if fitted else 2 * half * 24
    star = run["star"]
    name = f"TIC {star.get('tic', '')}".strip()
    timed = table.dropna(subset=["o_minus_c_min"])
    rms = f"; timing scatter {np.sqrt(np.mean(timed.o_minus_c_min**2)):.0f} min (rms O−C)" if len(timed) >= 2 else ""
    fig.text(0.06, 1 - 0.2 / height, f"{name}: {n} eclipses against one orbit", color=INK,
             fontsize=13, va="top")
    fig.text(0.06, 1 - 0.5 / height,
             f"period {period:.5f} d, eccentricity {geo['e']:.2f}, companion {geo['m2']:.2f} Msun; "
             f"predicted depth {100 * model_depth:.1f}%, length {length_h:.1f} h{rms}",
             color=INK2, fontsize=8.5, va="top")

    out = Path(args.out) if args.out else folder / "eclipse_gallery.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    fig.savefig(out.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"{n} eclipses; saved {out}, {out.with_suffix('.pdf').name} and eclipse_table.csv")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
