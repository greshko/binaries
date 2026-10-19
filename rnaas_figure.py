#!/usr/bin/env python
"""
rnaas_figure.py - one figure, for a Research Note of the AAS, covering several eclipsing binaries.

For each star (one joint_orbit.py output folder each) the figure has a block with:
  * all its observed eclipses stacked on the orbit's ephemeris, against the
    model eclipse: every eclipse colored by when it was observed;
  * each eclipse on its own, in a near-square grid (2 x 2 for four, 3 x 3 for
    nine), in the same colors, numbered in time order;
  * the whole TESS and ATLAS record on one timeline, with every eclipse shown
    marked by its number and color.

Each star gets its own color scale, running through it in time order (by
default mako for the first star and rocket for the second, from seaborn; the
palest and darkest ends are left out so every color shows on white). The
stacked panel therefore shows which observation contributes what.

Usage (system_view.py, system_movie.py and eclipse_gallery.py beside it):

    python rnaas_figure.py joint_332229223 joint_149762453
    python rnaas_figure.py joint_332229223 joint_149762453 --out rnaas_fig1 --hours 5

Writes OUT.pdf (vector, for the journal) and OUT.png (300 dpi), default
rnaas_figure.*, in the current folder, plus OUT_eclipses.csv with every
eclipse's number, time, source, depth and timing.

The model curve, eclipse times and depths are the same as eclipse_gallery.py's:
each eclipse is fitted with the orbit's eclipse shape, free to shift in time
and scale in depth; eclipses with too few points to time are held at the
prediction. The stacked panel puts every eclipse on the ephemeris, not on its
own fitted time, so it shows how well the period predicts them all.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

import eclipse_gallery as eg  # noqa: E402
import system_movie as sm  # noqa: E402
import system_view as sv  # noqa: E402

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#dcdbd6"
TESS_GRAY, ATLAS_GRAY = "#5f5e5a", "#b9b8b2"
FALLBACK = {"mako": ["#2b1d40", "#3b4b8f", "#3484a0", "#40b3a6", "#a8dfb7"],     # approximations, if
            "rocket": ["#3a1a40", "#7a1f4f", "#c8314b", "#ef7a55", "#f6c1a0"]}   # seaborn is missing


def color_scale(name, lo=0.15, hi=0.78):
    """A sequential scale with its palest and darkest ends cut off."""
    try:
        import seaborn as sns
        cmap = sns.color_palette(name, as_cmap=True)
    except Exception:
        cmap = plt.get_cmap(name) if name in plt.colormaps() else \
            LinearSegmentedColormap.from_list(name, FALLBACK.get(name, FALLBACK["mako"]))
    return lambda x: cmap(lo + (hi - lo) * np.clip(x, 0, 1))


# ----------------------------------------------------------------------------
# One star: the same measurements as eclipse_gallery.py
# ----------------------------------------------------------------------------
def gather(folder, hours=None):
    run = sv.read_run(folder)
    text = (folder / "summary.txt").read_text()
    t0 = float(re.findall(r"KNOWN ECLIPSE in \S+: cent(?:re|er) JD ([\d.]+)", text)[-1])
    cand = pd.read_csv(folder / "candidate_periods.csv").sort_values("probability", ascending=False)
    period = float(cand.period_days.iloc[0])
    run["period"][0] = period
    geo = sv.orbit_geometry(run)
    orbit = sm.Orbit(run, geo, period, t0)
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
    lo = mid - int(np.argmin(dipping[mid::-1])) + 1
    hi = mid + int(np.argmin(dipping[mid:])) - 1
    half = 0.5 * (tt[hi] - tt[lo])
    model_depth = 1.0 - float(model[mid])
    window = hours / 24 if hours else max(2.2 * half, 0.2)

    def profile(dt):
        return 1.0 - np.interp(dt, tt, model, left=1.0, right=1.0)

    curves = eg.read_all(folder)
    first, last = min(c["t"][0] for c in curves), max(c["t"][-1] for c in curves)
    events = []
    for k in range(int(np.floor((first - t0) / period)), int(np.ceil((last - t0) / period)) + 1):
        when = t0 + k * period
        tess = [c for c in curves if c["kind"] == "TESS" and (np.abs(c["t"] - when) < half).sum() >= 3]
        atlas = [c for c in curves if c["kind"] == "ATLAS" and (np.abs(c["t"] - when) < half).any()]
        if tess:
            most = max((np.abs(c["t"] - when) < half).sum() for c in tess)
            good = [c for c in tess if (np.abs(c["t"] - when) < half).sum() >= 0.6 * most]
            best = min(good, key=lambda c: c["name"].count("_"))
            sel = np.abs(best["t"] - when) < max(window, 3 * half)
            dt = best["t"][sel] - when
            f, e = eg.local_level(dt, best["f"][sel], best["e"][sel], half)
            keep = np.abs(dt) < window
            parts = [(best["label"], "TESS", dt[keep], f[keep], e[keep])]
            source = best["label"].split(" (")[0]
        elif sum((np.abs(c["t"] - when) < half).sum() for c in atlas) >= 2:
            parts = []
            for c in atlas:
                sel = np.abs(c["t"] - when) < window
                parts.append((c["label"], "ATLAS", c["t"][sel] - when, c["f"][sel], c["e"][sel]))
            source = "ATLAS"
        else:
            continue
        dt, f, e, groups, jitter = eg.stack_parts(parts, curves)
        tau, tau_err, s, s_err, timed = eg.measure(dt, f, e, profile, half, True, groups, jitter)
        events.append({"k": k, "when": when, "dt": dt, "f": f, "e": e, "kind": parts[0][1], "source": source,
                       "tau": tau, "tau_err": tau_err, "timed": timed,
                       "depth": s * model_depth, "depth_err": s_err * model_depth})
    star = run["star"]
    return {"name": f"TIC {star.get('tic', folder.name)}", "period": period, "t0": t0, "events": events,
            "curves": eg.main_versions(curves), "profile": profile, "half": half, "window": window,
            "model_depth": model_depth}


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def style(ax, size=6.5):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=size, length=2, width=0.6, pad=1.5)


def binned(dt, f, width):
    key = np.floor(dt / width).astype(int)
    out_t, out_f = [], []
    for k in np.unique(key):
        sel = key == k
        if sel.sum() >= 2:
            out_t.append(dt[sel].mean())
            out_f.append(f[sel].mean())
    return np.array(out_t), np.array(out_f)


def tess_points(ax, dt, f, color, ms, raw=True):
    """TESS points: 10-minute bins over the raw points when the cadence is short, else the points."""
    cadence = np.median(np.diff(np.sort(dt))) * 1440 if len(dt) > 2 else 10
    if cadence < 5:
        if raw:
            ax.plot(dt * 24, f, ".", ms=0.8, color=color, alpha=0.25, zorder=2)
        bt, bf = binned(dt, f, 10 / 1440)
        ax.plot(bt * 24, bf, "o", ms=ms, color=color, mec="none", zorder=3)
    else:
        ax.plot(dt * 24, f, "o", ms=ms, color=color, mec="none", zorder=3)


def draw_star(subfig, star, scale, letter):
    ev = star["events"]
    n = len(ev)
    colors = [scale(i / max(n - 1, 1)) for i in range(n)]
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    w = star["window"] * 24
    hours = np.linspace(-star["window"], star["window"], 800)
    curve = 1 - star["profile"](hours)
    allf = np.concatenate([e["f"] for e in ev])
    lo = min(1 - 1.35 * star["model_depth"], np.nanpercentile(allf, 0.5) - 0.004)
    hi = max(1 + 0.30 * star["model_depth"], np.nanpercentile(allf, 99.5) + 0.004)

    outer = subfig.add_gridspec(2, 2, height_ratios=[rows * 1.0, 0.95], width_ratios=[1.05, 1.35],
                                hspace=0.38, wspace=0.16, left=0.075, right=0.99, top=0.9, bottom=0.11)
    subfig.text(0.0, 0.985, f"({letter}) {star['name']}", fontsize=9, color=INK, va="top", fontweight="bold")
    subfig.text(0.205, 0.983, f"P = {star['period']:.5f} d;  {n} eclipses (TESS and ATLAS), numbered and colored "
                "in time order", fontsize=6.5, color=INK2, va="top")

    # ---- stacked ----
    ax = subfig.add_subplot(outer[0, 0])
    style(ax, 6.5)
    for i, e in enumerate(ev):
        if e["kind"] == "TESS":
            tess_points(ax, e["dt"], e["f"], colors[i], 2.2, raw=False)
        else:                                       # survey visits: lighter, so TESS stays readable
            ax.errorbar(e["dt"] * 24, e["f"], yerr=e["e"], fmt="o", ms=2.4, color=colors[i], mec="white",
                        mew=0.3, elinewidth=0.4, alpha=0.55, zorder=2)
    ax.plot(hours * 24, curve, color=INK, lw=0.9, zorder=5)
    ax.set_xlim(-w, w)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Hours from predicted mid-eclipse", fontsize=6.5, color=INK2, labelpad=1)
    ax.set_ylabel("Relative flux", fontsize=6.5, color=INK2, labelpad=1)
    ax.set_title("All eclipses on the ephemeris (TESS in 10-min bins)", fontsize=6.5, color=INK2, loc="left", pad=3)

    # ---- each eclipse ----
    grid = outer[0, 1].subgridspec(rows, cols, hspace=0.32, wspace=0.08)
    for i, e in enumerate(ev):
        a = subfig.add_subplot(grid[i // cols, i % cols])
        style(a, 5)
        if e["kind"] == "TESS":
            tess_points(a, e["dt"], e["f"], colors[i], 1.6)
        else:
            a.errorbar(e["dt"] * 24, e["f"], yerr=e["e"], fmt="o", ms=2.2, color=colors[i], mec="white", mew=0.3,
                       elinewidth=0.5, zorder=3)
        a.plot(hours * 24, curve, color=INK, lw=0.6, zorder=4)
        a.set_xlim(-w, w)
        a.set_ylim(lo, hi)
        if i % cols:
            a.set_yticklabels([])
        if i // cols < rows - 1 and i + cols < n:
            a.set_xticklabels([])
        date = pd.Timestamp(e["when"] - 2440587.5, unit="D").strftime("%Y-%m-%d")
        a.annotate(f"{i + 1}", (0, 1), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                   fontsize=6.5, color=colors[i], fontweight="bold", va="bottom")
        a.annotate(f"{date}  {e['source']}" if cols <= 3 else date, (0, 1), xycoords="axes fraction", xytext=(9, 2.3),
                   textcoords="offset points", fontsize=4.8, color=INK2, va="bottom")
        a.text(0.96, 0.05, f"{100 * e['depth']:.1f}%" + (f"\n{e['tau'] * 1440:+.0f} min" if e["timed"] else ""),
               transform=a.transAxes, fontsize=4.6, color=INK2, ha="right", va="bottom", linespacing=1.15)

    # ---- timeline ----
    ax = subfig.add_subplot(outer[1, :])
    style(ax, 6.5)

    def year(t):
        return 2000.0 + (np.asarray(t) - 2451544.5) / 365.25

    for c in star["curves"]:
        if c["kind"] == "TESS":
            tb, fb = binned(c["t"] - c["t"][0], c["f"], 1 / 24)
            ax.plot(year(tb + c["t"][0]), fb, ".", ms=1.0, color=TESS_GRAY, zorder=3)
        else:
            ax.plot(year(c["t"]), c["f"], ".", ms=1.0, color=ATLAS_GRAY, zorder=2)
    span = (min(c["t"][0] for c in star["curves"]), max(c["t"][-1] for c in star["curves"]))
    ax.set_xlim(year(span[0]) - 0.04, year(span[1]) + 0.04)
    ax.set_ylim(lo, hi + 0.55 * (hi - lo))
    top = hi + 0.12 * (hi - lo)
    order = np.argsort([e["when"] for e in ev])
    for i, e in enumerate(ev):
        x = year(e["when"])
        ax.plot([x, x], [lo, top], color=colors[i], lw=0.5, alpha=0.5, zorder=1)
        lift = 0.18 * (hi - lo) * (i % 2) if n > 6 else 0.0          # stagger crowded labels
        ax.plot(x, top + lift, "v", ms=3.5, color=colors[i], zorder=5)
        ax.text(x, top + lift + 0.07 * (hi - lo), f"{i + 1}", fontsize=5.5, color=colors[i], ha="center",
                va="bottom", fontweight="bold")
    ax.set_ylabel("Relative flux", fontsize=6.5, color=INK2, labelpad=1)
    ax.set_xlabel("Year", fontsize=6.5, color=INK2, labelpad=1)
    ax.plot([], [], "o", ms=2, color=TESS_GRAY, label="TESS (1-h bins)")
    ax.plot([], [], "o", ms=2, color=ATLAS_GRAY, label="ATLAS")
    leg = ax.legend(loc="lower left", fontsize=5.5, frameon=False, markerscale=1, ncol=2, handletextpad=0.2)
    for t in leg.get_texts():
        t.set_color(INK2)
    del order


def main():
    p = argparse.ArgumentParser(description="One Research-Note figure for several eclipsing binaries.")
    p.add_argument("folders", nargs="+", help="joint_orbit.py output folders, one per star")
    p.add_argument("--scales", nargs="*", default=["mako", "rocket", "viridis", "magma"],
                   help="color scale for each star in turn (default: mako rocket ...)")
    p.add_argument("--hours", type=float, default=None, help="half-width of the eclipse panels, hours")
    p.add_argument("--width", type=float, default=7.1, help="figure width, inches (default 7.1, a full AAS page width)")
    p.add_argument("--out", default="rnaas_figure", help="output name without extension")
    args = p.parse_args()

    stars = [gather(Path(f), args.hours) for f in args.folders]
    plt.rcParams.update({"font.family": "sans-serif", "pdf.fonttype": 42, "ps.fonttype": 42})
    heights = []
    for s in stars:
        n = len(s["events"])
        rows = int(np.ceil(n / int(np.ceil(np.sqrt(n)))))
        heights.append(1.35 * rows + 1.75)
    fig = plt.figure(figsize=(args.width, sum(heights)), facecolor="white")
    subs = fig.subfigures(len(stars), 1, height_ratios=heights, hspace=0.02) if len(stars) > 1 else [fig]
    rows = []
    for k, (s, sub) in enumerate(zip(stars, np.atleast_1d(subs))):
        draw_star(sub, s, color_scale(args.scales[k % len(args.scales)]), "abcdefgh"[k])
        for i, e in enumerate(s["events"]):
            rows.append({"star": s["name"], "number": i + 1, "orbit": e["k"], "source": e["source"],
                         "predicted_bjd": e["when"],
                         "measured_bjd": e["when"] + e["tau"] if e["timed"] else np.nan,
                         "time_error_min": e["tau_err"] * 1440 if e["timed"] else np.nan,
                         "o_minus_c_min": e["tau"] * 1440 if e["timed"] else np.nan,
                         "depth_pct": 100 * e["depth"], "depth_error_pct": 100 * e["depth_err"]})
    out = Path(args.out)
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    fig.savefig(out.with_suffix(".png"), dpi=300, facecolor="white")
    pd.DataFrame(rows).to_csv(out.parent / f"{out.name}_eclipses.csv", index=False)
    plt.close(fig)
    print(f"Saved {out.with_suffix('.pdf')}, {out.with_suffix('.png')} and {out.name}_eclipses.csv "
          f"({args.width:.1f} x {sum(heights):.1f} inches)")


if __name__ == "__main__":
    main()
