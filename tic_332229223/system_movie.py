#!/usr/bin/env python
"""
system_movie.py - a short video of the binary that joint_orbit.py describes:
the orbit in 3D, and every eclipse that TESS and ATLAS caught, on one timeline.

The companion travels its orbit while a clock runs along the timeline of all
the observations. Between eclipses the clock races ahead, orbit after orbit;
at each eclipse that was observed it slows down, the companion crosses the
star (shown as seen from Earth), and the measured brightness appears point by
point beside the brightness the orbit predicts.

Usage
-----
Run it in the folder that holds joint_orbit.py and system_view.py, on the
output folder of a joint_orbit.py run (TESS light curves tess_s*.csv, and
ATLAS light curves atlas_c.csv / atlas_o.csv from atlas_check.py, if any):

    python system_movie.py joint_332229223
    python system_movie.py joint_149762453 --period 23.2191

The period is taken from candidate_periods.csv (the most probable entry) when
joint_orbit.py wrote one, otherwise from --period, which overrides it. The
known eclipse is taken from summary.txt; --t0 overrides it. Eccentricity,
companion mass and the rest are joint_orbit.py's most probable values.

Output
------
    <folder>/system_movie.mp4   (or .gif if no video encoder can be found)

MP4 needs ffmpeg. If it is not installed, the script tries the imageio-ffmpeg
package (pip install imageio-ffmpeg), which carries its own copy; failing
that it writes a smaller GIF.

What the video is, and is not
-----------------------------
* The orbit is the single most probable one. Its eccentricity and size are
  uncertain, and the mirror-image orientation fits the eclipse equally well
  unless velocities settled it.
* In the 3D view the stars are drawn far larger than they are; the orbit is
  to scale. The view from Earth is to scale.
* The predicted brightness comes from that orbit, so it shows how well the
  most probable orbit matches the data, not a fit to each eclipse.
* An eclipse counts as observed when a light curve has points in it: three
  for TESS, two visits for ATLAS.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import Circle

NEEDED = ('joint_orbit.py', 'gaia_tools.py', 'stellar_model.py', 'system_view.py')
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import joint_orbit as jo
    import system_view as sv
except ImportError as exc:
    here = Path(__file__).resolve().parent
    missing = [f for f in NEEDED if not (here / f).exists()]
    sys.exit(f"Could not import {exc.name or 'a module'}: {exc}\n"
             + (f"Missing from {here}: {', '.join(missing)}" if missing else
                f"(all of {', '.join(NEEDED)} are in {here}; if the module named above is a package, "
                f"pip install it)"))

SURFACE, INK, INK2, MUTED, GRID = sv.SURFACE, sv.INK, sv.INK2, sv.MUTED, sv.GRID
TESS_COLOUR, ATLAS_COLOUR, MODEL_COLOUR, ACCENT = "#7fb2f0", "#f0a35e", "#e8e6df", sv.ACCENT
FPS = 24


# ----------------------------------------------------------------------------
# The orbit in 3D, and the brightness it predicts
# ----------------------------------------------------------------------------
class Orbit:
    def __init__(self, run, geo, period, t0):
        self.run, self.geo, self.period, self.t0 = run, geo, period, t0
        e, omega = geo["e"], geo["omega"]
        f_ecl = np.pi / 2 - omega                                   # true anomaly at the eclipse
        ecc_anom = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(f_ecl / 2))
        self.phase_ecl = ((ecc_anom - e * np.sin(ecc_anom)) / (2 * np.pi)) % 1.0
        self.tilt = np.radians(geo["tilt"] or 0.0)
        star = run["star"]
        # the star radius that gives the fitted crossing speed on this orbit (see sv.orbit_geometry)
        self.r1 = geo.get("r1_light", star["rad"] * sv.RSUN_AU)
        self.r2 = geo["k"] * self.r1                     # the fitted radius ratio (summary.txt)
        self.light2 = geo.get("light2", sv.companion_light(run, geo["m2"]))
        u1, u2 = jo.limb_darkening(star["teff"])
        self.z1, self.h1 = jo.blocked_light(self.r2 / self.r1, u1, u2)          # star hidden by companion
        v1, v2 = jo.limb_darkening(sv.companion_teff(run, geo))
        self.z2, self.h2 = jo.blocked_light(self.r1 / self.r2, v1, v2)          # companion hidden by star

    def position(self, t):
        """Companion relative to the star, in AU: x across the sky, y away from Earth, z up the sky."""
        phase = self.phase_ecl + (np.asarray(t, float) - self.t0) / self.period
        px, py = sv.orbit_path(self.geo["a"], self.geo["e"], self.geo["omega"], phase % 1.0)
        return px, py * np.cos(self.tilt), -py * np.sin(self.tilt)

    def brightness(self, t):
        x, y, z = self.position(t)
        sep = np.hypot(x, z)
        front = y < 0                                     # companion between the star and Earth
        dip_front = (1 - self.light2) * np.interp(sep / self.r1, self.z1, self.h1, right=0.0)
        dip_back = self.light2 * np.interp(sep / self.r2, self.z2, self.h2, right=0.0)
        return 1.0 - np.where(front, dip_front, dip_back)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_curves(folder):
    """TESS sectors (every version) and ATLAS filters, as lists of (name, kind, t, f)."""
    curves = []
    for path in sorted(folder.glob("tess_s*.csv")) + sorted(folder.glob("atlas_*.csv")):
        if path.name.startswith("atlas_check"):
            continue
        df = pd.read_csv(path)
        t, f = df["BJD"].to_numpy(float), df["FLUX"].to_numpy(float)
        good = np.isfinite(t) & np.isfinite(f) & (f > 0)
        t, f = t[good], f[good]
        if len(t) < 10:
            continue
        kind = "TESS" if path.name.startswith("tess") else "ATLAS"
        f = f / np.median(f)
        curves.append((path.stem, kind, t, f))
    return curves


def observed_eclipses(curves, orbit, half):
    """Predicted eclipses that some light curve covers, with the best data for each."""
    first = min(c[2].min() for c in curves)
    last = max(c[2].max() for c in curves)
    k0 = int(np.floor((first - orbit.t0) / orbit.period))
    k1 = int(np.ceil((last - orbit.t0) / orbit.period))
    events = []
    for k in range(k0, k1 + 1):
        mid = orbit.t0 + k * orbit.period
        best = None
        for name, kind, t, f in curves:
            inside = np.abs(t - mid) < half
            need = 3 if kind == "TESS" else 2
            if inside.sum() >= need and (best is None or inside.sum() > best[0]):
                near = np.abs(t - mid) < max(6 * half, 0.5)
                tt, ff = t[near], f[near]
                out = np.abs(tt - mid) > 1.2 * half
                if kind == "TESS" and out.sum() > 10:
                    ff = ff / np.median(ff[out])            # local level around the eclipse
                best = (int(inside.sum()), name, kind, tt, ff)
        if best:
            events.append({"k": k, "mid": mid, "name": best[1], "kind": best[2], "t": best[3], "f": best[4]})
    # the same sector in several versions: keep one event per eclipse (done above); report them
    return events


def segments(t, gap=0.5):
    breaks = np.r_[0, np.where(np.diff(t) > gap)[0] + 1, len(t)]
    return [(t[a], t[b - 1]) for a, b in zip(breaks[:-1], breaks[1:])]


def year(bjd):
    return 2000.0 + (np.asarray(bjd, float) - 2451544.5) / 365.25


# ----------------------------------------------------------------------------
# The script of the film: a time for every frame
# ----------------------------------------------------------------------------
def ease(n):
    x = np.linspace(0, 1, n)
    return x * x * (3 - 2 * x)                            # slow, fast, slow


def build_schedule(events, orbit, half, seconds_travel, seconds_eclipse):
    """Times (BJD) for every frame, and which event each frame belongs to (-1 between events)."""
    times, owner = [], []
    window = max(3 * half, 0.25)
    first = events[0]["mid"]
    # an opening orbit at an even pace, ending as the first eclipse approaches
    start = first - orbit.period - window
    n = int(4 * FPS)
    times += list(np.linspace(start, first - window, n))
    owner += [-1] * n
    for i, ev in enumerate(events):
        if i > 0:
            a, b = events[i - 1]["mid"] + window, ev["mid"] - window
            n = int(seconds_travel * FPS)
            times += list(a + (b - a) * ease(n))
            owner += [-1] * n
        n = int(seconds_eclipse * FPS)
        times += list(np.linspace(ev["mid"] - window, ev["mid"] + window, n))
        owner += [i] * n
        n = int(0.8 * FPS)                                # a pause after each eclipse
        times += [ev["mid"] + window] * n
        owner += [i] * n
    n = int(2.5 * FPS)
    times += [times[-1]] * n
    owner += [-2] * n                                    # the closing summary
    return np.array(times), np.array(owner), window


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="A video of the orbit and every observed eclipse.")
    p.add_argument("folder", help="output folder of a joint_orbit.py run")
    p.add_argument("--period", type=float, default=None, help="orbital period, days (default: candidate_periods.csv)")
    p.add_argument("--t0", type=float, default=None, help="a mid-eclipse time, BTJD or JD (default: summary.txt)")
    p.add_argument("--travel", type=float, default=2.5, help="seconds spent between eclipses (default 2.5)")
    p.add_argument("--linger", type=float, default=4.5, help="seconds spent on each eclipse (default 4.5)")
    p.add_argument("--out", default=None)
    p.add_argument("--stills", default=None, metavar="SECONDS",
                   help="instead of the video, save pictures at these moments, e.g. 3,9.5,20 (seconds into it)")
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
            print(f"Using the most probable of {len(cand)} candidate periods: {period:.5f} days "
                  f"({100 * cand.probability.iloc[0]:.0f}%).")
    if period is None:
        sys.exit("Give --period: joint_orbit.py has not narrowed the period to a list (no candidate_periods.csv).")
    run["period"][0] = period
    geo = sv.orbit_geometry(run)
    orbit = Orbit(run, geo, period, t0)
    star = run["star"]
    name = f"TIC {star.get('tic', '')}".strip()

    # length of the model eclipse, for deciding which eclipses were observed
    tt = t0 + np.linspace(-1.0, 1.0, 4001)
    dipping = orbit.brightness(tt) < 0.999
    # only the dip around t0 itself: a companion eclipse within a day must not stretch it
    mid = int(np.argmin(np.abs(tt - t0)))
    if dipping[mid]:
        lo = mid - int(np.argmin(dipping[mid::-1])) + 1 if not dipping[:mid + 1].all() else 0
        hi = mid + int(np.argmin(dipping[mid:])) - 1 if not dipping[mid:].all() else len(tt) - 1
        half = 0.5 * (tt[hi] - tt[lo])
    else:
        half = 0.08
    print(f"Orbit: period {period:.5f} d, eccentricity {geo['e']:.2f}, companion {geo['m2']:.2f} Msun; "
          f"model eclipse {2 * half * 24:.1f} hours, {100 * (1 - orbit.brightness(t0)):.1f}% deep")

    curves = read_curves(folder)
    if not curves:
        sys.exit(f"No tess_s*.csv or atlas_*.csv light curves in {folder}.")
    events = observed_eclipses(curves, orbit, half)
    if not events:
        sys.exit("None of the light curves covers an eclipse of this period.")
    print("Observed eclipses: " + ", ".join(f"{pd.Timestamp(e['mid'] - 2440587.5, unit='D').date()} ({e['kind']})"
                                            for e in events))
    times, owner, window = build_schedule(events, orbit, half, args.travel, args.linger)
    print(f"{len(times)} frames, {len(times) / FPS:.0f} seconds")

    # ---- the figure ------------------------------------------------------------------------------
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=SURFACE)
    fig.text(0.025, 0.955, f"{name}: one orbit every {period:.4f} days", color=INK, fontsize=17, va="top")
    sub = fig.text(0.025, 0.91, "", color=INK2, fontsize=10.5, va="top")

    ax3 = fig.add_axes([-0.02, 0.27, 0.58, 0.64], projection="3d")
    ax3.set_facecolor(SURFACE)
    ax3.set_axis_off()
    ring = orbit.position(t0 + period * np.linspace(0, 1, 600))
    ax3.plot(*ring, color=INK2, lw=1.0, alpha=0.7)
    span = max(np.ptp(ring[0]), np.ptp(ring[1]))
    lim = 0.62 * span
    cx, cy = 0.5 * (ring[0].max() + ring[0].min()), 0.5 * (ring[1].max() + ring[1].min())
    ax3.set_xlim(cx - lim, cx + lim)
    ax3.set_ylim(cy - lim, cy + lim)
    ax3.set_zlim(-0.5 * lim, 0.5 * lim)
    try:
        ax3.set_box_aspect((1, 1, 0.5), zoom=1.6)
    except TypeError:                                     # matplotlib older than 3.6
        ax3.set_box_aspect((1, 1, 0.5))
    ax3.plot([0, 0], [0, cy - 1.15 * lim], [0, 0], color=ACCENT, lw=1.0, ls=(0, (2, 3)))
    ax3.text(0, cy - 1.2 * lim, 0, "to Earth", color=ACCENT, fontsize=9, ha="center")
    c1 = sv.glow_colour(star["teff"])
    c2 = sv.glow_colour(sv.companion_teff(run, geo))
    ax3.scatter([0], [0], [0], s=260, color=c1, depthshade=False, zorder=5)
    comp = ax3.scatter([0], [0], [0], s=110, color=c2, edgecolors=ACCENT, linewidths=1.2, depthshade=False,
                       zorder=6)
    trail, = ax3.plot([], [], [], color=c2, lw=2.0, alpha=0.6)
    fig.text(0.025, 0.285, "Orbit to scale; stars drawn far larger than they are.", color=MUTED, fontsize=8)

    # ---- the view from Earth, to scale ----
    axe = fig.add_axes([0.56, 0.58, 0.17, 0.3])
    axe.set_facecolor(SURFACE)
    axe.set_xticks([])
    axe.set_yticks([])
    for s in axe.spines.values():
        s.set_visible(False)
    axe.set_title("Seen from Earth", color=INK, fontsize=11, loc="left")
    sv.draw_disc(axe, 0, 0, 1.0, star["teff"], zorder=3)
    k = orbit.r2 / orbit.r1
    comp_disc = Circle((5, 5), k, color=c2, zorder=4)
    comp_edge = Circle((5, 5), k, fill=False, ec=ACCENT, lw=1.2, zorder=5)
    axe.add_patch(comp_disc)
    axe.add_patch(comp_edge)
    axe.set_xlim(-2.4, 2.4)
    axe.set_ylim(-2.4, 2.4)
    axe.set_aspect("equal")
    behind_note = axe.text(0.5, 0.02, "", transform=axe.transAxes, color=MUTED, fontsize=8, ha="center")

    # ---- brightness around the current eclipse ----
    axl = fig.add_axes([0.77, 0.58, 0.205, 0.3])
    axl.set_facecolor(SURFACE)
    axl.tick_params(colors=INK2, labelsize=8)
    for s in axl.spines.values():
        s.set_color(GRID)
    axl.set_title("Brightness", color=INK, fontsize=11, loc="left")
    axl.set_xlabel("Hours from mid-eclipse", color=INK2, fontsize=8.5)
    depth = 1 - orbit.brightness(t0)
    lowest = min(min(float(np.percentile(e["f"], 2)) for e in events), 1 - 1.15 * depth)
    axl.set_xlim(-window * 24, window * 24)
    axl.set_ylim(lowest - 0.02, 1 + 0.7 * depth)
    model_line, = axl.plot([], [], color=MODEL_COLOUR, lw=1.3, label="the orbit predicts")
    data_pts, = axl.plot([], [], "o", ms=3.5, color=TESS_COLOUR, label="measured")
    cursor = axl.axvline(0, color=MUTED, lw=0.8)
    axl.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left", ncol=2, handlelength=1.2,
               columnspacing=0.8)
    lc_note = axl.text(0.98, 0.80, "", transform=axl.transAxes, color=INK2, fontsize=8.5, ha="right", va="top")

    # ---- the timeline of every observation ----
    axt = fig.add_axes([0.05, 0.08, 0.92, 0.17])
    axt.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        axt.spines[s].set_visible(False)
    axt.spines["bottom"].set_color(GRID)
    axt.tick_params(colors=INK2, labelsize=9)
    axt.set_yticks([0.75, 0.25])
    axt.set_yticklabels(["TESS", "ATLAS"], color=INK2)
    for name_, kind, t, f in curves:
        row = 0.75 if kind == "TESS" else 0.25
        if kind == "TESS":
            for a, b in segments(t):
                axt.fill_between(year([a, b]), row - 0.12, row + 0.12, color=TESS_COLOUR, alpha=0.5, lw=0)
        else:
            axt.vlines(year(t), row - 0.1, row + 0.1, color=ATLAS_COLOUR, lw=0.4, alpha=0.6)
    k_all = np.arange(int(np.floor((times.min() - t0) / period)), int(np.ceil((times.max() - t0) / period)) + 1)
    axt.vlines(year(t0 + k_all * period), 1.02, 1.1, color=MUTED, lw=0.6)
    axt.text(year(times.min()), 1.16, "every predicted eclipse", color=MUTED, fontsize=8, va="bottom")
    marks = []
    for ev in events:
        m, = axt.plot(year(ev["mid"]), 1.06, "v", ms=8, color=MUTED, zorder=4)
        marks.append(m)
    pad = 0.03 * (year(times.max()) - year(times.min()))
    axt.set_xlim(year(times.min()) - pad, year(times.max()) + pad)
    axt.set_ylim(0, 1.3)
    now_line = axt.axvline(year(times[0]), color=INK, lw=1.2)
    axt.set_xlabel("Year", color=INK2, fontsize=9)

    lit = set()

    def frame(i):
        t = times[i]
        x, y, z = orbit.position(t)
        comp._offsets3d = ([x], [y], [z])
        step = abs(times[min(i + 1, len(times) - 1)] - times[max(i - 1, 0)]) / 2
        racing = step > 0.08 * period
        comp.set_alpha(0.25 if racing else 1.0)
        back = np.linspace(t - min(0.12 * period, 40 * step + 1e-6), t, 30)
        tx, ty, tz = orbit.position(back)
        trail.set_data_3d(tx, ty, tz)
        trail.set_alpha(0.0 if racing else 0.6)
        ax3.view_init(elev=30 + 5 * np.sin(i / 160), azim=-95 + 15 * np.sin(i / 220))

        # view from Earth (in units of the star's radius)
        comp_disc.center = comp_edge.center = (x / orbit.r1, z / orbit.r1)
        comp_disc.set_zorder(4 if y < 0 else 2)
        comp_edge.set_zorder(5 if y < 0 else 2)
        behind_note.set_text("" if y < 0 or np.hypot(x, z) > 3 * orbit.r1 else "companion behind the star")

        n_orbit = int(np.floor((t - t0) / period + orbit.phase_ecl + 0.5))
        date = pd.Timestamp(t - 2440587.5, unit="D")
        sub.set_text(f"{date:%d %B %Y}   ·   orbit {n_orbit - int(np.floor((times[0] - t0) / period + 0.5)) + 1}"
                     + ("   ·   fast-forward" if racing else ""))
        now_line.set_xdata([year(t), year(t)])

        o = owner[i]
        if o >= 0:
            ev = events[o]
            lit.add(o)
            hours = (ev["t"] - ev["mid"]) * 24
            seen = ev["t"] <= t
            data_pts.set_data(hours[seen], ev["f"][seen])
            data_pts.set_color(TESS_COLOUR if ev["kind"] == "TESS" else ATLAS_COLOUR)
            grid = ev["mid"] + np.linspace(-window, t - ev["mid"], 300)
            model_line.set_data((grid - ev["mid"]) * 24, orbit.brightness(grid))
            cursor.set_xdata([(t - ev["mid"]) * 24] * 2)
            lc_note.set_text(f"{ev['kind']}, {pd.Timestamp(ev['mid'] - 2440587.5, unit='D'):%b %Y}")
        elif o == -1:
            data_pts.set_data([], [])
            model_line.set_data([], [])
            lc_note.set_text("")
        else:
            sub.set_text(f"{len(events)} eclipses seen by {' and '.join(sorted({e['kind'] for e in events}))}, "
                         f"{events[-1]['k'] - events[0]['k']} orbits apart, all on one {period:.4f}-day orbit")
        for j, m in enumerate(marks):
            m.set_color(ACCENT if j in lit else MUTED)
        return []

    if args.stills:
        for sec in (float(v) for v in args.stills.split(",")):
            i = int(np.clip(sec * FPS, 0, len(times) - 1))
            for j in range(0, i + 1, max(1, i // 40)):      # earlier frames light up the timeline marks
                frame(j)
            frame(i)
            path = folder / f"system_movie_{sec:g}s.png"
            fig.savefig(path, dpi=100, facecolor=SURFACE)
            print(f"Saved {path}")
        return

    out = Path(args.out) if args.out else folder / "system_movie.mp4"
    writer = None
    if out.suffix == ".mp4":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            try:
                import imageio_ffmpeg

                ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except ImportError:
                ffmpeg = None
        if ffmpeg:
            plt.rcParams["animation.ffmpeg_path"] = ffmpeg
            writer = animation.FFMpegWriter(fps=FPS, bitrate=4000, codec="libx264",
                                            extra_args=["-pix_fmt", "yuv420p"])
        else:
            print("No ffmpeg found (pip install imageio-ffmpeg gives one); writing a GIF instead.")
            out = out.with_suffix(".gif")
    if writer is None:
        writer = animation.PillowWriter(fps=12)
        keep = np.arange(0, len(times), 2)                # half the frames for a GIF
        times, owner = times[keep], owner[keep]
    print(f"Rendering {len(times)} frames to {out} (a few minutes) ...")
    anim = animation.FuncAnimation(fig, frame, frames=len(times), blit=False)
    anim.save(str(out), writer=writer, dpi=100, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
