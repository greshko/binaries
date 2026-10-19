#!/usr/bin/env python
"""
atlas_check.py - use ATLAS forced photometry to test eclipse periods.

ATLAS has watched the whole sky every night or two since about 2015. Each
visit gives one brightness, so it rarely catches an eclipse by chance, but
over ten years and thousands of visits a few dozen fall inside eclipses. If
a trial period is right, those visits line up at its predicted eclipse
times and are fainter; if it is wrong, the visits there are ordinary.

Usage
-----
Request the photometry from the ATLAS forced-photometry server with "Use
reduced (input) images" ticked: the default, difference images, measures
only changes from a reference image, not the star's brightness. Save the
result as a text file (the plain output, starting with ###MJD), then:

  Refine a period you already have (TIC 149762453):
    python atlas_check.py atlas_149762453.txt --folder joint_149762453 \\
        --t0 3436.443 --duration 4.3 --depth 0.079 --period 23.2191

  Test every candidate from two eclipses (TIC 332229223):
    python atlas_check.py atlas_332229223.txt --folder joint_332229223 \\
        --t0 3377.806 --duration 3.2 --depth 0.18 --gap 733.9527 --divisors 1-36

  Test a list of exact periods (TIC 284759415, the Gaia-dip candidates):
    python atlas_check.py atlas_284759415.txt --folder joint_output \\
        --t0 3441.261 --duration 2.5 --depth 0.041 --candidates joint_output/gaia_candidate_periods.csv

--t0 is a known mid-eclipse (BTJD or JD), --duration the eclipse length in
hours, --depth its depth at mid-eclipse as a fraction.

Output (in the --folder, or next to the ATLAS file)
---------------------------------------------------
    atlas_c.csv, atlas_o.csv   the cleaned light curve per ATLAS filter, in
                               barycentric JD and normalised flux, ready for
                               joint_orbit.py --extra atlas_c.csv atlas_o.csv
    atlas_check.txt            everything printed
    atlas_check.csv            one row per trial period
    atlas_fold.png             the visits folded on the best periods

How a period is judged
----------------------
For each trial period, the visits within a third of the eclipse length of a
predicted mid-eclipse are "in eclipse". Their average brightness gives a
depth with an error. The model eclipse is V-shaped near the middle, so a
visit there should show roughly 70% of the full depth or more.
  supported   the in-eclipse visits are fainter than normal by over 4 sigma
              and by at least a third of the expected depth
  excluded    at least 3 in-eclipse visits, and their depth is below half
              the expected depth at 3 sigma
  untested    fewer than 3 visits fell in an eclipse
With a single period (--period), a narrow range around it is scanned and the
period that puts the faintest visits in eclipse is reported.

Cautions
--------
* ATLAS saturates for stars brighter than roughly magnitude 11 to 12; bright
  stars give flat-topped, noisy light curves. The script reports the scatter.
* The two filters (c, cyan; o, orange) see slightly different depths if the
  stars differ in colour. They are cleaned separately and combined only in
  the period test, each relative to its own normal brightness.
* A period that is "supported" by a handful of points still needs the fold
  plot to be convincing. Look at it.
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


def read_atlas(path):
    """The ATLAS forced-photometry table, whatever the exact spacing of its header."""
    lines = Path(path).read_text(errors="replace").splitlines()
    head = next((i for i, l in enumerate(lines) if "MJD" in l and ("uJy" in l or "m " in l)), None)
    if head is None:
        sys.exit(f"{path}: no header line with MJD and uJy; is this the ATLAS forced-photometry output?")
    names = lines[head].lstrip("#").split()
    rows = [l.split() for l in lines[head + 1:] if l.strip() and not l.startswith("#")]
    rows = [r for r in rows if len(r) == len(names)]
    df = pd.DataFrame(rows, columns=names)
    for c in df.columns:
        if c not in ("F", "Obs"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def clean(df, say):
    """Keep good visits; return one table per filter with time (BJD), flux and error, normalised."""
    out = {}
    for band in sorted(df.F.dropna().unique()):
        d = df[df.F == band].copy()
        n0 = len(d)
        if np.nanmedian(d.uJy) < 5 * np.nanmedian(d.duJy):
            sys.exit("These fluxes are centred on zero: they were measured on ATLAS's difference images, which\n"
                     "show only changes from a reference image, not the star's own brightness. Request the\n"
                     "photometry again with 'Use reduced (input) images' ticked (reduced=True in the API).")
        good = np.isfinite(d.MJD) & np.isfinite(d.uJy) & np.isfinite(d.duJy) & (d.duJy > 0) & (d.uJy > 0)
        if "err" in d:
            good &= d.err == 0
        if "chi/N" in d and np.isfinite(d["chi/N"]).any():
            # bright stars on unsubtracted images fit the PSF with large chi/N as a matter of course,
            # so only visits far worse than this star's usual are dropped
            chi = d["chi/N"].to_numpy(float)
            good &= ~(chi > 5 * np.nanmedian(chi))
        if "mag5sig" in d and "m" in d:
            # cloud, twilight or moonlight: the image reached barely deeper than the star itself
            star_mag = np.nanmedian(d.m[good])
            good &= d.mag5sig > star_mag + 3
        d = d[good]
        if len(d) < 30:
            say(f"  filter {band}: only {len(d)} usable visits; skipped")
            continue
        unit = d.Obs.str[:3] if "Obs" in d else pd.Series("all", index=d.index)
        flux = d.uJy.to_numpy(float)
        err = d.duJy.to_numpy(float)
        norm = np.empty(len(d))
        for u in unit.unique():                                   # each ATLAS telescope on its own scale
            sel = (unit == u).to_numpy()
            norm[sel] = np.median(flux[sel])
        f, e = flux / norm, err / norm
        # drop wild points: far brighter than normal, or fainter than any eclipse could make them
        mad = 1.4826 * np.median(np.abs(f - np.median(f)))
        keep = (f < 1 + 5 * mad) & (f > 0.5) & (e < 5 * np.median(e))
        say(f"  filter {band}: {n0} visits, {keep.sum()} kept; scatter {100 * mad:.1f}% per visit "
            f"(typical quoted error {100 * np.median(e[keep]):.1f}%); {len(unit.unique())} telescope(s)")
        out[band] = pd.DataFrame({"MJD": d.MJD.to_numpy(float)[keep], "f": f[keep], "e": np.maximum(e[keep], mad / 2)})
    return out


def to_bjd(mjd, ra, dec):
    """ATLAS times (MJD, UTC) to barycentric JD. Which ATLAS telescope took a visit changes this by
    well under a second, so one site stands in for all."""
    import astropy.units as u
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from astropy.utils import iers
    iers.conf.auto_download = False
    t = Time(mjd, format="mjd", scale="utc", location=EarthLocation(lat=20.7 * u.deg, lon=-156.3 * u.deg))
    corr = t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), "barycentric")
    return (t.tdb + corr).jd


def judge(times, flux, err, t0, period, half_window, depth):
    """Depth and error of the visits near predicted mid-eclipses for one period."""
    phase = (times - t0) / period
    off = (phase - np.round(phase)) * period
    near = np.abs(off) <= half_window
    n = int(near.sum())
    if n == 0:
        return n, np.nan, np.nan, "untested"
    w = 1 / err[near] ** 2
    d = 1 - np.sum(w * flux[near]) / w.sum()
    s = 1 / np.sqrt(w.sum())
    s = max(s, np.std(flux[near]) / np.sqrt(n)) if n > 2 else s
    expect = 0.7 * depth
    if n >= 2 and d / s > 4 and d > expect / 3:
        verdict = "supported"
    elif n >= 3 and d + 3 * s < 0.5 * expect:
        verdict = "excluded"
    else:
        verdict = "untested" if n < 3 else "unclear"
    return n, d, s, verdict


def main():
    p = argparse.ArgumentParser(description="Test eclipse periods with ATLAS forced photometry.")
    p.add_argument("atlas", nargs="+", help="ATLAS forced-photometry text file(s)")
    p.add_argument("--folder", default=None, help="joint_orbit.py folder (for star.json, and where to write)")
    p.add_argument("--ra", type=float, default=None)
    p.add_argument("--dec", type=float, default=None)
    p.add_argument("--t0", type=float, required=True, help="a known mid-eclipse, BTJD or JD")
    p.add_argument("--duration", type=float, required=True, help="eclipse length, hours")
    p.add_argument("--depth", type=float, required=True, help="depth at mid-eclipse, as a fraction (0.08 = 8%)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--period", type=float, help="refine this period")
    g.add_argument("--gap", type=float, help="time between two known eclipses; test it divided by --divisors")
    g.add_argument("--candidates", help="CSV with a period_days column (e.g. gaia_candidate_periods.csv)")
    p.add_argument("--divisors", default="1-40", help="with --gap: whole numbers to try, e.g. 1-36 (default 1-40)")
    p.add_argument("--scan", type=float, default=0.02, help="with --period: search this many days either side")
    args = p.parse_args()

    folder = Path(args.folder) if args.folder else Path(args.atlas[0]).resolve().parent
    if args.ra is None or args.dec is None:
        star_file = folder / "star.json"
        if not star_file.exists():
            sys.exit("Need the star's position: give --folder with star.json, or --ra and --dec.")
        star = json.loads(star_file.read_text(encoding="utf-8-sig"))
        args.ra, args.dec = star["ra"], star["dec"]
    t0 = args.t0 + 2457000.0 if args.t0 < 100000 else args.t0
    dur = args.duration / 24.0
    half = dur / 3.0
    log = []

    def say(text=""):
        print(text)
        log.append(text)

    say("ATLAS DATA")
    tables = [read_atlas(f) for f in args.atlas]
    bands = clean(pd.concat(tables, ignore_index=True), say)
    if not bands:
        sys.exit("No usable ATLAS visits.")
    pieces = []
    for band, d in bands.items():
        bjd = to_bjd(d.MJD.to_numpy(), args.ra, args.dec)
        pd.DataFrame({"BJD": bjd, "FLUX": d.f, "FLUX_ERR": d.e}).to_csv(folder / f"atlas_{band}.csv", index=False)
        pieces.append(pd.DataFrame({"t": bjd, "f": d.f.to_numpy(), "e": d.e.to_numpy(), "band": band}))
        say(f"  wrote {folder / f'atlas_{band}.csv'} (BJD {bjd.min():.0f} to {bjd.max():.0f})")
    lc = pd.concat(pieces, ignore_index=True)
    t, f, e = lc.t.to_numpy(), lc.f.to_numpy(), lc.e.to_numpy()

    # ---- trial periods ------------------------------------------------------------------------------
    if args.period:
        baseline = np.abs(t - t0).max()
        step = 0.1 * dur / max(baseline / args.period, 1)          # slip of a tenth of the eclipse at the far end
        trials = np.arange(args.period - args.scan, args.period + args.scan + step, step)
        labels = [f"{v:.5f}" for v in trials]
    elif args.gap:
        lo, hi = (int(x) for x in re.split(r"[-:]", args.divisors)) if re.search(r"[-:]", args.divisors) \
            else (int(args.divisors), int(args.divisors))
        trials = np.array([args.gap / n for n in range(lo, hi + 1)])
        labels = [f"gap/{n}" for n in range(lo, hi + 1)]
    else:
        table = pd.read_csv(args.candidates)
        trials = table.period_days.to_numpy(float)
        labels = [f"{v:.4f}" for v in trials]

    rows = []
    for label, period in zip(labels, trials):
        n, d, s, verdict = judge(t, f, e, t0, period, half, args.depth)
        rows.append({"trial": label, "period_days": period, "visits_in_eclipse": n, "depth": d, "error": s,
                     "sigma": d / s if s and np.isfinite(s) else np.nan, "verdict": verdict})
    res = pd.DataFrame(rows)
    res.to_csv(folder / "atlas_check.csv", index=False)

    say(f"\nTRIAL PERIODS (eclipse visits = within {half * 24 * 60:.0f} minutes of a predicted mid-eclipse; "
        f"expected depth there about {100 * 0.7 * args.depth:.1f}%)")
    if args.period:
        valid = res[res.visits_in_eclipse >= 2]
        best = valid.loc[valid.sigma.idxmax()] if len(valid) else None
        at = res.iloc[int(np.argmin(np.abs(trials - args.period)))]
        say(f"  At {args.period}: {at.visits_in_eclipse} visits in eclipse, depth "
            f"{100 * at.depth:.1f} +/- {100 * at.error:.1f}% -> {at.verdict}")
        if best is not None:
            strong = valid[valid.sigma > best.sigma - 1]
            say(f"  Best in {args.period - args.scan:.4f} to {args.period + args.scan:.4f}: {best.period_days:.5f} days, "
                f"{best.visits_in_eclipse} visits, depth {100 * best.depth:.1f} +/- {100 * best.error:.1f}% "
                f"({best.sigma:.1f} sigma)")
            say(f"  Periods within 1 sigma of the best: {strong.period_days.min():.5f} to {strong.period_days.max():.5f}")
            show = [best.period_days]
        else:
            show = [args.period]
    else:
        for r in res.itertuples():
            depth_txt = "" if not np.isfinite(r.depth) else f"depth {100 * r.depth:5.1f} +/- {100 * r.error:4.1f}%"
            say(f"  {r.trial:>10}  {r.period_days:9.4f} d   {r.visits_in_eclipse:3d} visits   {depth_txt:<24} {r.verdict}")
        counts = res.verdict.value_counts().to_dict()
        say("  " + ", ".join(f"{v} {k}" for k, v in counts.items()))
        # fold plots for the three best-supported trials (or the first trials, if none has visits)
        ranked = res[res.visits_in_eclipse >= 1].sort_values("sigma", ascending=False, na_position="last")
        show = list((ranked if len(ranked) else res).period_days.head(3))
        if args.gap:
            # If the period is P, then 2P, 3P ... also put eclipses on real eclipses (every 2nd, 3rd ...), so
            # they come out supported too. The shorter one wins only if its extra eclipses, the ones the
            # longer period skips, are faint as well.
            numbers = [int(l.split("/")[1]) for l in labels]
            good = [n for n, v in zip(numbers, res.verdict) if v == "supported"]
            for b in good:
                divisors = [a for a in good if a < b and b % a == 0]
                if not divisors:
                    continue
                a = max(divisors)                                  # the closest longer period that also fits
                period_b = args.gap / b
                cycle = np.round((t - t0) / period_b)
                extra = (cycle % (b // a)) != 0
                n_x, d_x, s_x, v_x = judge(t[extra], f[extra], e[extra], t0, period_b, half, args.depth)
                if v_x == "supported":
                    say(f"  gap/{b}: the eclipses it predicts between gap/{a}'s are faint too ({n_x} visits, "
                        f"{100 * d_x:.1f}% deep), so gap/{a} and its longer relatives are out.")
                elif v_x == "excluded":
                    say(f"  gap/{b}: fits only through the eclipses it shares with gap/{a}; the others are not there "
                        f"({n_x} visits), so gap/{b} is out.")
            if len(good) > 1:
                say("  (Periods that are whole multiples of the true one also come out 'supported', because every")
                say("  second or third true eclipse lands on their predictions. The lines above sort that out where")
                say("  the data allow; joint_orbit.py does the same with the ATLAS light curves given as --extra.)")

    # ---- fold plots ---------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(show), figsize=(4.2 * len(show), 3.6), squeeze=False, sharey=True)
    for ax, period in zip(axes[0], show):
        phase = (t - t0) / period
        hours = (phase - np.round(phase)) * period * 24
        for band, colour in (("c", "#2a78d6"), ("o", "#eb6834")):
            sel = (lc.band == band).to_numpy()
            ax.plot(hours[sel], f[sel], ".", ms=3, color=colour, alpha=0.5, label=f"ATLAS {band}")
        ax.axvspan(-half * 24, half * 24, color="0.9", zorder=0)
        ax.axhline(1 - args.depth, color="0.6", lw=0.8, ls="--")
        ax.set_xlim(-4 * args.duration, 4 * args.duration)
        ax.set_title(f"{period:.5f} days", fontsize=10)
        ax.set_xlabel("Hours from predicted mid-eclipse")
    axes[0, 0].set_ylabel("Brightness (normal = 1)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    lowest = max(0.5, 1 - 2.5 * args.depth)
    axes[0, 0].set_ylim(lowest, 1 + 0.5 * args.depth)
    fig.tight_layout()
    fig.savefig(folder / "atlas_fold.png", dpi=110)
    plt.close(fig)
    say(f"\nSaved atlas_check.csv, atlas_fold.png and the light curves in {folder}/")
    say("To use the visits in the full analysis as well:  joint_orbit.py ... --extra "
        + " ".join(f"{folder}/atlas_{b}.csv" for b in bands))
    (folder / "atlas_check.txt").write_text("\n".join(log) + "\n")


if __name__ == "__main__":
    main()
