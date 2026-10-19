#!/usr/bin/env python
"""
gaia_neighbours.py - could the eclipse be on a nearby star instead of the target?

An eclipse seen in a survey is on the target only if no other star in the
measurement could have produced it. This script asks Gaia DR3 about every
source around the target and, for each one, whether it is bright enough and
close enough to produce the observed eclipse depth:

  * in TESS, whose pixels are 21 arcsec, so the light of every star within
    an aperture of a few pixels is added together;
  * in ATLAS, whose forced photometry is fitted at the target's position with
    a point-spread function a few arcsec wide, so a star a few arcsec away
    contributes much less.

A neighbour can host the eclipse only if it supplies enough of the measured
light in BOTH surveys: an 18%-deep dip needs a star giving at least 18% of the
light, and more if its own eclipse is partial. A neighbour that could do it in
TESS but not in ATLAS is ruled out by ATLAS seeing the same depth.

It also reports what Gaia says about the target itself:
  * signs of a companion too close for Gaia to separate (RUWE, the fraction
    of scans that saw two peaks, the image-shape harmonic, and the BP/RP flux
    excess that blending raises);
  * whether its velocity varied between Gaia's spectrograph visits (RVS):
    a star eclipsed by a companion on this orbit must swing by tens of km/s;
  * whether Gaia catalogued it as an eclipsing binary or fitted an orbit.

Usage
-----
    python gaia_neighbours.py joint_output_332229223            # reads star.json and summary.txt
    python gaia_neighbours.py --ra 182.4157979 --dec 45.6821472 --depth 0.18
    python gaia_neighbours.py FOLDER --radius 90 --tess-aperture 42 --atlas-fwhm 4.5

Needs the internet (the ESA Gaia archive). No extra packages. If the archive
cannot be reached, download the same query from https://gea.esac.esa.int/archive/
as CSV and pass it with --csv (the query is printed).

Output (in the folder, or the current directory): gaia_neighbours.csv,
gaia_neighbours.png, gaia_neighbours.txt.

Caveats
-------
* Gaia separates two stars of similar brightness down to about 0.4-0.7 arcsec,
  and a much fainter one only beyond about 1-2 arcsec. Anything closer is not
  in the list; the target's RUWE and image-shape flags are the check for that.
* The TESS aperture is approximated by a circle (--tess-aperture, default 42
  arcsec, two pixels); the real one is a set of pixels the pipeline chose.
* TESS magnitudes come from Gaia colours (Stassun et al. 2019); ATLAS fluxes
  are taken in proportion to Gaia G.
"""

import argparse
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"

COLUMNS = ("source_id, ra, dec, parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, phot_bp_mean_mag, "
           "phot_rp_mean_mag, bp_rp, ruwe, ipd_frac_multi_peak, ipd_gof_harmonic_amplitude, "
           "phot_bp_rp_excess_factor, radial_velocity, radial_velocity_error, rv_nb_transits, "
           "rv_amplitude_robust, rv_chisq_pvalue, rv_renormalised_gof, non_single_star, phot_variable_flag")


def tap(query, timeout=120):
    data = urllib.parse.urlencode({"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                                   "QUERY": query}).encode()
    with urllib.request.urlopen(urllib.request.Request(TAP, data=data), timeout=timeout) as r:
        return pd.read_csv(io.StringIO(r.read().decode()))


def cone_query(ra, dec, radius):
    return (f"SELECT {COLUMNS}, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) * 3600 AS sep "
            f"FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {ra}, {dec}, {radius / 3600})) ORDER BY sep")


def tess_mag(g, bp_rp):
    """TESS magnitude from Gaia (Stassun et al. 2019, TIC v8)."""
    x = np.asarray(bp_rp, float)
    t = g - 0.00522555 * x**3 + 0.0891337 * x**2 - 0.633923 * x + 0.0324473
    return np.where(np.isfinite(x), t, g - 0.430)


def excess_significance(c, bp_rp, g):
    """Corrected BP/RP flux excess C* in units of its expected scatter (Riello et al. 2021).
    Large positive values mean extra light in BP and RP from a neighbour, i.e. blending."""
    x = np.asarray(bp_rp, float)
    f = np.where(x < 0.5, 1.154360 + 0.033772 * x + 0.032277 * x**2,
                 np.where(x < 4.0, 1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
                          1.057572 + 0.140537 * x))
    sigma = 0.0059898 + 8.817481e-12 * np.asarray(g, float) ** 7.618399
    return (np.asarray(c, float) - f) / sigma


def offsets(ra, dec, ra0, dec0):
    """Arcsec east and north of the target."""
    east = (np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600
    north = (np.asarray(dec) - dec0) * 3600
    return east, north


def main():
    p = argparse.ArgumentParser(description="Could a Gaia neighbour host the eclipse?")
    p.add_argument("folder", nargs="?", default=None, help="joint_orbit.py output folder (star.json, summary.txt)")
    p.add_argument("--ra", type=float, default=None)
    p.add_argument("--dec", type=float, default=None)
    p.add_argument("--depth", type=float, default=None, help="eclipse depth as a fraction (default: summary.txt)")
    p.add_argument("--atlas-depth", type=float, default=None, help="ATLAS depth, if different (default: same)")
    p.add_argument("--radius", type=float, default=90.0, help="search radius, arcsec (default 90)")
    p.add_argument("--tess-aperture", type=float, default=42.0, help="TESS aperture radius, arcsec (default 42)")
    p.add_argument("--atlas-fwhm", type=float, default=4.5, help="ATLAS image width, arcsec (default 4.5)")
    p.add_argument("--period", type=float, default=None, help="orbital period, days (for the velocity check)")
    p.add_argument("--e", type=float, default=None, help="eccentricity (for the velocity check)")
    p.add_argument("--m2", type=float, default=None, help="companion mass, Msun (for the velocity check)")
    p.add_argument("--csv", default=None, help="use this Gaia table instead of querying the archive")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()

    folder = Path(args.folder) if args.folder else None
    star = {}
    text = ""
    if folder:
        star = json.loads((folder / "star.json").read_text(encoding="utf-8-sig"))
        text = (folder / "summary.txt").read_text() if (folder / "summary.txt").exists() else ""
    ra0 = args.ra if args.ra is not None else star.get("ra")
    dec0 = args.dec if args.dec is not None else star.get("dec")
    if ra0 is None or dec0 is None:
        sys.exit("Give a folder with star.json, or --ra and --dec.")
    depth = args.depth
    if depth is None:
        found = re.findall(r"deepest about ([\d.]+)%", text)
        depth = float(found[-1]) / 100 if found else None
    if depth is None:
        sys.exit("Give --depth (the eclipse depth as a fraction, e.g. 0.18).")
    atlas_depth = args.atlas_depth if args.atlas_depth is not None else depth
    period = args.period
    if period is None and folder and (folder / "candidate_periods.csv").exists():
        period = float(pd.read_csv(folder / "candidate_periods.csv").sort_values("probability").period_days.iloc[-1])
    ecc = args.e
    if ecc is None:
        found = re.findall(r"Eccentricity ([\d.]+) \(68%", text)
        ecc = float(found[-1]) if found else None
    m2 = args.m2
    if m2 is None:
        found = re.findall(r"^\s+mass\s+([\d.]+) Msun", text, re.M)
        m2 = float(found[-1]) if found else None
    outdir = Path(args.outdir) if args.outdir else (folder if folder else Path("."))
    outdir.mkdir(parents=True, exist_ok=True)

    lines = []

    def say(t=""):
        print(t)
        lines.append(t)

    # ---- the Gaia sources ----------------------------------------------------------------------
    query = cone_query(ra0, dec0, args.radius)
    if args.csv:
        gaia = pd.read_csv(args.csv)
        gaia.columns = [c.lower() for c in gaia.columns]
        if "sep" not in gaia.columns:
            e, n = offsets(gaia.ra, gaia.dec, ra0, dec0)
            gaia["sep"] = np.hypot(e, n)
    else:
        try:
            gaia = tap(query)
        except Exception as exc:
            print(f"The Gaia archive did not answer ({str(exc)[:120]}).")
            print("Run this query at https://gea.esac.esa.int/archive/ (Search > Advanced (ADQL)), download the")
            print("result as CSV and rerun with --csv that_file.csv:\n")
            print(query)
            sys.exit(1)
    gaia = gaia.sort_values("sep").reset_index(drop=True)
    if len(gaia) == 0:
        sys.exit("Gaia returned no sources there; check the position.")
    # the target: the brightest source within 2 arcsec (the nearest if none)
    close = gaia[gaia.sep < 2.0]
    ti = int(close.phot_g_mean_mag.idxmin()) if len(close) else 0
    target = gaia.loc[ti]
    gaia["east"], gaia["north"] = offsets(gaia.ra, gaia.dec, target.ra, target.dec)
    gaia["sep"] = np.hypot(gaia.east, gaia.north)
    gaia["tmag"] = tess_mag(gaia.phot_g_mean_mag, gaia.bp_rp)
    gaia["dG"] = gaia.phot_g_mean_mag - target.phot_g_mean_mag
    flux_t = 10 ** (-0.4 * (gaia.tmag - gaia.tmag[ti]))           # TESS-band flux, target = 1
    flux_g = 10 ** (-0.4 * gaia.dG)                               # G-band flux, as a stand-in for ATLAS

    # share of the measured light each source gives, in each survey
    in_ap = gaia.sep <= args.tess_aperture
    total_tess = flux_t[in_ap].sum()
    sigma = args.atlas_fwhm / 2.3548
    weight = np.exp(-gaia.sep**2 / (4 * sigma**2))                 # overlap of two Gaussian images
    total_atlas = (flux_g * weight).sum()
    gaia["tess_share"] = np.where(in_ap, flux_t / total_tess, 0.0)
    gaia["atlas_share"] = flux_g * weight / total_atlas
    gaia["needed_depth_tess"] = depth / np.maximum(gaia.tess_share, 1e-12)
    gaia["needed_depth_atlas"] = atlas_depth / np.maximum(gaia.atlas_share, 1e-12)
    gaia["could_host"] = (gaia.needed_depth_tess <= 1.0) & (gaia.needed_depth_atlas <= 1.0)
    gaia.loc[ti, "could_host"] = True
    gaia.to_csv(outdir / "gaia_neighbours.csv", index=False)

    say(f"GAIA DR3 AROUND RA {ra0:.6f}, Dec {dec0:+.6f} (radius {args.radius:.0f} arcsec)")
    say(f"  target: Gaia DR3 {int(target.source_id)}, G = {target.phot_g_mean_mag:.2f}, "
        f"BP-RP = {target.bp_rp:.2f}, {target.sep if ti else 0:.2f} arcsec from the position given")
    if np.isfinite(target.parallax) and target.parallax > 0:
        say(f"  parallax {target.parallax:.3f} +/- {target.parallax_error:.3f} mas (about {1000 / target.parallax:.0f} pc)")
    say(f"  eclipse depth {100 * depth:.1f}% in TESS" + (f", {100 * atlas_depth:.1f}% in ATLAS" if atlas_depth != depth else
                                                        " (ATLAS sees the same)"))
    say(f"  {len(gaia) - 1} other Gaia sources within {args.radius:.0f} arcsec; "
        f"{int(in_ap.sum()) - 1} inside a {args.tess_aperture:.0f}-arcsec TESS aperture, "
        f"which together add {100 * (1 - gaia.tess_share[ti]):.1f}% to the light TESS measures")

    say("\nNEIGHBOURS (closest first; the depth each would need to produce the eclipse by itself)")
    say("   sep(\")   G      dG    BP-RP  parallax   TESS share  needs   ATLAS share  needs    verdict")
    for i, r in gaia.iterrows():
        if i == ti or (r.sep > args.tess_aperture and i > 15):
            continue
        need_t = "  -  " if r.tess_share == 0 else ("  >1 " if r.needed_depth_tess > 1 else f"{100 * r.needed_depth_tess:4.0f}%")
        need_a = "  >1 " if r.needed_depth_atlas > 1 else f"{100 * r.needed_depth_atlas:4.0f}%"
        verdict = "COULD HOST IT" if r.could_host else ("too faint" if r.needed_depth_tess > 1 else
                                                         "ruled out by ATLAS")
        plx = f"{r.parallax:6.2f}" if np.isfinite(r.parallax) else "   -  "
        bprp = f"{r.bp_rp:5.2f}" if np.isfinite(r.bp_rp) else "  -  "
        say(f"  {r.sep:6.1f}  {r.phot_g_mean_mag:5.2f}  {r.dG:+5.2f}  {bprp}  {plx}     {100 * r.tess_share:6.2f}%   "
            f"{need_t}   {100 * r.atlas_share:7.3f}%   {need_a}   {verdict}")
    hosts = gaia[gaia.could_host & (gaia.index != ti)]
    say("")
    if len(hosts):
        say(f"  {len(hosts)} neighbour(s) could produce the eclipse in both surveys: worth a closer look")
        say("  (a centroid test on the TESS pixels, or deeper images, can tell).")
    else:
        brightest = gaia[(gaia.index != ti) & in_ap].sort_values("tess_share", ascending=False)
        say("  No resolved Gaia neighbour can produce the eclipse: none is bright enough in TESS and close")
        say("  enough to the target to show the same depth in ATLAS.")
        if len(brightest):
            b = brightest.iloc[0]
            need = "an eclipse deeper than total, impossible" if b.needed_depth_tess > 1 else \
                f"a {100 * b.needed_depth_tess:.0f}%-deep eclipse of its own"
            say(f"  (The brightest in the TESS aperture gives {100 * b.tess_share:.2f}% of the light, so it would need {need}.)")
    need_star = depth
    say(f"  Anything hosting the eclipse must give at least {100 * need_star:.0f}% of the light, so it is at most "
        f"{-2.5 * np.log10(need_star / (1 - need_star)):.1f} magnitudes fainter than the target.")

    # ---- the target itself ---------------------------------------------------------------------
    say("\nTHE TARGET ITSELF: a companion too close for Gaia to separate?")
    ruwe = target.get("ruwe", np.nan)
    say(f"  RUWE {ruwe:.2f}" + ("  (above 1.4: the motion does not fit one star; a close companion is likely)"
                                if ruwe > 1.4 else "  (normal: no sign of a companion between about 0.1 and 1 arcsec,"
                                " though a faint one could hide)"))
    multi = target.get("ipd_frac_multi_peak", np.nan)
    if np.isfinite(multi):
        say(f"  scans that saw two peaks: {multi:.0f}%" + ("  (a resolved or nearly resolved double)" if multi > 2 else
                                                           "  (none to speak of)"))
    harm = target.get("ipd_gof_harmonic_amplitude", np.nan)
    if np.isfinite(harm):
        say(f"  image-shape harmonic amplitude {harm:.3f}" + ("  (elongated images: a close double)" if harm > 0.1 and ruwe > 1.4
                                                             else "  (ordinary)"))
    if np.isfinite(target.get("phot_bp_rp_excess_factor", np.nan)):
        z = float(excess_significance(target.phot_bp_rp_excess_factor, target.bp_rp, target.phot_g_mean_mag))
        say(f"  BP/RP flux excess {z:+.1f} sigma from a single star of this colour"
            + ("  (extra light in the colour photometry: a blend)" if z > 3 else "  (normal)"))
    nss = target.get("non_single_star", 0)
    say(f"  Gaia non-single-star flag: {int(nss) if np.isfinite(nss) else 0}"
        + ("  (Gaia fitted an orbit or acceleration; see below)" if nss and np.isfinite(nss) and nss > 0 else ""))

    say("\nTHE TARGET'S VELOCITY IN GAIA (RVS)")
    rv = target.get("radial_velocity", np.nan)
    if not np.isfinite(rv):
        say("  Gaia has no velocity for this star (too faint for RVS, or rejected).")
    else:
        nt = target.get("rv_nb_transits", np.nan)
        amp = target.get("rv_amplitude_robust", np.nan)
        pval = target.get("rv_chisq_pvalue", np.nan)
        gof = target.get("rv_renormalised_gof", np.nan)
        say(f"  median {rv:+.1f} km/s from {nt:.0f} visits; error {target.radial_velocity_error:.1f} km/s")
        if np.isfinite(amp):
            say(f"  spread of the visits (robust peak to peak): {amp:.1f} km/s")
        else:
            say("  Gaia gives no visit-to-visit spread or constancy test for this star: it publishes those only for")
            say("  stars brighter than G_RVS = 12. Fainter ones get one velocity from all visits' spectra added")
            say("  together, which hides any swing, so Gaia cannot answer this question here.")
        if np.isfinite(pval) and np.isfinite(gof):
            variable = (pval <= 0.01) and (gof > 4) and (nt >= 10)
            say(f"  constancy test: p = {pval:.2g}, renormalised goodness of fit {gof:.1f} -> "
                + ("VARIABLE (Katz et al. 2023 criteria)" if variable else "consistent with constant"))
        if period and ecc is not None and m2 and np.isfinite(amp):
            main = re.search(r"MAIN STAR: mass ([\d.]+) Msun", text)
            m1 = float(main.group(1)) if main else star.get("mass", 1.0)
            a_au = ((m1 + m2) * (period / 365.25) ** 2) ** (1 / 3)
            k1 = 2 * np.pi * a_au * m2 / (m1 + m2) / period / np.sqrt(1 - ecc**2) * 1731.4568
            say(f"  If the eclipse is on this star (P {period:.2f} d, e {ecc:.2f}, companion {m2:.2f} Msun), its velocity")
            say(f"  swings over {2 * k1:.0f} km/s peak to peak. Gaia's visits rarely catch both extremes, and the")
            say("  companion's lines blended with the star's shrink what a single-star fit measures, but a few")
            say("  dozen visits should still show tens of km/s. A spread of a few km/s with many visits points")
            say("  to the eclipse being on another star.")

    # ---- Gaia's own variability and orbit tables -----------------------------------------------
    if not args.csv:
        sid = int(target.source_id)
        for table, label in (("gaiadr3.vari_eclipsing_binary", "eclipsing-binary catalogue"),
                             ("gaiadr3.nss_two_body_orbit", "orbit solutions"),
                             ("gaiadr3.nss_acceleration_astro", "astrometric accelerations")):
            try:
                found = tap(f"SELECT * FROM {table} WHERE source_id = {sid}", timeout=60)
            except Exception:
                say(f"  ({label}: could not be queried)")
                continue
            if len(found) == 0:
                say(f"  Gaia {label}: not listed")
                continue
            r = found.iloc[0]
            if "frequency" in r:
                say(f"  Gaia {label}: LISTED, period {1 / r['frequency']:.4f} d" if r["frequency"] else f"  Gaia {label}: LISTED")
            elif "period" in r and np.isfinite(r.get("period", np.nan)):
                say(f"  Gaia {label}: LISTED ({r.get('nss_solution_type', '')}), period {r['period']:.3f} d, "
                    f"eccentricity {r.get('eccentricity', np.nan):.2f}")
            else:
                say(f"  Gaia {label}: LISTED ({r.get('nss_solution_type', '')})")

    make_figure(gaia, ti, args, depth, outdir, star)
    (outdir / "gaia_neighbours.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSaved gaia_neighbours.csv, gaia_neighbours.png and gaia_neighbours.txt in {outdir}/")


def make_figure(gaia, ti, args, depth, outdir, star):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 1]})
    target = gaia.loc[ti]
    for ax, half, title in ((axes[0], args.tess_aperture * 1.5, "Around the TESS aperture"),
                            (axes[1], max(4 * args.atlas_fwhm, 12), "Close in: what ATLAS sees")):
        ax.set_facecolor(SURFACE)
        for side in ax.spines.values():
            side.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8, length=2)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        near = gaia[(gaia.east.abs() < half) & (gaia.north.abs() < half)]
        size = 6 + 260 * (10 ** (-0.4 * np.clip(near.dG, -1, 12))) ** 0.5
        cannot = near[~near.could_host]
        can = near[near.could_host & (near.index != ti)]
        cannot = cannot[cannot.index != ti]
        if len(cannot):
            ax.scatter(cannot.east, cannot.north, s=size[cannot.index].to_numpy(), color=MUTED, alpha=0.6, lw=0,
                       label="cannot produce the eclipse")
        if len(can):
            ax.scatter(can.east, can.north, s=size[can.index].to_numpy(), color=ORANGE, lw=0, label="could produce it")
        ax.scatter([0], [0], s=[float(size[ti]) if ti in size.index else 260.0], color=BLUE, lw=0, label="target",
                   zorder=5)
        for r in near[(near.index != ti) & (near.dG < 6)].itertuples():
            ax.annotate(f"ΔG {r.dG:+.1f}", (r.east, r.north), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=INK2)
        if ax is axes[0]:
            ax.add_patch(Circle((0, 0), args.tess_aperture, fill=False, ec=BLUE, lw=1, ls=(0, (4, 3))))
            ax.text(0, args.tess_aperture, "TESS aperture (assumed)", color=BLUE, fontsize=7, ha="center", va="bottom")
            ax.add_patch(plt.Rectangle((-10.5, -10.5), 21, 21, fill=False, ec=GRID, lw=0.8))
            ax.text(10.5, -10.5, " one TESS pixel", color=MUTED, fontsize=7, va="top")
        else:
            ax.add_patch(Circle((0, 0), args.atlas_fwhm, fill=False, ec=ORANGE, lw=1, ls=(0, (4, 3))))
            ax.text(0, args.atlas_fwhm, "ATLAS image width", color=ORANGE, fontsize=7, ha="center", va="bottom")
            ax.add_patch(Circle((0, 0), 0.7, fill=False, ec=MUTED, lw=0.8, ls=(0, (1, 2))))
            ax.annotate("inner ring: Gaia cannot\nseparate stars this close", (0, -0.7), xytext=(0, -2.2 * args.atlas_fwhm / 2),
                        textcoords="data", color=MUTED, fontsize=7, ha="center", va="top",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.set_xlim(half, -half)                                   # east to the left, as on the sky
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel("arcsec east", color=INK2, fontsize=9)
        ax.set_ylabel("arcsec north", color=INK2, fontsize=9)
    leg = axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK2)
    name = f"TIC {star.get('tic')}" if star.get("tic") else f"Gaia DR3 {int(target.source_id)}"
    fig.suptitle(f"{name}: Gaia neighbours that could, or could not, produce a {100 * depth:.0f}% eclipse",
                 color=INK, fontsize=12, x=0.06, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outdir / "gaia_neighbours.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
