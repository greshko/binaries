#!/usr/bin/env python
"""
make_rnaas_table.py - the Research Note's table, filled in from joint_orbit.py output folders.

    python make_rnaas_table.py joint_332229223 joint_149762453
    python make_rnaas_table.py joint_332229223 joint_149762453 --out rnaas_table --pdf

Writes OUT.tex: an AASTeX (RNAAS style) document holding one deluxetable with
a column per star. With --pdf it also runs pdflatex (needs aastex631.cls, or
aastex701.cls with --class aastex701, beside it or installed).

Where each value comes from, per folder:
  summary.txt           everything joint_orbit.py measured or fitted: the catalog
                        and Gaia DR3 values (its CATALOGUE VALUES block: TIC
                        position, TESS magnitude and temperature, source ID,
                        parallax, RUWE, excess noise, G, BP-RP, the RVS velocity
                        and how Gaia measured it), epoch and its error, depth,
                        eccentricity, both stars' masses, radii and temperatures,
                        the light share, the flux ratio, whether a companion
                        (secondary) eclipse was seen
  rv_epochs.csv         LAMOST velocities (two rows marked A and B = double-lined)
  candidate_periods.csv the period and its 1-sigma error
  light curves          (through rnaas_figure.py) which TESS sectors and how many
                        ATLAS nights caught an eclipse, and the eclipse's length
Summaries from older joint_orbit.py versions have no CATALOGUE VALUES block; then
star.json, gaia.json and gaia_diagnostics.json are read instead, with a warning.

Anything that cannot be found is printed in red as "?" and listed on screen,
so nothing is silently made up. Values from the model are medians with 68%
ranges; where a range runs into the edge of what the model allows a footnote
says so. The model footnote describes how joint_orbit.py split the light: one
MIST isochrone fitted to the system's photometry (stellar_split.json), or the
older mass-luminosity relation, as its summary.txt records.

Run joint_orbit.py (current version) on each folder first: older versions
rounded ranges to grid points and did not print the epoch's error.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

MISSING = r"\tbd{?}"


def number(text, pattern, group=1, cast=float):
    found = re.findall(pattern, text, re.M)
    if not found:
        return None
    value = found[-1]
    if isinstance(value, tuple):
        value = value[group - 1]
    return cast(value)


def decimals_for(err, sig=2):
    """Decimal places that show an error to `sig` significant figures."""
    if err is None or not np.isfinite(err) or err <= 0:
        return 3
    return max(0, sig - 1 - int(np.floor(np.log10(err))))


def pm(value, err, sig=2):
    if value is None:
        return MISSING
    if err is None or not np.isfinite(err) or err <= 0:
        return f"${value}$"
    d = decimals_for(err, sig)
    return f"${value:.{d}f} \\pm {err:.{d}f}$"


def rng(med, lo, hi, d=2):
    if med is None:
        return MISSING
    return f"{med:.{d}f} ({lo:.{d}f}--{hi:.{d}f})"


def gather(folder, warn):
    folder = Path(folder)
    out = {"folder": folder.name}
    text = (folder / "summary.txt").read_text() if (folder / "summary.txt").exists() else ""
    if not text:
        warn(f"{folder}: no summary.txt; run joint_orbit.py first")

    # Catalogue values: from summary.txt's CATALOGUE VALUES block (current joint_orbit.py). Older
    # summaries lack it, and then star.json, gaia.json and gaia_diagnostics.json stand in.
    from system_view import read_catalogue
    cat = read_catalogue(text)
    if not cat:
        warn(f"{folder}: summary.txt has no CATALOGUE VALUES block (older joint_orbit.py); reading star.json "
             "and gaia.json instead - rerun the current joint_orbit.py to need only summary.txt")
        star = json.loads((folder / "star.json").read_text(encoding="utf-8-sig")) \
            if (folder / "star.json").exists() else {}
        cat = dict(star, tic=star.get("tic", folder.name))
        if (folder / "gaia.json").exists():
            cat.update(json.loads((folder / "gaia.json").read_text()))
        diag = folder / "gaia_diagnostics.json"
        if diag.exists():
            for k, v in ((json.loads(diag.read_text()).get("target") or {})).items():
                if cat.get(k) is None:
                    cat[k] = v

    def g(key):
        v = cat.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return None
        return v

    out["tic"] = g("tic") or folder.name
    out["ra"], out["dec"] = g("ra"), g("dec")
    out["tmag"], out["teff"] = g("tmag"), g("teff")
    sid = g("source_id")
    out["source_id"] = str(int(sid)) if sid is not None and int(sid) > 10**10 else None
    if out["source_id"] is None:
        warn(f"{folder}: no usable Gaia source ID ({sid}); rerun joint_orbit.py online")
    out["plx"], out["plx_err"] = g("parallax"), g("parallax_error")
    out["ruwe"], out["aen"] = g("ruwe"), g("astrometric_excess_noise")
    out["G"], out["bprp"] = g("phot_g_mean_mag"), g("bp_rp")
    if out["bprp"] is None:
        warn(f"{folder}: no BP-RP color (rerun joint_orbit.py online)")
    out["rv"], out["rv_err"] = g("radial_velocity"), g("radial_velocity_error")
    out["rv_n"], out["rv_amp"] = g("rv_nb_transits"), g("rv_amplitude_robust")
    out["rv_method"] = g("rv_method_used")

    # LAMOST
    lam = []
    if (folder / "rv_epochs.csv").exists():
        t = pd.read_csv(folder / "rv_epochs.csv")
        if len(t):
            t["STAR"] = t.get("STAR", pd.Series(["1"] * len(t))).astype(str).str.upper()
            lam = t[t.SOURCE.astype(str).str.contains("LAMOST", case=False)].to_dict("records")
    out["lamost"] = lam

    # period and epoch
    cand = folder / "candidate_periods.csv"
    if cand.exists():
        c = pd.read_csv(cand).sort_values("probability", ascending=False).iloc[0]
        out["P"], out["P_err"] = float(c.period_days), float(c.plus_minus)
    else:
        out["P"] = out["P_err"] = None
        warn(f"{folder}: no candidate_periods.csv (the period needs two eclipses)")
    out["T0"] = number(text, r"KNOWN ECLIPSE in \S+: cent(?:re|er) JD ([\d.]+)")
    t0_err_min = number(text, r"mid-eclipse uncertainty ([\d.]+) minutes")
    out["T0_err"] = t0_err_min / 1440 if t0_err_min is not None else None
    if out["T0"] is not None and out["T0_err"] is None:
        warn(f"{folder}: summary.txt has no epoch uncertainty; rerun the current joint_orbit.py")

    # model values
    out["ecc"] = number(text, r"Eccentricity ([\d.]+) \(68%: ([\d.]+) to ([\d.]+)", 0, str)
    ecc = re.findall(r"Eccentricity ([\d.]+) \(68%: ([\d.]+) to ([\d.]+)", text)
    out["ecc"] = tuple(float(v) for v in ecc[-1]) if ecc else None
    final = text.split("(all evidence)")[-1] if "(all evidence)" in text else ""
    mass = re.search(r"mass\s+([\d.]+) Msun\s+\(68%: ([\d.]+) to ([\d.]+);", final)
    out["m2"] = tuple(float(v) for v in mass.groups()) if mass else None
    out["m2_capped"] = "[piles up at the largest mass allowed]" in final.split("\n\n")[0]
    main = re.search(r"MAIN STAR: mass ([\d.]+) Msun, radius ([\d.]+) Rsun\s+\(68%: ([\d.]+) to ([\d.]+) Msun, "
                     r"([\d.]+) to ([\d.]+) Rsun", text)
    out["m1"] = (float(main.group(1)), float(main.group(3)), float(main.group(4))) if main else None
    out["r1"] = (float(main.group(2)), float(main.group(5)), float(main.group(6))) if main else None
    # older summaries: "...share of the catalogue's light (..)"; current: "...share of the TESS-band light (..)"
    share = re.search(r"companion's share of the (?:catalogue's|TESS-band) light \((\d+)%, 68%: (\d+) to (\d+)\)",
                      text)
    out["share"] = tuple(float(v) for v in share.groups()) if share else None
    r2 = re.search(r"companion radius ([\d.]+) Rsun \(68%: ([\d.]+) to ([\d.]+)\)", text)
    if r2:
        out["r2"] = tuple(float(v) for v in r2.groups())
    else:                                                  # older summaries: the radius under the final stage
        r2 = re.search(r"radius\s+([\d.]+) Rsun\s+\(68%: ([\d.]+) to ([\d.]+);", final)
        out["r2"] = tuple(float(v) for v in r2.groups()) if r2 else None
    temps = re.search(r"temperatures: eclipsed star (\d+) K \(68%: (\d+) to (\d+)\), companion (\d+) K "
                      r"\(68%: (\d+) to (\d+)\)", text)
    out["teff1"] = tuple(float(temps.group(i)) for i in (1, 2, 3)) if temps else None
    out["teff2"] = tuple(float(temps.group(i)) for i in (4, 5, 6)) if temps else None
    if temps is None and text:
        warn(f"{folder}: summary.txt has no 'temperatures:' line (older joint_orbit.py); rerun it to fill "
             "in the stars' temperatures")
    fr = re.search(r"flux ratio fainter/brighter in (\w+): ([\d.]+) \(68%: ([\d.]+) to ([\d.]+)\); "
                   r"measured ([\d.]+) \+/- ([\d.]+)", text)
    out["fr_band"] = fr.group(1) if fr else None
    out["fr_model"] = tuple(float(fr.group(i)) for i in (2, 3, 4)) if fr else None
    out["fr_meas"] = (float(fr.group(5)), float(fr.group(6))) if fr else None
    brighter = re.search(r"the companion is the brighter star in (\d+)% of the probability", text)
    out["comp_brighter"] = float(brighter.group(1)) if brighter else (0.0 if main else None)
    iso = re.search(r"STARS: isochrone (\S+) at log age ([\d.]+)", text)
    out["iso"] = (iso.group(1), float(iso.group(2))) if iso else None
    out["third_light"] = number(text, r"third light up to (\d+)% allowed")
    out["gaia_rv_median_only"] = "robust amplitude not used" in text
    if main is None:
        warn(f"{folder}: no blend-corrected MAIN STAR line; rerun joint_orbit.py without --no-blend-correct")
    if "POSSIBLE COMPANION ECLIPSE" in text:
        out["secondary"] = "possible (unconfirmed)"
    elif "PART F" in text:
        clear = number(text, r"no eclipse at all \(it passes clear of the star\):\s+([\d.]+)%")
        out["secondary"] = "not detected" + (f" (model: none in {clear:.0f}\\% of orbits)" if clear is not None else "")
    elif "--skip-secondary" in text or "PART E" in text:
        out["secondary"] = "not tested"
    else:
        out["secondary"] = MISSING

    # the eclipses themselves, measured exactly as in the figure
    try:
        import rnaas_figure as rf
        s = rf.gather(folder)
        ev = s["events"]
        tess = [e for e in ev if e["kind"] == "TESS"]
        sectors = sorted({re.sub(r"\D", "", e["source"]) for e in tess}, key=int)
        n_atlas = sum(e["kind"] == "ATLAS" for e in ev)
        out["seen"] = ("TESS " + ", ".join(f"S{x}" for x in sectors) if sectors else "") + \
            (f"; ATLAS ({n_atlas})" if n_atlas else "")
        if tess:
            w = np.array([1 / e["depth_err"] ** 2 for e in tess])
            d = np.array([e["depth"] for e in tess])
            out["depth"] = (100 * float(np.sum(w * d) / w.sum()), 100 * float(max(1 / np.sqrt(w.sum()), np.std(d))))
        else:
            out["depth"] = None
        out["length_h"] = 2 * s["half"] * 24
    except Exception as exc:
        warn(f"{folder}: could not measure the eclipses ({str(exc)[:100]}); depth, length and coverage left blank")
        out["seen"], out["depth"], out["length_h"] = MISSING, None, None
    # the duration: first to last contact of the eclipse joint_orbit.py fitted (summary.txt), rather
    # than measured on the drawn model, whose shallow edges a cutoff in depth would trim
    fitted = number(text, r"BEST-FIT ECLIPSE: .*?length ([\d.]+) hours")
    if fitted is not None:
        out["length_h"] = fitted
    elif out.get("length_h") is not None:
        warn(f"{folder}: no BEST-FIT ECLIPSE line in summary.txt; duration measured on the drawn model "
             "(rerun the current joint_orbit.py)")
    if out.get("depth") is None:
        d = number(text, r"deepest about ([\d.]+)%")
        out["depth"] = (d, None) if d is not None else None
    return out


def build(stars, cls):
    def cell(fn):
        return " & ".join(fn(s) for s in stars)

    def f(v, fmt):
        return MISSING if v is None else format(v, fmt)

    notes = {}

    def note(key, text):
        letter = notes.setdefault(key, (chr(ord("a") + len(notes)), text))[0]
        return f"\\tablenotemark{{{letter}}}"

    def rv_gaia(s):
        if s["rv"] is None:
            return r"\nodata"
        txt = pm(s["rv"], s["rv_err"], 2)
        if s["rv_n"]:
            if s["rv_method"] == 2:
                how = f"from the combined spectrum of {int(s['rv_n'])} RVS visits (no per-visit spread published)"
            else:
                how = f"median of {int(s['rv_n'])} RVS visits" + (
                    f", which span a robust range of {s['rv_amp']:.0f}~km\\,s$^{{-1}}$" if s["rv_amp"] else "")
            if s["lamost"] and any(str(r.get("STAR", "1")).upper() in ("A", "B") for r in s["lamost"]):
                how += "; with two sets of lines this is a blend of both stars"
                if s["gaia_rv_median_only"]:
                    how += ", and only the median was used in the model"
            txt += note(f"rv{s['tic']}", f"TIC {s['tic']}: {how}.")
        return txt

    def lamost(s):
        rows = s["lamost"]
        if not rows:
            return r"\nodata"
        kinds = {str(r.get("STAR", "1")).upper() for r in rows}
        if {"A", "B"} <= kinds:
            ab = [r for r in rows if str(r.get("STAR")).upper() in ("A", "B")]
            jd = float(ab[0]["JD"])
            jd = jd + 2400000.5 if jd < 2400000 else jd
            return (", ".join(f"${r['RV']:+.1f}$" for r in ab) + " (SB2)"
                    + note(f"sb2{s['tic']}", f"TIC {s['tic']}: two-component fit to the LAMOST medium-resolution "
                           f"spectrum of JD~{jd:.2f}; which set of lines belongs to the eclipsed star is not known."))
        r = rows[0]
        return f"${r['RV']:+.1f}$ (single-lined)"

    def model(key, d=2):
        def inner(s):
            v = s.get(key)
            if v is None:
                return MISSING
            return rng(*v, d=d)
        return inner

    def m2(s):
        if s["m2"] is None:
            return MISSING
        txt = rng(*s["m2"])
        if s["m2_capped"]:
            txt += note("cap", "The range runs into the largest companion mass the model allows, so its upper "
                        "end is a limit set by the model rather than by the data.")
        return txt

    def temp(key):
        def inner(s):
            v = s.get(key)
            return MISSING if v is None else rng(*v, d=0)
        return inner

    def flux_ratio(s):
        if s["fr_model"] is None:
            return r"\nodata"
        meas = s["fr_meas"]
        return (rng(*s["fr_model"]) + note(f"fr{s['tic']}", f"TIC {s['tic']}: fainter/brighter star in the "
                f"{s['fr_band']} band; the spectroscopic value {meas[0]:.2f}~$\\pm$~{meas[1]:.2f} was used as a "
                "constraint."))

    def brighter(s):
        v = s.get("comp_brighter")
        return MISSING if v is None else f"{v:.0f}"

    isos = {s["iso"] for s in stars if s.get("iso")}
    if isos and len(isos) == len([s for s in stars if s.get("iso")]) == len(stars):
        model_text = ("Medians with 68\\% ranges, from the eclipse's shape and length. Each star's radius, "
                      "temperature and light follow from its mass on one MIST isochrone fitted to the system's "
                      "photometry ("
                      + "; ".join(f"TIC {s['tic']}: log age {s['iso'][1]:.2f}" for s in stars)
                      + "); either star may be the brighter one.")
    elif isos:
        model_text = ("Medians with 68\\% ranges, from the eclipse's shape and length. Where an isochrone "
                      "was fitted (" + ", ".join(f"TIC {s['tic']}" for s in stars if s.get("iso"))
                      + ") each star's radius, temperature and light follow from its mass on it; otherwise the "
                      "catalog light is shared as $L \\propto M^{4.5}$ with a main-sequence mass--radius relation.")
    else:
        model_text = ("Medians with 68\\% ranges, from the eclipse's shape and length, with the catalog mass and "
                      "radius shared between the stars by a mass--luminosity relation ($L \\propto M^{4.5}$) and "
                      "a main-sequence mass--radius relation.")
    if any(s.get("third_light") for s in stars):
        model_text += " Light from further unresolved stars was allowed for."

    lines = [
        rf"\documentclass[RNAAS]{{{cls}}}",
        r"\newcommand{\tbd}[1]{\textcolor{red}{#1}}",
        r"\begin{document}",
        r"\title{Table draft}",
        r"\section{}",
        r"Table~\ref{tab:systems}.",
        "",
        r"\begin{deluxetable}{l" + "c" * len(stars) + "}",
        r"\tablecaption{Properties of " + " and ".join(f"TIC~{s['tic']}" for s in stars) + r"\label{tab:systems}}",
        r"\tablewidth{0pt}",
        r"\tablehead{\colhead{Property} & " + " & ".join(f"\\colhead{{TIC {s['tic']}}}" for s in stars) + "}",
        r"\startdata",
        r"\sidehead{\textit{Identification and astrometry (Gaia DR3)}}",
        "Gaia DR3 source ID & " + cell(lambda s: s["source_id"] or MISSING) + r" \\",
        "R.A., Decl. (deg) & " + cell(lambda s: f"{s['ra']:.5f}, ${s['dec']:+.5f}$" if s["ra"] is not None else MISSING)
        + r" \\",
        "Parallax (mas) & " + cell(lambda s: pm(s["plx"], s["plx_err"], 2)) + r" \\",
        "RUWE & " + cell(lambda s: f(s["ruwe"], ".2f")) + r" \\",
        "Astrometric excess noise (mas) & " + cell(lambda s: f(s["aen"], ".2f")) + r" \\",
        r"\sidehead{\textit{Photometry and spectroscopy}}",
        r"$G$, $G_\mathrm{BP}-G_\mathrm{RP}$ (mag) & "
        + cell(lambda s: f"{f(s['G'], '.2f')}, {f(s['bprp'], '.2f')}") + r" \\",
        r"$T$ (TESS, mag) & " + cell(lambda s: f(s["tmag"], ".2f")) + r" \\",
        r"$T_\mathrm{eff}$ (TIC, combined light; K) & " + cell(lambda s: f(s["teff"], ".0f")) + r" \\",
        r"RV, Gaia DR3 (km\,s$^{-1}$) & " + cell(rv_gaia) + r" \\",
        r"RV, LAMOST MRS (km\,s$^{-1}$) & " + cell(lamost) + r" \\",
        r"\sidehead{\textit{Eclipses (measured)}}",
        "Period (d) & " + cell(lambda s: pm(s["P"], s["P_err"], 2)) + r" \\",
        r"Epoch of primary minimum (BJD$_\mathrm{TDB}$) & " + cell(lambda s: pm(s["T0"], s["T0_err"], 1)) + r" \\",
        r"Depth (\%) & " + cell(lambda s: pm(*s["depth"], 1) if s["depth"] else MISSING) + r" \\",
        "Duration (h) & " + cell(lambda s: f(s["length_h"], ".1f")) + r" \\",
        "Eclipses observed & " + cell(lambda s: s["seen"] or MISSING) + r" \\",
        "Secondary eclipse & " + cell(lambda s: s["secondary"]) + r" \\",
        r"\sidehead{\textit{Orbit and components (model-dependent)" + note("model", model_text) + "}}",
        "Eccentricity & " + cell(model("ecc")) + r" \\",
        r"Eclipsed star mass ($M_\odot$) & " + cell(model("m1")) + r" \\",
        r"Eclipsed star radius ($R_\odot$) & " + cell(model("r1")) + r" \\",
        r"Eclipsed star $T_\mathrm{eff}$ (K) & " + cell(temp("teff1")) + r" \\",
        r"Companion mass ($M_\odot$) & " + cell(m2) + r" \\",
        r"Companion radius ($R_\odot$) & " + cell(model("r2")) + r" \\",
        r"Companion $T_\mathrm{eff}$ (K) & " + cell(temp("teff2")) + r" \\",
        r"Companion share of $T$-band light (\%) & " + cell(model("share", 0)) + r" \\",
    ] + ([r"Flux ratio, fainter/brighter & " + cell(flux_ratio) + r" \\"]
         if any(s["fr_model"] for s in stars) else []) + [
        r"Chance the companion is the brighter star (\%) & " + cell(brighter) + r" \\",
        r"\enddata",
    ]
    for key, (letter, text) in sorted(notes.items(), key=lambda kv: kv[1][0]):
        lines.append(rf"\tablenotetext{{{letter}}}{{{text}}}")
    lines += [r"\tablecomments{Measured values carry 1$\sigma$ errors. The depth is the TESS value at mid-eclipse; "
              r"the duration runs from first to last contact. The epoch is the center of the eclipse's missing light, "
              r"which agrees with the time of minimum to within its error.}",
              r"\end{deluxetable}", r"\end{document}", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="The RNAAS table, filled in from joint_orbit.py folders.")
    p.add_argument("folders", nargs="+")
    p.add_argument("--out", default="rnaas_table")
    p.add_argument("--class", dest="cls", default="aastex631", help="aastex631 (default) or aastex701")
    p.add_argument("--pdf", action="store_true", help="also run pdflatex")
    args = p.parse_args()

    missing = []
    stars = [gather(f, missing.append) for f in args.folders]
    tex = build(stars, args.cls)
    out = Path(args.out).with_suffix(".tex")
    out.write_text(tex)
    blanks = tex.count(r"\tbd{?}")
    print(f"Wrote {out}" + (f"; {blanks} value(s) could not be found and are marked in red:" if blanks else ""))
    for m in missing:
        print("  " + m)
    if args.pdf:
        for _ in range(2):
            run = subprocess.run(["pdflatex", "-interaction=nonstopmode", out.name], cwd=out.parent or ".",
                                 capture_output=True, text=True)
        pdf = out.with_suffix(".pdf")
        print(f"Wrote {pdf}" if pdf.exists() else "pdflatex failed; see the .log file")


if __name__ == "__main__":
    main()
