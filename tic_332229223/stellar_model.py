#!/usr/bin/env python
"""
stellar_model.py - the stars' masses, radii and light from one MIST isochrone,
instead of main-sequence relations and light ~ mass^4.5.

Why
---
The TESS Input Catalog gives one mass, radius and temperature for the combined
light. joint_orbit.py and triple_fit.py used to share that light out with
L ~ M^4.5, take the eclipsed star's radius from its share of the light, and the
companion's from a main-sequence table. That is not self-consistent when the
system is over-luminous for its masses (an evolved star, or extra light): the
eclipsed star then comes out large while its companion stays on the main sequence.

This module fits the system's photometry (Gaia G, BP, RP; 2MASS J, H, Ks; TESS T)
and parallax with two or three coeval stars on a MIST isochrone, with age,
extinction and distance free. Every star's radius, temperature and brightness
in each band then follows from its mass at the fitted age, and the eclipsed
star's mass is whatever makes the stars' TESS-band light add up to what is
observed. Spectroscopic constraints (component temperatures, surface gravities
or a flux ratio, e.g. from a double-lined spectrum) can be added as priors on
the brighter (A) and fainter (B) star.

Getting an isochrone file
-------------------------
Easiest: --iso auto --feh <[Fe/H]> downloads MIST's packaged UBVRIplus
isochrones once (into ~/.cache/mist) and takes the file nearest that [Fe/H].
By hand: from https://waps.cfa.harvard.edu/MIST/, take the UBVRIplus .iso.cmd file
(UBV(RI)c+2MASS+Kepler+Hipparcos+Tycho+Gaia+TESS) nearest the star's [Fe/H], or
use the web interpolator with that [Fe/H], log10(age) 9.0-10.1 in steps of 0.02
and the same photometric system, and put it in the star's folder (any name
ending .iso.cmd), or give its path with --iso.

Usage
-----
joint_orbit.py runs this itself (see its --iso option). On its own:

    python stellar_model.py joint_<TIC>                            # two stars
    python stellar_model.py joint_<TIC> --nstars 3                 # allow a third
    python stellar_model.py joint_<TIC> --iso auto --feh -0.2
    python stellar_model.py joint_<TIC> --teff-a 6250 150 --teff-b 5600 200

Writes stellar_split.json (posterior summary, the isochrone near the fitted age,
and the observed absolute TESS flux) and stellar_split_samples.csv. joint_orbit.py
and triple_fit.py use stellar_split.json automatically when it is in the folder.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# extinction in each band relative to V (Wang & Chen 2019; TESS taken as ~RP)
A_OVER_AV = {"G": 0.789, "BP": 1.002, "RP": 0.589, "J": 0.243, "H": 0.131, "K": 0.078, "T": 0.60}
BAND_COLUMNS = {
    "G": ("Gaia_G_EDR3", "Gaia_G_DR2Rev", "Gaia_G_MAW"),
    "BP": ("Gaia_BP_EDR3", "Gaia_BP_DR2Rev", "Gaia_BP_MAWf"),
    "RP": ("Gaia_RP_EDR3", "Gaia_RP_DR2Rev", "Gaia_RP_MAW"),
    "J": ("2MASS_J",), "H": ("2MASS_H",), "K": ("2MASS_Ks",), "T": ("TESS",),
}
MODEL_FLOOR = 0.02            # mag added in quadrature: isochrone and zero-point systematics


# ----------------------------------------------------------------------------
# Getting and reading MIST isochrones
# ----------------------------------------------------------------------------
MIST_TARBALL = "https://waps.cfa.harvard.edu/MIST/data/tarballs_v1.2/MIST_v1.2_vvcrit0.4_UBVRIplus.txz"


def mist_cache():
    return Path.home() / ".cache" / "mist"


def fetch_mist(feh, say=print):
    """The MIST v1.2 UBVRIplus isochrone file with [Fe/H] nearest `feh`, downloading MIST's
    packaged synthetic photometry once into ~/.cache/mist (a large file). Returns its path."""
    import tarfile
    import urllib.request
    cache = mist_cache()
    cache.mkdir(parents=True, exist_ok=True)
    files = sorted(cache.glob("*.iso.cmd"))
    if not files:
        tarball = cache / Path(MIST_TARBALL).name
        if not tarball.exists():
            say(f"  downloading MIST isochrones (once) from {MIST_TARBALL}")
            urllib.request.urlretrieve(MIST_TARBALL, tarball)
        with tarfile.open(tarball) as tar:
            for m in tar.getmembers():
                if m.name.endswith(".iso.cmd"):
                    m.name = Path(m.name).name
                    tar.extract(m, cache)
        files = sorted(cache.glob("*.iso.cmd"))

    def file_feh(p):
        m = re.search(r"feh_([mp])(\d+\.\d+)", p.name)
        return (-1 if m.group(1) == "m" else 1) * float(m.group(2)) if m else np.nan

    fehs = np.array([file_feh(p) for p in files])
    if not np.isfinite(fehs).any():
        raise FileNotFoundError(f"no MIST .iso.cmd files with [Fe/H] in their names in {cache}")
    return files[int(np.nanargmin(np.abs(fehs - feh)))]


# ----------------------------------------------------------------------------
# Reading MIST isochrones
# ----------------------------------------------------------------------------
def read_mist(path, max_phase=2):
    """Read a MIST .iso.cmd file into a table of (log_age, initial mass, radius, Teff, log g,
    absolute magnitudes). Pre-main-sequence points and anything past the red-giant branch
    (phase > max_phase) are dropped."""
    header, rows = None, []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                parts = s[1:].split()
                if parts and parts[0] == "EEP":
                    header = parts
                continue
            if header is not None:
                rows.append(s.split())
    if header is None or not rows:
        raise ValueError(f"{path}: not a MIST isochrone file (no '# EEP ...' header)")
    df = pd.DataFrame(rows, columns=header[:len(rows[0])]).apply(pd.to_numeric, errors="coerce")
    if "phase" in df:
        df = df[(df["phase"] >= 0) & (df["phase"] <= max_phase)]
    out = pd.DataFrame({
        "log_age": df["log10_isochrone_age_yr"],
        "mass": df["initial_mass"],
        "teff": 10 ** df["log_Teff"],
        "logg": df["log_g"],
    })
    out["radius"] = np.sqrt(10 ** df["log_L"]) / (out["teff"] / 5772.0) ** 2
    for band, names in BAND_COLUMNS.items():
        col = next((n for n in names if n in df.columns), None)
        if col is None and band != "T":
            raise ValueError(f"{path}: no {band} magnitudes (need the UBVRIplus photometric system)")
        out[band] = df[col] if col else np.nan
    feh = df["[Fe/H]_init"].iloc[0] if "[Fe/H]_init" in df else np.nan
    return out.dropna(subset=["mass", "teff", "G"]).reset_index(drop=True), float(feh)


def grid_from_table(tab, n_mass=400):
    """Put every age on one common initial-mass grid (NaN beyond each age's last mass)."""
    ages = np.sort(tab["log_age"].unique())
    m_lo = tab["mass"].min()
    m_hi = tab["mass"].max()
    masses = np.exp(np.linspace(np.log(max(m_lo, 0.1)), np.log(m_hi), n_mass))
    keys = ["radius", "teff", "logg"] + list(BAND_COLUMNS)
    grid = {k: np.full((len(ages), n_mass), np.nan) for k in keys}
    for i, a in enumerate(ages):
        sub = tab[tab["log_age"] == a].sort_values("mass")
        sub = sub[np.r_[True, np.diff(sub["mass"].to_numpy()) > 0]]
        inside = (masses >= sub["mass"].min()) & (masses <= sub["mass"].max())
        for k in keys:
            grid[k][i, inside] = np.interp(masses[inside], sub["mass"], sub[k])
    return {"log_age": ages, "mass": masses, **grid}


class SystemIsochrone:
    """Interpolation in an age x initial-mass grid. `at(log_age)` gives a single isochrone."""

    def __init__(self, grid):
        self.g = {k: np.asarray(v, float) for k, v in grid.items()}
        self.ages = self.g["log_age"]
        self.masses = self.g["mass"]

    @classmethod
    def from_json(cls, block):
        return cls({k: np.asarray(v, float) for k, v in block.items()})

    def at(self, log_age):
        a = float(np.clip(log_age, self.ages[0], self.ages[-1]))
        i = int(np.clip(np.searchsorted(self.ages, a) - 1, 0, len(self.ages) - 2))
        w = (a - self.ages[i]) / (self.ages[i + 1] - self.ages[i])
        iso = {k: (1 - w) * self.g[k][i] + w * self.g[k][i + 1]
               for k in self.g if k not in ("log_age", "mass")}
        return Isochrone(self.masses, iso)


class Isochrone:
    """One isochrone: properties of a star of any initial mass up to the last one alive."""

    def __init__(self, masses, props):
        ok = np.isfinite(props["G"])
        self.m = masses[ok]
        self.p = {k: v[ok] for k, v in props.items()}
        self.m_max = float(self.m[-1]) if len(self.m) else 0.0
        self.m_min = float(self.m[0]) if len(self.m) else 0.0

    def __call__(self, key, mass):
        return np.interp(mass, self.m, self.p[key], left=np.nan, right=np.nan)

    def flux(self, band, mass):
        """Flux relative to a star of absolute magnitude 0 in that band."""
        return 10 ** (-0.4 * self(band, mass))

    def mass_for_flux(self, band, target):
        """Mass whose flux in `band` equals `target` (on the rising part of the isochrone);
        NaN when no star on it is that faint or that bright."""
        f = 10 ** (-0.4 * self.p[band])
        ok = np.r_[True, np.diff(f) > 0]                          # keep the monotonic part
        mm, ff = self.m[ok], f[ok]
        target = np.asarray(target, float)
        return np.where((target >= ff[0]) & (target <= ff[-1]), np.interp(target, ff, mm), np.nan)


def load_split(folder):
    """stellar_split.json from a star's folder, or None."""
    path = Path(folder) / "stellar_split.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    d["system"] = SystemIsochrone.from_json(d["grid"])
    d["iso"] = d["system"].at(d["log_age"][0])
    return d


# ----------------------------------------------------------------------------
# Fitting the photometry
# ----------------------------------------------------------------------------
def combined_mags(iso, masses, dm, av):
    out = {}
    for band in BAND_COLUMNS:
        f = sum(iso.flux(band, m) for m in masses)
        out[band] = -2.5 * np.log10(f) + dm + A_OVER_AV[band] * av
    return out


def gather_photometry(folder, args):
    star = json.loads((folder / "star.json").read_text(encoding="utf-8-sig"))
    gaia = json.loads((folder / "gaia.json").read_text())
    phot = {}
    for key, (m, e) in (star.get("photometry") or {}).items():
        phot[key] = (float(m), float(e))
    if not phot:
        phot = fetch_photometry(star, gaia)
        star["photometry"] = phot
        (folder / "star.json").write_text(json.dumps(star, indent=1))
    for item in args.mag or []:
        band, val = item.split("=")
        m, e = val.split(":")
        phot[band.strip().upper()] = (float(m), float(e))
    return star, gaia, phot


def fetch_photometry(star, gaia):
    """Gaia G/BP/RP from the archive and 2MASS J/H/Ks and TESS T from the TIC."""
    from astroquery.gaia import Gaia
    from astroquery.mast import Catalogs
    phot = {}
    q = f"""SELECT phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                   phot_g_mean_flux_over_error, phot_bp_mean_flux_over_error, phot_rp_mean_flux_over_error
            FROM gaiadr3.gaia_source WHERE source_id = {int(gaia['source_id'])}"""
    r = Gaia.launch_job(q).get_results().to_pandas().iloc[0]
    for b, n in (("G", "g"), ("BP", "bp"), ("RP", "rp")):
        phot[b] = (float(r[f"phot_{n}_mean_mag"]), float(1.0857 / r[f"phot_{n}_mean_flux_over_error"]))
    row = Catalogs.query_criteria(catalog="Tic", ID=int(star["tic"]))[0]
    for b, col in (("J", "Jmag"), ("H", "Hmag"), ("K", "Kmag"), ("T", "Tmag")):
        if not np.ma.is_masked(row[col]):
            err = row[f"e_{col}"] if f"e_{col}" in row.colnames and not np.ma.is_masked(row[f"e_{col}"]) else 0.03
            phot[b] = (float(row[col]), float(err))
    return phot


def fit(system, phot, parallax, args, say=print):
    """Sample (log age, A_V, parallax, masses) with emcee. Masses are ordered A >= B (>= C)."""
    import emcee
    bands = [b for b in BAND_COLUMNS if b in phot]
    if args.skip_band:
        bands = [b for b in bands if b not in [s.upper() for s in args.skip_band]]
    obs = np.array([phot[b][0] for b in bands])
    err = np.sqrt(np.array([phot[b][1] for b in bands]) ** 2 + MODEL_FLOOR**2)
    plx, plx_err = parallax
    n = args.nstars
    a_lo, a_hi = system.ages[0], system.ages[-1]

    spec = []                                    # (star index, key, value, error)
    for idx, name in ((0, "a"), (1, "b")):
        for key in ("teff", "logg"):
            v = getattr(args, f"{key}_{name}")
            if v:
                spec.append((idx, key, v[0], v[1]))

    def logpost(theta):
        log_age, av, p, *ms = theta
        if not (a_lo <= log_age <= a_hi) or av < 0 or p <= 0:
            return -np.inf
        if any(ms[i] < ms[i + 1] for i in range(n - 1)) or ms[-1] < 0.1:
            return -np.inf
        iso = system.at(log_age)
        if ms[0] > iso.m_max:
            return -np.inf
        model = combined_mags(iso, ms, 5 * np.log10(100.0 / p), av)
        mod = np.array([model[b] for b in bands])
        if not np.all(np.isfinite(mod)):
            return -np.inf
        lp = -0.5 * np.sum(((obs - mod) / err) ** 2)
        lp += -0.5 * ((p - plx) / plx_err) ** 2 - 0.5 * ((av - args.av[0]) / args.av[1]) ** 2
        for idx, key, v, e in spec:
            lp += -0.5 * ((iso(key, ms[idx]) - v) / e) ** 2
        if args.flux_ratio:
            band, v, e = args.flux_ratio[0].upper(), float(args.flux_ratio[1]), float(args.flux_ratio[2])
            lp += -0.5 * ((iso.flux(band, ms[1]) / iso.flux(band, ms[0]) - v) / e) ** 2
        return lp

    # start from a coarse search over age and the brighter mass
    best, start = -np.inf, None
    for la in np.linspace(a_lo, a_hi, 25):
        iso = system.at(la)
        for m0 in np.linspace(0.5, iso.m_max, 25):
            for q in (0.5, 0.75, 0.95):
                ms = [m0, m0 * q] + ([m0 * q * 0.6] if n == 3 else [])
                lp = logpost([la, args.av[0], plx] + ms)
                if lp > best:
                    best, start = lp, np.array([la, args.av[0], plx] + ms)
    if start is None:
        sys.exit("No point on the isochrone fits the photometry at all; check the bands and the file.")
    ndim = len(start)
    nw = max(4 * ndim, 32)
    rng = np.random.default_rng(args.seed)
    p0 = start + rng.normal(0, 1, (nw, ndim)) * np.r_[0.02, 0.02, plx_err * 0.3, [0.01] * n]
    p0[:, 3:] = -np.sort(-p0[:, 3:], axis=1)
    for i in range(nw):
        while not np.isfinite(logpost(p0[i])):
            p0[i] = start + rng.normal(0, 1, ndim) * np.r_[0.005, 0.005, plx_err * 0.1, [0.002] * n]
            p0[i, 3:] = -np.sort(-p0[i, 3:])
    sampler = emcee.EnsembleSampler(nw, ndim, logpost)
    sampler.run_mcmc(p0, args.steps, progress=False)
    chain = sampler.get_chain(discard=args.steps // 2, flat=True)
    lps = sampler.get_log_prob(discard=args.steps // 2, flat=True)
    names = ["log_age", "av", "parallax"] + ["m_a", "m_b", "m_c"][:n]
    df = pd.DataFrame(chain, columns=names)
    df["log_post"] = lps
    i_best = int(np.argmax(lps))
    chi2 = -2 * (lps[i_best] - 0.0)
    return df, bands, chi2, logpost


def parser():
    import argparse
    p = argparse.ArgumentParser(description="Fit the system's photometry with 2 or 3 coeval MIST stars.")
    p.add_argument("folder")
    p.add_argument("--iso", default=None,
                   help="MIST .iso.cmd file (default: the one in the folder); 'auto' downloads MIST's "
                        "UBVRIplus isochrones once and takes the file nearest --feh")
    p.add_argument("--feh", type=float, default=0.0, help="[Fe/H] for --iso auto (e.g. LAMOST's; default 0)")
    p.add_argument("--nstars", type=int, default=2, choices=(2, 3))
    p.add_argument("--av", type=float, nargs=2, default=(0.12, 0.08), metavar=("MEAN", "SD"),
                   help="prior on V-band extinction (default 0.12 +/- 0.08)")
    p.add_argument("--teff-a", type=float, nargs=2, metavar=("K", "ERR"), help="brighter star's Teff")
    p.add_argument("--teff-b", type=float, nargs=2, metavar=("K", "ERR"), help="fainter star's Teff")
    p.add_argument("--logg-a", type=float, nargs=2, metavar=("DEX", "ERR"))
    p.add_argument("--logg-b", type=float, nargs=2, metavar=("DEX", "ERR"))
    p.add_argument("--flux-ratio", nargs=3, metavar=("BAND", "B/A", "ERR"),
                   help="flux ratio fainter/brighter in a band (G, BP, RP, J, H, K or T)")
    p.add_argument("--mag", nargs="*", metavar="BAND=MAG:ERR", help="override or add a magnitude")
    p.add_argument("--skip-band", nargs="*", default=[], help="leave bands out (e.g. T if it is not independent)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=1)
    return p


def run(args, say=print):
    """Fit the photometry and write stellar_split.json; returns its path. `args` as from parser()."""
    folder = Path(args.folder)
    if args.iso == "auto":
        iso_path = fetch_mist(args.feh, say)
    else:
        iso_path = Path(args.iso) if args.iso else next(iter(sorted(folder.glob("*.iso.cmd"))), None)
    if iso_path is None or not iso_path.exists():
        raise FileNotFoundError("no MIST isochrone (.iso.cmd); see the top of stellar_model.py for where to get one")
    tab, feh = read_mist(iso_path)
    system = SystemIsochrone(grid_from_table(tab))
    star, gaia, phot = gather_photometry(folder, args)
    say(f"Isochrone: {iso_path.name} ([Fe/H] = {feh:+.2f}), log age {system.ages[0]:.2f}-{system.ages[-1]:.2f}")
    say("Photometry: " + ", ".join(f"{b} {m:.3f}+/-{e:.3f}" for b, (m, e) in phot.items()))
    plx = (gaia["parallax"], gaia.get("parallax_error") or 0.03)

    df, bands, chi2, logpost = fit(system, phot, plx, args)
    q = df.quantile([0.16, 0.5, 0.84])
    best = df.iloc[int(df["log_post"].idxmax())]
    iso_best = system.at(q.loc[0.5, "log_age"])
    say(f"\nBest fit: -2 ln(posterior) {chi2:.1f} (chi-squared plus prior terms) for {len(bands)} bands "
          f"({', '.join(bands)}) "
          f"with {args.nstars} stars" + (" plus spectroscopic priors" if (args.teff_a or args.teff_b
                                          or args.logg_a or args.logg_b or args.flux_ratio) else ""))
    for col in df.columns[:-1]:
        say(f"  {col:9s} {q.loc[0.5, col]:.3f}  (-{q.loc[0.5, col] - q.loc[0.16, col]:.3f} "
              f"+{q.loc[0.84, col] - q.loc[0.5, col]:.3f})")

    # derived: radii, temperatures, light shares, and the observed absolute TESS-band flux
    derived = {}
    names = ["m_a", "m_b", "m_c"][:args.nstars]
    sub = df.sample(min(2000, len(df)), random_state=1)
    rows = []
    for _, r in sub.iterrows():
        iso = system.at(r["log_age"])
        ms = [r[n] for n in names]
        row = {"log_age": r["log_age"]}
        tot = {b: sum(iso.flux(b, m) for m in ms) for b in ("G", "RP", "T")}
        for n, m in zip(names, ms):
            k = n[-1]
            row[f"r_{k}"], row[f"teff_{k}"], row[f"logg_{k}"] = iso("radius", m), iso("teff", m), iso("logg", m)
            row[f"lT_{k}"] = iso.flux("T", m) / tot["T"]
        dm = 5 * np.log10(100.0 / r["parallax"])
        t_obs = phot["T"][0] if "T" in phot else np.nan
        row["fT_obs"] = 10 ** (-0.4 * (t_obs - dm - A_OVER_AV["T"] * r["av"]))
        row["fT_model"] = tot["T"]
        rows.append(row)
    dd = pd.DataFrame(rows)
    for col in dd.columns:
        lo, mid, hi = np.nanpercentile(dd[col], [16, 50, 84])
        derived[col] = [float(mid), float(lo), float(hi)]
    say("  derived (median, 16th, 84th):")
    for k, v in derived.items():
        say(f"    {k:9s} {v[0]:.4g}  ({v[1]:.4g} .. {v[2]:.4g})")
    if args.nstars == 2:
        say("\n  If chi-squared is poor, or the brighter star has to be well off the main sequence\n"
              "  to supply the light, rerun with --nstars 3 and compare.")

    # the isochrone grid near the fitted ages (kept small) and the summary
    keep = (system.ages >= q.loc[0.16, "log_age"] - 0.1) & (system.ages <= q.loc[0.84, "log_age"] + 0.1)
    idx = np.where(keep)[0]
    idx = np.arange(max(idx.min() - 1, 0), min(idx.max() + 2, len(system.ages)))
    grid = {"log_age": system.ages[idx].tolist(), "mass": system.masses.tolist()}
    for k in ["radius", "teff", "logg"] + list(BAND_COLUMNS):
        grid[k] = np.where(np.isfinite(system.g[k][idx]), system.g[k][idx], None).tolist()
    fT = derived.get("fT_obs", [np.nan] * 3)
    out = {
        "isochrone_file": iso_path.name, "feh": feh, "nstars": args.nstars, "bands": bands,
        "chi2_best": float(chi2),
        "log_age": [float(q.loc[0.5, "log_age"]), float(q.loc[0.16, "log_age"]), float(q.loc[0.84, "log_age"])],
        "av": [float(q.loc[0.5, "av"]), float(q.loc[0.16, "av"]), float(q.loc[0.84, "av"])],
        "masses": {n: [float(q.loc[0.5, n]), float(q.loc[0.16, n]), float(q.loc[0.84, n])] for n in names},
        "fT_obs": fT,                                     # absolute TESS-band flux of the whole system
        "fT_obs_rel_err": float((fT[2] - fT[1]) / (2 * fT[0])) if np.isfinite(fT[0]) else 0.05,
        "derived": derived, "photometry": {k: list(v) for k, v in phot.items()}, "grid": grid,
    }
    (folder / "stellar_split.json").write_text(json.dumps(out, indent=1))
    df.to_csv(folder / "stellar_split_samples.csv", index=False)
    say(f"\nWrote {folder / 'stellar_split.json'} and stellar_split_samples.csv")
    return folder / "stellar_split.json"


def main():
    try:
        run(parser().parse_args())
    except FileNotFoundError as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
