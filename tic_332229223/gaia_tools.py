#!/usr/bin/env python
"""
gaia_tools.py - forward models of what Gaia DR3 recorded, shared by joint_orbit.py
and triple_fit.py.

Two things are modelled here, both at the level of individual Gaia transits:

1. ASTROMETRY (RUWE). Gaia measures the star's position along its scan direction
   on every CCD it crosses (about 8 per transit). DR3 fits five parameters
   (position, proper motion, parallax) to these, and RUWE is essentially the
   square root of the reduced chi-squared of that fit, normalised so that well
   behaved single stars sit near 1. An orbit adds displacements the five-
   parameter model cannot absorb; this module projects any predicted photocentre
   track onto the real scan directions, removes the best five-parameter fit, and
   turns what is left into a predicted RUWE and a likelihood for the observed one.

   The likelihood is a mixture. Most stars of this brightness and colour scatter
   tightly about the median RUWE ("core"); a few percent sit far above it for
   reasons no orbit model here would capture (resolved neighbours, bad
   calibration, companions outside the model). That fraction, measured from
   comparison stars, caps how much any single model can be penalised for missing
   the observed value. Without it a 2-sigma mismatch would count as a 40-sigma one.

2. RADIAL VELOCITIES. Gaia's DR3 velocity for a star this bright (G_RVS < 12) is
   the median of its per-transit velocities, and rv_amplitude_robust is their
   range after outliers are removed. With two sets of lines, the per-transit
   cross-correlation follows the light-weighted blend when the lines overlap and
   locks onto the brighter star once they separate by more than about the
   spectrograph's resolution (R ~ 11,500, ~26 km/s, plus line widths).

Scan times and angles
---------------------
Taken automatically from Gaia's DR3 commanded scan law through the gaiaunlimited
package (pip install gaiaunlimited; its first use downloads the scan law once),
and saved in the star's folder as gaia_scanlaw.csv. A GOST export
(https://gaia.esac.esa.int/gost/, dates 2014-07-25 .. 2017-05-28) saved as
gost.csv, or a gaiamock healpix_scans/*.fits file, takes precedence if present.
Without any of these a synthetic scan law is used and every result says so: the
predicted RUWE is then good to perhaps 20%.

Comparison stars
----------------
`calibrate()` queries Gaia DR3 for stars near the target with similar G, colour
and well-measured parallax and records the median RUWE, the spread of the core,
and the fraction above 1.4. Saved as gaia_calibration.json. Defaults are used
(and reported) if it is missing.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DR3_START, DR3_END = 2456863.5, 2457901.5          # 2014-07-25 .. 2017-05-28
DR3_REF = 2457389.0                                 # J2016.0, DR3's reference epoch
CCDS_PER_TRANSIT = 8                                # astrometric field CCDs (8 or 9; 8 as in gaiamock)


# ----------------------------------------------------------------------------
# Per-CCD along-scan precision
# ----------------------------------------------------------------------------
def al_sigma_ccd(g):
    """Along-scan uncertainty per CCD (mas) at magnitude G: the 'EDR3 adjusted' curve of
    Holl et al. (2023, Fig. 3), as tabulated in gaiamock (El-Badry et al. 2024)."""
    gv = [4, 5, 6, 7, 8.2, 8.4, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    sv = [0.4, 0.35, 0.15, 0.17, 0.23, 0.13, 0.13, 0.135, 0.125, 0.13, 0.15, 0.23, 0.36, 0.63, 1.05,
          2.05, 4.1]
    return float(np.interp(g, gv, sv))


# ----------------------------------------------------------------------------
# Scan law: GOST file, gaiamock file, or a synthetic stand-in
# ----------------------------------------------------------------------------
def _col(columns, *keys):
    for key in keys:
        for c in columns:
            if key.lower() in c.lower():
                return c
    return None


def load_scan_law(path, start=DR3_START, end=DR3_END):
    """Transit times (BJD), scan angles (rad) and along-scan parallax factors from a GOST
    CSV export or a gaiamock healpix_scans FITS file, restricted to [start, end]."""
    path = Path(path)
    if path.suffix.lower() in (".fits", ".fit"):
        from astropy.table import Table
        df = Table.read(path).to_pandas()
    else:
        df = pd.read_csv(path, comment="#")
    tcol = _col(df.columns, "ObservationTimeAtBarycentre", "BarycentricJulianDate")
    pcol = _col(df.columns, "scanAngle")
    fcol = _col(df.columns, "parallaxFactorAlongScan")
    if tcol is None or pcol is None:
        raise ValueError(f"{path}: no transit-time or scan-angle column (is this a GOST export?)")
    t = pd.to_numeric(df[tcol], errors="coerce").to_numpy(float)
    psi = pd.to_numeric(df[pcol], errors="coerce").to_numpy(float)
    plx = pd.to_numeric(df[fcol], errors="coerce").to_numpy(float) if fcol else np.full(len(t), np.nan)
    ok = np.isfinite(t) & np.isfinite(psi) & (t >= start) & (t <= end)
    t, psi, plx = t[ok], psi[ok], plx[ok]
    order = np.argsort(t)
    return {"t": t[order], "psi": psi[order], "plx": plx[order], "source": str(path), "synthetic": False}


def _earth_xyz(jd):
    """Barycentric Earth position (AU, equatorial), low precision (~0.01 AU)."""
    n = jd - 2451545.0
    L = np.radians(280.460 + 0.9856474 * n)
    g = np.radians(357.528 + 0.9856003 * n)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    r = 1.00014 - 0.01671 * np.cos(g) - 0.00014 * np.cos(2 * g)
    eps = np.radians(23.439)
    # the Sun's geocentric position; the Earth's heliocentric position is minus that
    xs, ys, zs = r * np.cos(lam), r * np.cos(eps) * np.sin(lam), r * np.sin(eps) * np.sin(lam)
    return -xs, -ys, -zs


def parallax_factor_al(jd, psi, ra_deg, dec_deg):
    a, d = np.radians(ra_deg), np.radians(dec_deg)
    x, y, z = _earth_xyz(jd)
    f_ra = x * np.sin(a) - y * np.cos(a)
    f_dec = x * np.cos(a) * np.sin(d) + y * np.sin(a) * np.sin(d) - z * np.cos(d)
    return f_ra * np.sin(psi) + f_dec * np.cos(psi)


def synthetic_scan_law(ra, dec, n_transits=40, seed=11):
    """A stand-in when no GOST file is available: transits in preceding/following-field
    pairs 106.5 minutes apart, at random times, with random scan angles."""
    rng = np.random.default_rng(seed)
    n_pairs = max(int(np.ceil(n_transits / 2)), 3)
    t1 = np.sort(rng.uniform(DR3_START, DR3_END, n_pairs))
    psi1 = rng.uniform(0, 2 * np.pi, n_pairs)
    t = np.ravel(np.column_stack([t1, t1 + 106.5 / 1440]))[:n_transits]
    psi = np.ravel(np.column_stack([psi1, psi1]))[:n_transits]
    return {"t": t, "psi": psi, "plx": parallax_factor_al(t, psi, ra, dec),
            "source": "synthetic (no gost.csv)", "synthetic": True}


def scan_law_from_gaiaunlimited(ra, dec, start=DR3_START, end=DR3_END):
    """Transit times and scan angles from Gaia's DR3 commanded scan law, via the gaiaunlimited
    package (Cantat-Gaudin et al. 2023). Its first use downloads the scan law (a large file, kept
    in ~/.gaiaunlimited); after that it works offline for any star.

    gaiaunlimited's own query() returns only times. The scan direction follows from the same
    pointing data: Gaia spins about the axis normal to both fields of view, so at a star s the
    scan runs along spin_axis x s. The angle is measured from north through east, the same
    convention as the parallax factors and the five-parameter design matrix used here, so the
    model is self-consistent (a 180-degree flip changes nothing).

    DR3's data gaps for astrometry ("dr3/Astrometry") are removed, as gaiaunlimited does.
    """
    from gaiaunlimited.scanninglaw import GaiaScanningLaw, cartesian_to_spherical, check_gaps
    sl = GaiaScanningLaw(version="dr3_nominal", gaplist="dr3/Astrometry")
    a, d = np.radians(ra), np.radians(dec)
    s = np.array([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a), np.sin(d)])
    north = np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])
    east = np.array([-np.sin(a), np.cos(a), 0.0])
    times, angles = [], []
    for tree, offset, lon0 in ((sl.tree_fov1, 221 / 3600.0, 0.0), (sl.tree_fov2, -221 / 3600.0, 106.5)):
        idx = tree.query_ball_point(s, sl.r_search, return_sorted=True)
        if len(idx) == 0:
            continue
        idx = np.array(idx)
        body = np.einsum("nij,j->ni", sl.rotmat[idx], s)
        lon, lat = cartesian_to_spherical(body)
        inside = idx[(np.abs(lat - offset) < 0.35) & (np.abs(lon - lon0) < 1.0)]
        if len(inside) == 0:
            continue
        first = inside[np.insert(np.diff(inside) > 360, 0, True)]      # one per field crossing
        t = sl.tcb_at_gaia[first]
        keep = check_gaps(sl.gaps, t)
        first, t = first[keep], t[keep]
        spin = sl.rotmat[first][:, 2, :]                                  # body z-axis in ICRS
        scan = np.cross(spin, s)
        times.append(t + 2455197.5)                                       # days since 2010.0 -> JD
        angles.append(np.arctan2(scan @ east, scan @ north))
    if not times:
        raise RuntimeError("the scan law has no transits at this position")
    t, psi = np.concatenate(times), np.concatenate(angles)
    ok = (t >= start) & (t <= end)
    order = np.argsort(t[ok])
    t, psi = t[ok][order], psi[ok][order]
    return {"t": t, "psi": psi, "plx": parallax_factor_al(t, psi, ra, dec),
            "source": "Gaia DR3 commanded scan law (gaiaunlimited)", "synthetic": False}


def save_scan_law(law, path):
    """Save a scan law with GOST's column names, so later runs (and triple_fit.py) reuse it."""
    pd.DataFrame({"ObservationTimeAtBarycentre[BarycentricJulianDateInTCB]": law["t"],
                  "scanAngle[rad]": law["psi"], "parallaxFactorAlongScan": law["plx"]}).to_csv(path, index=False)


def scan_law_for(folder, ra, dec, n_obs_al=None, explicit=None, say=print, fetch=True):
    """The scan law for a star, from the first of these that is available:
    an explicit file; gost*.csv / gost*.fits / gaia_scanlaw.csv in the folder (a GOST export, a
    gaiamock healpix file, or what an earlier run saved); Gaia's DR3 commanded scan law through
    gaiaunlimited (saved as gaia_scanlaw.csv); else a synthetic stand-in sized from Gaia's number
    of good along-scan observations."""
    folder = Path(folder)
    candidates = [Path(explicit)] if explicit else (
        sorted(folder.glob("gost*.csv")) + sorted(folder.glob("gost*.fits"))
        + sorted(folder.glob("healpix_64_*.fits")) + sorted(folder.glob("gaia_scanlaw.csv")))
    for path in candidates:
        if path.exists():
            law = load_scan_law(path)
            if np.isnan(law["plx"]).any():
                law["plx"] = parallax_factor_al(law["t"], law["psi"], ra, dec)
            say(f"  Gaia scan law: {len(law['t'])} DR3 transits from {path.name}")
            return law
    if fetch:
        try:
            law = scan_law_from_gaiaunlimited(ra, dec)
            save_scan_law(law, folder / "gaia_scanlaw.csv")
            say(f"  Gaia scan law: {len(law['t'])} DR3 transits from Gaia's commanded scan law "
                "(saved as gaia_scanlaw.csv)")
            return law
        except ImportError:
            say("  (pip install gaiaunlimited to get Gaia's real scan law automatically)")
        except Exception as exc:
            say(f"  could not get Gaia's scan law through gaiaunlimited: {str(exc)[:120]}")
    n_tr = int(round(n_obs_al / CCDS_PER_TRANSIT)) if n_obs_al else 40
    say(f"  Gaia scan law: none available; using {n_tr} synthetic transits, so RUWE predictions are "
        f"approximate. (pip install gaiaunlimited, or save GOST's CSV export as {folder / 'gost.csv'})")
    return synthetic_scan_law(ra, dec, n_tr)


def tess_sectors(ra, dec, tic=0):
    """Every TESS sector (number, mid-time BJD) whose cameras cover the star, from tess-point's
    pointing tables. Empty if tess-point is not installed."""
    try:
        from tess_stars2px import TESS_Spacecraft_Pointing_Data, tess_stars2px_function_entry
    except ImportError:
        return []
    out = tess_stars2px_function_entry(int(tic), float(ra), float(dec))
    secs = sorted({int(s) for s in out[3] if int(s) > 0})
    pointing = TESS_Spacecraft_Pointing_Data()
    mids = dict(zip([int(s) for s in pointing.sectors], np.asarray(pointing.midtimes, float)))
    return [(s, mids[s]) for s in secs if s in mids and mids[s] > 2400000]


# ----------------------------------------------------------------------------
# Comparison stars and diagnostics (need the Gaia archive)
# ----------------------------------------------------------------------------
DEFAULT_CALIBRATION = {"median_ruwe": 1.0, "core_sigma": 0.05, "f_outlier": 0.05,
                       "n_stars": 0, "source": "defaults (no gaia_calibration.json)"}


def calibrate(ra, dec, g, bp_rp, ruwe_obs=None, radius_deg=3.0, dg=0.5, dcol=0.2):
    """RUWE distribution of comparison stars near the target in the sky, magnitude and colour."""
    from astroquery.gaia import Gaia
    q = f"""SELECT ruwe, astrometric_excess_noise FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_deg}))
              AND phot_g_mean_mag BETWEEN {g - dg} AND {g + dg}
              AND bp_rp BETWEEN {bp_rp - dcol} AND {bp_rp + dcol}
              AND parallax_over_error > 10 AND ruwe IS NOT NULL"""
    tab = Gaia.launch_job_async(q).get_results().to_pandas()
    r = tab["ruwe"].to_numpy(float)
    if len(r) < 30:
        raise RuntimeError(f"only {len(r)} comparison stars; widen the cone")
    med = float(np.median(r))
    low = r[r <= med]
    core = float(1.4826 * np.median(med - low))                 # spread of the well-behaved side
    out = {"median_ruwe": med, "core_sigma": max(core, 0.01),
           "f_outlier": float(np.mean(r > 1.4)), "n_stars": int(len(r)),
           "aen_median": float(np.median(tab["astrometric_excess_noise"])),
           "source": f"Gaia DR3, {len(r)} stars within {radius_deg} deg, G {g:.2f}+/-{dg}, "
                     f"BP-RP {bp_rp:.2f}+/-{dcol}, parallax/error > 10"}
    if ruwe_obs is not None:
        out["f_at_or_above_observed"] = float(np.mean(r >= ruwe_obs))
    return out


def load_calibration(folder, say=print):
    path = Path(folder) / "gaia_calibration.json"
    if path.exists():
        cal = json.loads(path.read_text())
        say(f"  RUWE of comparison stars: median {cal['median_ruwe']:.3f}, core spread "
            f"{cal['core_sigma']:.3f}, {100 * cal['f_outlier']:.1f}% above 1.4 ({cal['n_stars']} stars)")
        return cal
    say("  No gaia_calibration.json: assuming comparison stars have RUWE 1.00 +/- 0.05 with 5% "
        "above 1.4. Run gaia_tools.py on the folder (needs the internet) to measure them.")
    return dict(DEFAULT_CALIBRATION)


DIAG_COLUMNS = ("source_id, ra, dec, phot_g_mean_mag, bp_rp, parallax, ruwe, astrometric_excess_noise, "
                "astrometric_excess_noise_sig, astrometric_n_good_obs_al, astrometric_n_obs_al, "
                "visibility_periods_used, ipd_frac_multi_peak, ipd_gof_harmonic_amplitude, "
                "ipd_frac_odd_win, non_single_star, duplicated_source, radial_velocity, "
                "radial_velocity_error, rv_nb_transits, rv_amplitude_robust, rv_chisq_pvalue, "
                "rv_renormalised_gof, rv_method_used, rv_template_teff, grvs_mag")


def diagnostics(ra, dec, radius_arcsec=5.0):
    """Gaia DR3 quality flags for the target and every source within radius_arcsec."""
    from astroquery.gaia import Gaia
    q = f"""SELECT {DIAG_COLUMNS},
                   DISTANCE(POINT(ra, dec), POINT({ra}, {dec})) * 3600 AS sep_arcsec
            FROM gaiadr3.gaia_source
            WHERE 1 = CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_arcsec / 3600}))
            ORDER BY phot_g_mean_mag"""
    tab = Gaia.launch_job(q).get_results().to_pandas()
    rows = []
    for _, r in tab.iterrows():
        row = {}
        for k, v in r.items():
            if k == "source_id":
                row[k] = int(v)
            else:
                try:
                    row[k] = None if pd.isna(v) else float(v)
                except (TypeError, ValueError):
                    row[k] = str(v)
        rows.append(row)
    return {"target": rows[0] if rows else None, "neighbours": rows[1:]}


def report_diagnostics(diag, say=print):
    """Plain-language warnings from Gaia's image and velocity quality flags."""
    if not diag or not diag.get("target"):
        return []
    t, notes = diag["target"], []
    if (t.get("ipd_frac_multi_peak") or 0) > 2:
        notes.append(f"ipd_frac_multi_peak = {t['ipd_frac_multi_peak']:.0f}%: some transits saw two "
                     "peaks, as for a partly resolved pair (~0.1-1 arcsec)")
    if (t.get("ipd_gof_harmonic_amplitude") or 0) > 0.1:
        notes.append(f"ipd_gof_harmonic_amplitude = {t['ipd_gof_harmonic_amplitude']:.2f}: the image "
                     "shape depends on scan direction, as for an elongated (resolved-pair) image")
    if (t.get("duplicated_source") or 0) > 0:
        notes.append("duplicated_source is set: Gaia matched this source twice")
    if (t.get("non_single_star") or 0) > 0:
        notes.append(f"non_single_star = {t['non_single_star']:.0f}: Gaia DR3 has a non-single-star solution")
    for n in diag.get("neighbours", []):
        dg = (n.get("phot_g_mean_mag") or 99) - (t.get("phot_g_mean_mag") or 0)
        if n.get("sep_arcsec", 99) < 4 and dg < 6:
            notes.append(f"a neighbour {n['sep_arcsec']:.1f} arcsec away, {dg:.1f} mag fainter in G "
                         "(it can disturb the astrometry and dilute TESS and ATLAS)")
    if (t.get("rv_chisq_pvalue") is not None) and t["rv_chisq_pvalue"] < 0.01:
        notes.append(f"rv_chisq_pvalue = {t['rv_chisq_pvalue']:.2g}: Gaia's velocities vary")
    say("  Gaia quality flags: " + ("none raised" if not notes else ""))
    for n in notes:
        say(f"    - {n}")
    return notes


def load_diagnostics(folder):
    path = Path(folder) / "gaia_diagnostics.json"
    return json.loads(path.read_text()) if path.exists() else None


# ----------------------------------------------------------------------------
# RUWE model
# ----------------------------------------------------------------------------
class RuweModel:
    """Predict RUWE for a photocentre track sampled at Gaia's transit times, and the
    likelihood of the observed RUWE.

    The 5-parameter fit is linear, so with equal weights its residuals are a fixed
    projection of the along-scan displacements. Summed over CCDs, the extra chi-squared
    is lambda = n_ccd * sum(residual^2) / sigma_ccd^2, and the reduced chi-squared is
    1 + lambda / nu on average (non-central chi-squared, variance 2(nu + 2 lambda)).
    """

    def __init__(self, law, gmag, ruwe_obs, calibration=None, n_ccd=CCDS_PER_TRANSIT,
                 outlier_range=(1.0, 10.0)):
        self.t = np.asarray(law["t"], float)
        psi = np.asarray(law["psi"], float)
        self.sin, self.cos = np.sin(psi), np.cos(psi)
        ty = (self.t - DR3_REF) / 365.25
        m = np.column_stack([self.sin, ty * self.sin, self.cos, ty * self.cos, law["plx"]])
        self.q, _ = np.linalg.qr(m)
        self.n_ccd = n_ccd
        self.nu = n_ccd * len(self.t) - 5
        self.sigma = al_sigma_ccd(gmag)
        self.obs = float(ruwe_obs)
        cal = dict(DEFAULT_CALIBRATION, **(calibration or {}))
        self.s = float(cal["median_ruwe"])
        shot0 = self.s * np.sqrt(2.0 / self.nu) / 2.0               # shot-noise spread of RUWE at lambda = 0
        self.core = float(np.sqrt(max(cal["core_sigma"] ** 2 - shot0**2, 0.0)))
        self.f_out = float(np.clip(cal["f_outlier"], 1e-4, 0.5))
        lo, hi = outlier_range
        self.out_density = (1.0 / (self.obs * np.log(hi / lo))) if lo <= self.obs <= hi else 1e-6
        self.synthetic = bool(law.get("synthetic"))

    def along_scan(self, dra, ddec):
        """Along-scan displacement (mas) from offsets in RA* and Dec (mas), [..., n_transits]."""
        return dra * self.sin + ddec * self.cos

    def lam(self, eta):
        resid = eta - (eta @ self.q) @ self.q.T
        return self.n_ccd * np.sum(resid**2, axis=-1) / self.sigma**2

    def predict(self, eta):
        """Mean predicted RUWE and its spread, from along-scan displacements [..., n]."""
        lam = self.lam(eta)
        mean = self.s * np.sqrt(1.0 + lam / self.nu)
        shot = self.s**2 * np.sqrt(2.0 * (self.nu + 2.0 * lam)) / self.nu / (2.0 * mean)
        spread = np.sqrt(shot**2 + (self.core * mean / self.s) ** 2)
        return mean, spread, lam

    def predicted_aen(self, eta):
        """Excess noise (mas) the extra scatter corresponds to, per CCD."""
        lam = self.lam(eta)
        return self.sigma * np.sqrt(lam / self.nu)

    def loglike(self, eta):
        mean, spread, _ = self.predict(eta)
        core = np.exp(-0.5 * ((self.obs - mean) / spread) ** 2) / (np.sqrt(2 * np.pi) * spread)
        return np.log((1 - self.f_out) * core + self.f_out * self.out_density)

    def describe(self):
        return (f"{len(self.t)} transits x {self.n_ccd} CCDs, {self.sigma:.3f} mas per CCD; "
                f"single stars: RUWE {self.s:.3f} +/- {self.core:.3f} (plus shot noise), "
                f"{100 * self.f_out:.1f}% outliers" + ("; SYNTHETIC scan law" if self.synthetic else ""))


def wobble_track(t, period, ecc, omega, tp, inc, big_omega, a0_mas):
    """Photocentre offsets (RA*, Dec; mas) of a Keplerian photocentre orbit with semi-major
    axis a0_mas, using the Thiele-Innes form (as gaiamock does)."""
    mean = 2 * np.pi * (np.asarray(t) - tp) / period
    ecc_anom = mean.copy()
    for _ in range(30):
        ecc_anom = ecc_anom - (ecc_anom - ecc * np.sin(ecc_anom) - mean) / (1 - ecc * np.cos(ecc_anom))
    x, y = np.cos(ecc_anom) - ecc, np.sqrt(1 - ecc**2) * np.sin(ecc_anom)
    cw, sw, co, so, ci = np.cos(omega), np.sin(omega), np.cos(big_omega), np.sin(big_omega), np.cos(inc)
    A = a0_mas * (cw * co - sw * so * ci)
    B = a0_mas * (cw * so + sw * co * ci)
    F = -a0_mas * (sw * co + cw * so * ci)
    G = -a0_mas * (sw * so - cw * co * ci)
    return B * x + G * y, A * x + F * y                         # (RA*, Dec)


# ----------------------------------------------------------------------------
# Gaia radial velocities
# ----------------------------------------------------------------------------
class GaiaRvModel:
    """Gaia DR3's median velocity and robust amplitude, predicted from per-transit velocities.

    Each transit's measured velocity is the light-weighted blend of the stars' lines while the
    eclipsing pair's lines overlap, moving smoothly to the brighter star's own velocity once they
    separate by more than `resolve` km/s (logistic, width `width`). A third star's lines, if any,
    stay in the blend.
    """

    def __init__(self, law, gaia, median_err=3.0, amp_err=6.0, epoch_err=3.0, resolve=35.0,
                 width=8.0):
        n_rv = int(gaia.get("rv_nb_transits") or 0)
        t = np.asarray(law["t"], float)
        if n_rv and len(t) > n_rv:                                # RVS sees most but not all transits
            t = t[np.unique(np.linspace(0, len(t) - 1, n_rv).round().astype(int))]
        self.t = t
        self.n = len(t)
        # DR3 used two methods. For bright stars (G_RVS <= 12, rv_method_used = 1) the velocity is
        # the median of the per-transit velocities, and their robust range is published. For fainter
        # stars (rv_method_used = 2) it comes from one cross-correlation of the combined spectrum,
        # which follows the mean of the transits, and there is no amplitude.
        method = gaia.get("rv_method_used")
        if method is None:
            method = 1 if gaia.get("rv_amplitude_robust") is not None else (
                2 if (gaia.get("grvs_mag") or 0) > 12 else 1)
        self.method = int(method)
        self.stat = "median" if self.method == 1 else "mean"
        self.median_obs = gaia.get("radial_velocity")
        self.median_err = float(np.hypot(median_err, gaia.get("radial_velocity_error") or 0.0))
        self.amp_obs = gaia.get("rv_amplitude_robust") if self.method == 1 else None
        # range of n pure-noise epochs: about 2 * Phi^-1(1 - 0.5 / n) sigma
        from scipy.stats import norm
        self.noise_range = 2.0 * float(norm.ppf(1 - 0.5 / max(n_rv or self.n, 2))) * epoch_err
        self.amp_err = float(amp_err)
        self.resolve, self.width = float(resolve), float(width)
        self.synthetic = bool(law.get("synthetic"))

    def measured(self, v1, v2, l1, l2, v3=None, l3=0.0):
        """Per-transit velocity Gaia would measure. v1, v2 (and v3): [..., n_transits] km/s;
        l1, l2, l3: the stars' shares of the light in the RVS band (broadcastable)."""
        l1, l2 = np.asarray(l1, float)[..., None], np.asarray(l2, float)[..., None]
        l3 = np.asarray(l3, float)[..., None]
        blend = l1 * v1 + l2 * v2 + (l3 * v3 if v3 is not None else 0.0)
        blend = blend / (l1 + l2 + (l3 if v3 is not None else 0.0))
        dom = np.where(l1 >= l2, v1, v2)
        if v3 is not None:                                      # the third star's lines stay blended in
            dom = (np.maximum(l1, l2) * dom + l3 * v3) / (np.maximum(l1, l2) + l3)
        w = 1.0 / (1.0 + np.exp(np.clip(-(np.abs(v1 - v2) - self.resolve) / self.width, -50, 50)))
        return (1 - w) * blend + w * dom

    def median_only(self):
        """Use only Gaia's median (or combined) velocity, modelled as the light-weighted blend of the
        two stars, and ignore the robust amplitude. For a double-lined pair of comparable brightness
        the per-transit velocities depend on how Gaia's pipeline handled two similar sets of calcium-
        triplet lines (broad wings, so they stay merged to large separations) and on which transits
        it rejected before computing rv_amplitude_robust; neither can be modelled reliably. The median
        lands near the system velocity whether Gaia measured a blend or picked a star at random."""
        self.amp_obs = None
        self.resolve = 1e9
        self.mode = "median only (double-lined)"
        return self

    def summarise(self, g, axis=0):
        """Gaia's summary statistic (median or mean) of per-transit values along `axis`."""
        return np.median(g, axis=axis) if self.stat == "median" else np.mean(g, axis=axis)

    def describe(self):
        if getattr(self, "mode", None):
            return (f"{self.stat} of {self.n} transits {self.median_obs:+.1f} +/- {self.median_err:.1f} km/s, used as "
                    "the blend of both stars' lines; robust amplitude not used (double-lined pair)")
        if self.method == 1:
            return (f"median of {self.n} transits {self.median_obs:+.1f} +/- {self.median_err:.1f} km/s"
                    + (f", robust amplitude {self.amp_obs:.0f} +/- {self.amp_err:.0f} km/s"
                       if self.amp_obs is not None else ""))
        return (f"from the combined spectrum of {self.n} transits (modelled as their mean) "
                f"{self.median_obs:+.1f} +/- {self.median_err:.1f} km/s; no amplitude for such stars")

    def predict(self, v_meas):
        med = self.summarise(v_meas, axis=-1)
        rng = np.ptp(v_meas, axis=-1)
        amp = np.sqrt(rng**2 + self.noise_range**2)
        return med, amp

    def loglike(self, v_meas):
        med, amp = self.predict(v_meas)
        ll = 0.0
        if self.median_obs is not None:
            ll = ll - 0.5 * ((self.median_obs - med) / self.median_err) ** 2
        if self.amp_obs is not None:
            ll = ll - 0.5 * ((self.amp_obs - amp) / self.amp_err) ** 2
        return ll


# ----------------------------------------------------------------------------
# Command line: fetch the comparison stars and diagnostics for a folder
# ----------------------------------------------------------------------------
def main():
    import argparse
    p = argparse.ArgumentParser(description="Fetch Gaia comparison stars, quality flags and scan law for a star "
                                            "(joint_orbit.py does this itself; this is for running it separately).")
    p.add_argument("folder", help="the star's output folder (with star.json and gaia.json, as joint_orbit.py writes)")
    p.add_argument("--radius", type=float, default=3.0, help="comparison-star cone radius, degrees")
    args = p.parse_args()
    folder = Path(args.folder)
    star = json.loads((folder / "star.json").read_text(encoding="utf-8-sig"))
    gaia = json.loads((folder / "gaia.json").read_text())
    diag = diagnostics(star["ra"], star["dec"])
    (folder / "gaia_diagnostics.json").write_text(json.dumps(diag, indent=1))
    t = diag["target"]
    print(f"Target: G {t['phot_g_mean_mag']:.2f}, BP-RP {t['bp_rp']:.2f}, RUWE {t['ruwe']:.2f}, "
          f"{t['astrometric_n_good_obs_al']:.0f} good along-scan observations")
    report_diagnostics(diag)
    cal = calibrate(star["ra"], star["dec"], t["phot_g_mean_mag"], t["bp_rp"], ruwe_obs=t["ruwe"],
                    radius_deg=args.radius)
    (folder / "gaia_calibration.json").write_text(json.dumps(cal, indent=1))
    print(f"Comparison stars ({cal['n_stars']}): median RUWE {cal['median_ruwe']:.3f}, core spread "
          f"{cal['core_sigma']:.3f}; {100 * cal['f_outlier']:.1f}% above 1.4, "
          f"{100 * cal.get('f_at_or_above_observed', np.nan):.1f}% at or above the target's {t['ruwe']:.2f}")
    gaia.update({k: t[k] for k in ("astrometric_n_good_obs_al", "rv_amplitude_robust", "rv_nb_transits",
                                   "radial_velocity", "radial_velocity_error", "bp_rp", "rv_method_used",
                                   "grvs_mag")
                 if t.get(k) is not None})
    (folder / "gaia.json").write_text(json.dumps(gaia, indent=1))
    scan_law_for(folder, star["ra"], star["dec"], t.get("astrometric_n_good_obs_al"))


if __name__ == "__main__":
    main()
