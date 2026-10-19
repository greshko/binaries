#!/usr/bin/env python
"""
lamost_rv.py - velocities and exact times from a LAMOST medium-resolution (MRS) spectrum,
including both stars' velocities when the spectrum is double-lined.

For each exposure and each arm (blue 4950-5350 A, red 6300-6800 A) the spectrum is
continuum-normalised and cross-correlated with a line mask (ESPRESSO G2/F9 masks, as shipped
with the iCCF package; air wavelengths converted to vacuum, since LAMOST MRS wavelengths are
vacuum and already heliocentric). The cross-correlation function (CCF) is fitted with one
Gaussian, and with two; the two-Gaussian fit gives both stars' velocities and the ratio of
their line strengths (about their flux ratio, for stars of similar temperature).

Times: DATE-OBS is the UTC mid-point of each exposure (DATE-BEG/END are Beijing time, UTC+8).
The mid-time of each exposure is converted to BJD_TDB.

Usage:
    python lamost_rv.py spectrum.fits.gz [--sb2] [--mask G2] [--plot out.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares

C_KMS = 299792.458


def air_to_vac(lam):
    """Air to vacuum wavelengths (Angstrom), Morton (2000) / Ciddor."""
    s2 = (1e4 / lam) ** 2
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return lam * n


def load_mask(name):
    import zipfile
    import io
    whl = sorted(Path(__file__).resolve().parent.glob("pkgs/iccf-*.whl"))
    if whl:
        data = zipfile.ZipFile(whl[0]).read(f"iCCF/data/ESPRESSO_{name}.fits")
        d = fits.open(io.BytesIO(data))[1].data
    else:
        import iCCF
        d = fits.open(Path(iCCF.__file__).parent / "data" / f"ESPRESSO_{name}.fits")[1].data
    return air_to_vac(np.asarray(d["lambda"], float)), np.asarray(d["contrast"], float)


def normalise(w, f, ivar, width=12.0, pct=85, iters=3):
    """Continuum by an upper running percentile, smoothed; then a clipped low-order polynomial."""
    good = (ivar > 0) & np.isfinite(f)
    cont = np.full_like(f, np.nan)
    for i in range(len(w)):
        sel = good & (np.abs(w - w[i]) < width / 2)
        if sel.sum() > 10:
            cont[i] = np.percentile(f[sel], pct)
    ok = good & np.isfinite(cont)
    x = (w - w.mean()) / np.ptp(w)
    use = ok.copy()
    for _ in range(iters):
        c = np.polyfit(x[use], f[use] / cont[use] * cont[use], 6) if False else np.polyfit(x[use], cont[use], 6)
        model = np.polyval(c, x)
        r = f / model
        use = ok & (r > 0.97) & (r < 1.05) | (ok & (np.abs(cont - model) / model < 0.02))
    model = np.polyval(np.polyfit(x[ok], cont[ok], 6), x)
    fn = f / model
    en = np.where(ivar > 0, 1 / np.sqrt(np.maximum(ivar, 1e-30)) / model, np.inf)
    return fn, en, good


def ccf(w, fn, en, good, mlam, mcon, vgrid, sigma_kms=10.0):
    """Weighted cross-correlation of line depth (1 - fn) with a Gaussian-line mask template."""
    depth = 1.0 - fn
    wt = np.where(good & np.isfinite(en) & (en < 1), 1.0 / en**2, 0.0)
    inside = (mlam > w[0] + 5) & (mlam < w[-1] - 5)
    ml, mc = mlam[inside], mcon[inside]
    out = np.zeros(len(vgrid))
    for k, v in enumerate(vgrid):
        centres = ml * (1 + v / C_KMS)
        sig = centres * sigma_kms / C_KMS
        # template = sum of Gaussians, evaluated only near each line
        tmpl = np.zeros_like(w)
        lo = np.searchsorted(w, centres - 4 * sig)
        hi = np.searchsorted(w, centres + 4 * sig)
        for c0, s0, a0, l0, h0 in zip(centres, sig, mc, lo, hi):
            tmpl[l0:h0] += a0 * np.exp(-0.5 * ((w[l0:h0] - c0) / s0) ** 2)
        out[k] = np.sum(wt * depth * tmpl) / np.sqrt(np.sum(wt * tmpl**2) + 1e-30)
    out = out / np.sqrt(np.sum(wt * depth**2))
    return out, inside.sum()


def gauss(v, a, mu, s):
    return a * np.exp(-0.5 * ((v - mu) / s) ** 2)


def fit_ccf(v, c, n_peaks, guesses=None):
    """Fit 1 or 2 Gaussians plus a straight-line background. Returns params, errors, chi2, model."""
    bg0 = np.median(np.r_[c[:30], c[-30:]])
    if guesses is None:
        i = np.argmax(c)
        guesses = [v[i]] if n_peaks == 1 else [v[i], v[i] + 150]
    p0, lo, hi = [bg0, 0.0], [-np.inf, -np.inf], [np.inf, np.inf]
    for g in guesses:
        p0 += [max(np.interp(g, v, c) - bg0, 1e-3), g, 15.0]
        lo += [0.0, v[0], 5.0]
        hi += [np.inf, v[-1], 60.0]

    def model(p):
        m = p[0] + p[1] * (v - v.mean()) / np.ptp(v)
        for j in range(n_peaks):
            m = m + gauss(v, *p[2 + 3 * j:5 + 3 * j])
        return m

    res = least_squares(lambda p: model(p) - c, p0, bounds=(lo, hi))
    r = res.fun
    dof = max(len(v) - len(p0), 1)
    s2 = np.sum(r**2) / dof
    try:
        cov = np.linalg.inv(res.jac.T @ res.jac) * s2
        err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = np.full(len(p0), np.nan)
    return res.x, err, float(np.sum(r**2)), model(res.x)


def bjd_tdb(mjd_utc, ra, dec):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    import astropy.units as u
    loc = EarthLocation(lon=117.58 * u.deg, lat=40.39 * u.deg, height=960 * u.m)   # Xinglong
    t = Time(mjd_utc, format="mjd", scale="utc", location=loc)
    ltt = t.light_travel_time(SkyCoord(ra * u.deg, dec * u.deg), kind="barycentric")
    return (t.tdb + ltt).jd


def exposures(path):
    h = fits.open(path)
    hd = h[0].header
    out = []
    for e in h[1:]:
        if e.data is None or not e.name[:2] in ("B-", "R-"):
            continue
        d = e.data
        mid = e.header["DATE-OBS"]                                  # UTC mid-exposure
        from astropy.time import Time
        out.append({"arm": e.name[0], "lmjm": e.header.get("LMJM"), "snr": e.header.get("SNR"),
                    "mid_mjd": Time(mid, format="isot", scale="utc").mjd,
                    "w": np.asarray(d["WAVELENGTH"][0], float), "f": np.asarray(d["FLUX"][0], float),
                    "ivar": np.asarray(d["IVAR"][0], float)})
    return hd, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("spectrum")
    p.add_argument("--sb2", action="store_true", help="fit two stars' lines")
    p.add_argument("--mask", default="G2")
    p.add_argument("--coadd", action="store_true", help="also measure the exposures stacked per arm")
    p.add_argument("--plot", default=None)
    args = p.parse_args()

    hd, exps = exposures(args.spectrum)
    mlam, mcon = load_mask(args.mask)
    vgrid = np.arange(-450.0, 350.0, 1.0)
    print(f"{hd['OBJNAME']} (Gaia DR3), obsid {hd['OBSID']}, {hd['DATE-OBS']} UTC mid, "
          f"HELIO={hd['HELIO']} (corr {hd['HELIO_RV']:+.2f} km/s), VACUUM={hd.get('VACUUM')}")
    rows, ccfs = [], []
    groups = {}
    for ex in exps:
        groups.setdefault(ex["arm"], []).append(ex)
    sets = [(f"{ex['arm']}-{ex['lmjm']}", [ex]) for ex in exps]
    if args.coadd:
        sets += [(f"{arm}-stack", g) for arm, g in groups.items()]
    for label, members in sets:
        cc_sum, wsum, nl = 0.0, 0.0, 0
        for ex in members:
            fn, en, good = normalise(ex["w"], ex["f"], ex["ivar"])
            c, nl = ccf(ex["w"], fn, en, good, mlam, mcon, vgrid)
            wgt = (ex["snr"] or 1.0) ** 2
            cc_sum, wsum = cc_sum + wgt * c, wsum + wgt
        c = cc_sum / wsum
        mid = float(np.average([m["mid_mjd"] for m in members], weights=[(m["snr"] or 1) ** 2 for m in members]))
        p1, e1, chi1, m1 = fit_ccf(vgrid, c, 1)
        row = {"set": label, "n_lines": nl, "snr": np.mean([m["snr"] for m in members]), "mid_mjd": mid,
               "v1": p1[3], "v1_err": e1[3], "chi2_1": chi1}
        if args.sb2:
            # second peak: strongest residual away from the first
            resid = c - m1
            far = np.abs(vgrid - p1[3]) > 60
            g2 = vgrid[far][np.argmax(resid[far])]
            p2, e2, chi2, m2 = fit_ccf(vgrid, c, 2, guesses=[p1[3], g2])
            a = [(p2[2], p2[3], p2[4], e2[3]), (p2[5], p2[6], p2[7], e2[6])]
            a.sort(key=lambda t: -t[0] * t[2])                        # stronger (larger area) first
            row.update({"vA": a[0][1], "vA_err": a[0][3], "vB": a[1][1], "vB_err": a[1][3],
                        "area_ratio_BA": (a[1][0] * a[1][2]) / (a[0][0] * a[0][2]),
                        "sigA": a[0][2], "sigB": a[1][2], "chi2_2": chi2})
            ccfs.append((label, c, m2))
        else:
            ccfs.append((label, c, m1))
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    df["bjd_tdb"] = bjd_tdb(df.mid_mjd.to_numpy(), hd["RA"], hd["DEC"])
    pd.set_option("display.width", 200)
    print(df.drop(columns=["mid_mjd"]).round(3).to_string(index=False))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, len(ccfs), figsize=(4 * len(ccfs), 3.2), sharey=False)
        for ax, (label, c, m) in zip(np.atleast_1d(axs), ccfs):
            ax.plot(vgrid, c, color="0.3", lw=1)
            ax.plot(vgrid, m, color="crimson", lw=1)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("heliocentric velocity (km/s)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=110)
    df.to_csv(Path(args.spectrum).with_suffix("").with_suffix(".rv.csv"), index=False)


if __name__ == "__main__":
    main()
