#!/usr/bin/env python
"""
lamost_double_lines.py - measure the velocities in a LAMOST medium-resolution
spectrum, and find out whether it shows one star or two.

If the companion in a binary is bright enough, its spectral lines appear as a
second set, shifted from the first by the difference in the two stars'
velocities. This script compares each exposure with a model stellar spectrum,
looks for one or two sets of lines, and measures the velocity of each and the
share of the light each contributes.

Usage
-----
With a spectrum you have already downloaded:

    python lamost_double_lines.py lamost_mrs_726616157.fits.gz

Or let it download the spectrum by its LAMOST number (ObsID):

    python lamost_double_lines.py --obsid 726616157

Useful options:

    --teff 6100      the star's temperature, if known (default: found from the spectrum)
    --gamma -53      the system's own velocity in km/s, if known; with two stars
                     this gives the ratio of their masses
    --outdir FOLDER  where to put the results (default: next to the spectrum)

The first run downloads a set of model spectra (about 46 MB, from the Python
package index) into a folder called spectrum_models next to this script.

Output
------
    <name>_double_lines.txt   everything printed to the screen
    <name>_velocities.csv     the measurements for each exposure and each arm
    <name>_double_lines.png   the comparison with the model, and a stretch of
                              spectrum with the fitted model drawn over it
    <name>_rv_epochs.csv      the same velocities laid out for joint_orbit.py:
                              copy the rows into its rv_epochs.csv

Method
------
1. Each exposure is read with its normalised brightness, and its wavelengths
   are converted from vacuum to air, the convention the model spectra use.
2. A model spectrum is slid along in velocity and compared with the data at
   each step (a cross-correlation). One star gives one peak. The same
   comparison of the model with itself shows what one star alone would give,
   so a second peak can be told from the model's own side bumps.
3. A one-star model and a two-star model are then fitted directly to the
   spectrum. The two-star model has two velocities and the share of the
   light that comes from the second star. Metal content and line width are
   chosen from a small grid.
4. The spectrum is called double-lined only if the two-star model fits much
   better, the second star carries a real share of the light, the two
   velocities are far enough apart for the spectrograph to separate, and the
   second peak stands clear of what one star alone would give.

What it cannot do
-----------------
* It cannot separate two stars whose velocities differ by less than about
  50 km/s (the spectrograph blurs lines by about 40 km/s), and how faint a
  companion it can see depends on how noisy the spectrum is. For every
  single-lined result the script therefore tests this directly: it adds a
  second star to copies of the spectrum and reports how often it finds it.
  A single-lined result does not mean a single star.
* For faint stars (signal-to-noise under 15 per exposure) the exposures of
  one visit are averaged before anything is measured, so the visit gives one
  velocity. LAMOST sometimes leaves its normalised spectrum empty for such
  stars; the script then normalises the raw spectrum itself.
* The share of the light depends on the temperatures assumed for the two
  stars. Both are given the same temperature; if they differ by several
  hundred degrees the true share can be off by 0.1.
* The model spectra are approximate. The quoted errors are widened to allow
  for the misfit, and velocities from the two arms typically differ by 1 to
  3 km/s; treat 2 to 3 km/s as the real accuracy.
* The hydrogen line at 6563 Angstrom is left out of the fit because the
  model reproduces it poorly.
* Times are the mid-exposure times in the file's header (UTC), not corrected
  for light travel time across the Earth's orbit (at most 8 minutes).
"""

import argparse
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.optimize import least_squares

C = 299792.458                     # speed of light, km/s
RESOLUTION = 7500.0                # LAMOST medium-resolution spectra
MODEL_PACKAGE = "thedoppler==1.1.32"
MODEL_FILE = "doppler/data/cannongrid_3000_18000_coolstars_norm_cubic_model.pkl"
ARMS = {"B": (4960.0, 5340.0), "R": (6300.0, 6800.0)}
HYDROGEN = 6562.8                  # left out, with this much either side:
HYDROGEN_HALF_WIDTH = 14.0
MIN_SEPARATION = 45.0              # km/s; closer than this the two stars cannot be told apart
FAINT = 15.0                       # below this signal-to-noise, a night's exposures are averaged first
TEST_SHARES = (0.25, 0.40)         # for the "what could this spectrum have shown" test:
TEST_SEPARATIONS = (60.0, 100.0, 150.0)   # the second star's share of the light, and km/s apart
TEST_TRIALS, TEST_BLANKS = 4, 16
INK, INK2, GRID, BLUE, ORANGE, SURFACE = "#0b0b0b", "#52514e", "#dddcd6", "#2a78d6", "#eb6834", "#fcfcfb"


class Tee:
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)


# ----------------------------------------------------------------------------
# Getting the data
# ----------------------------------------------------------------------------
def download_spectrum(obsid, folder):
    """Fetch a medium-resolution spectrum from the LAMOST archive by its number."""
    import requests

    for release in ("dr11", "dr10", "dr9", "dr8", "dr7"):
        url = f"https://www.lamost.org/{release}/v2.0/medspectrum/fits/{obsid}"
        try:
            reply = requests.get(url, timeout=120)
        except Exception:
            continue
        if reply.ok and len(reply.content) > 20000 and "html" not in reply.headers.get("Content-Type", ""):
            zipped = reply.content[:2] == b"\x1f\x8b"
            path = Path(folder) / (f"lamost_mrs_{obsid}.fits" + (".gz" if zipped else ""))
            path.write_bytes(reply.content)
            return path
    sys.exit(f"Could not download spectrum {obsid}. Search for it by ObsID at "
             "https://www.lamost.org/dr11/medcas/search and download the FITS file by hand.")


def vacuum_to_air(wave):
    s2 = (1e4 / wave) ** 2
    return wave / (1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2))


def normalise(wave, flux):
    """Divide by a smooth curve through the upper envelope (used only if the file has none)."""
    width = max(int(15.0 / np.median(np.diff(wave))), 21)
    top = percentile_filter(flux, 90, size=width, mode="nearest")
    return flux / np.maximum(gaussian_filter1d(top, width / 2), 1e-6)


def read_spectrum(path, say):
    """All usable spectra in one LAMOST medium-resolution file, as a list of dicts."""
    hdus = fits.open(path)
    head = hdus[0].header
    found, own_norm = [], 0
    for hdu in hdus[1:]:
        data = hdu.data
        if data is None or not hasattr(data, "dtype") or data.dtype.names is None:
            continue
        names = [n.upper() for n in data.dtype.names]
        if "WAVELENGTH" not in names or "FLUX" not in names:
            continue
        name = str(hdu.header.get("EXTNAME", ""))
        wave = np.ravel(data["WAVELENGTH"]).astype(float)
        if head.get("VACUUM", True):
            wave = vacuum_to_air(wave)
        flux = np.ravel(data["FLUX"]).astype(float)
        ivar = np.ravel(data["IVAR"]).astype(float) if "IVAR" in names else np.ones_like(flux)
        mask = np.zeros(len(flux), bool)
        for column in ("PIXMASK", "ANDMASK"):
            if column in names:
                mask |= np.ravel(data[column]) != 0
        norm = np.ravel(data["NORMALIZATION"]).astype(float) if "NORMALIZATION" in names else None
        # LAMOST fills the column with -9999 when its own normalisation failed (common for faint stars)
        if norm is None or np.mean(np.isfinite(norm) & (norm > 0.05) & (norm < 5)) < 0.8:
            own_norm += 1
            filled = np.where(np.isfinite(flux) & (ivar > 0), flux, np.nanmedian(flux))
            norm = normalise(wave, filled)
        arm = "B" if np.median(wave) < 5800 else "R"
        when = hdu.header.get("DATE-OBS") or head.get("DATE-OBS")
        err = norm / np.maximum(np.abs(flux) * np.sqrt(np.maximum(ivar, 1e-30)), 1e-6)   # error of the normalised flux
        ok = ~mask & (ivar > 0) & np.isfinite(norm) & np.isfinite(err) & (flux > 0)
        found.append({"name": name, "arm": arm, "coadd": name.upper().startswith("COADD"), "wave": wave,
                      "norm": norm, "err": np.clip(err, 1e-4, 1.0), "ok": ok,
                      "jd": Time(str(when).strip(), format="isot", scale="utc").jd if when else np.nan,
                      "utc": str(when).strip() if when else "", "snr": float(hdu.header.get("SNR", np.nan))})
    singles = [s for s in found if not s["coadd"]]
    spectra = singles if singles else found           # single exposures keep the time information
    if not spectra:
        sys.exit(f"{path}: no medium-resolution spectra found. (Low-resolution LAMOST files cannot "
                 "separate two stars and are not handled.)")
    if own_norm:
        say("  The file has no usable normalised spectrum (LAMOST's own step failed or was skipped),")
        say("  so the script normalised the raw spectrum itself.")
    if not head.get("HELIO", True):
        say("  Warning: the header says velocities are not corrected to the Sun's frame.")
    info = {"obsid": head.get("OBSID"), "desig": head.get("DESIG", ""), "ra": head.get("RA"), "dec": head.get("DEC")}
    return spectra, info


# ----------------------------------------------------------------------------
# Model spectra
# ----------------------------------------------------------------------------
def load_model(say):
    """The model-spectrum grid, downloaded once from the Python package index."""
    folder = Path(__file__).resolve().parent / "spectrum_models"
    target = folder / Path(MODEL_FILE).name
    if not target.exists():
        folder.mkdir(exist_ok=True)
        say("Downloading model spectra (about 46 MB, once) ...")
        done = subprocess.run([sys.executable, "-m", "pip", "download", MODEL_PACKAGE, "--no-deps", "-q",
                               "-d", str(folder)], capture_output=True, text=True)
        wheels = sorted(folder.glob("thedoppler-*.whl"))
        if not wheels:
            sys.exit("Could not download the model spectra:\n" + done.stderr[-500:])
        with zipfile.ZipFile(wheels[-1]) as archive:
            target.write_bytes(archive.read(MODEL_FILE))
        wheels[-1].unlink()
    with open(target, "rb") as handle:
        model = pickle.load(handle, encoding="latin1")
    labels = np.array(model["training_set_labels"], float)
    return {"wave": np.array(model["dispersion"], float), "theta": np.array(model["theta"], float),
            "terms": model["vectorizer"][1]["terms"],
            "centre": np.percentile(labels, 50, axis=0),
            "scale": np.ptp(np.percentile(labels, [2.5, 97.5], axis=0), axis=0)}


class Templates:
    """Model spectra at the spectrograph's sharpness, made on demand and remembered."""

    def __init__(self, model):
        keep = (model["wave"] > 4850) & (model["wave"] < 6900)
        self.model, self.keep = model, keep
        wave = model["wave"][keep]
        self.log_wave = np.linspace(np.log(wave[0]), np.log(wave[-1]), 2 * keep.sum())
        self.native = np.log(wave)
        self.made = {}

    def get(self, teff, feh, width, logg=4.2):
        key = (teff, feh, width, logg)
        if key not in self.made:
            x = (np.array([teff, logg, feh], float) - self.model["centre"]) / self.model["scale"]
            vector = np.array([1.0] + [np.prod([x[i] ** p for i, p in term]) for term in self.model["terms"]])
            flux = np.interp(self.log_wave, self.native, self.model["theta"][self.keep] @ vector)
            sigma = np.hypot(C / RESOLUTION / 2.3548, 0.6 * width)       # instrument, plus extra line width
            self.made[key] = gaussian_filter1d(flux, sigma / C / (self.log_wave[1] - self.log_wave[0]))
        return self.made[key]

    def at(self, teff, feh, width, wave, velocity):
        """The model as seen at observed wavelengths for a star moving at `velocity` km/s."""
        return np.interp(np.log(wave) - np.log1p(velocity / C), self.log_wave, self.get(teff, feh, width))


# ----------------------------------------------------------------------------
# Measuring
# ----------------------------------------------------------------------------
def region_mask(spec):
    lo, hi = ARMS[spec["arm"]]
    sel = spec["ok"] & (spec["wave"] > lo) & (spec["wave"] < hi)
    if spec["arm"] == "R":
        sel &= np.abs(spec["wave"] - HYDROGEN) > HYDROGEN_HALF_WIDTH
    return sel


def fitting_region(spec):
    sel = region_mask(spec)
    return spec["wave"][sel], spec["norm"][sel], spec["err"][sel]


def stack_exposures(spectra):
    """Average the exposures of each arm into one spectrum (for faint stars)."""
    out = []
    for arm in ("B", "R"):
        group = [s for s in spectra if s["arm"] == arm]
        if not group:
            continue
        wave = group[0]["wave"]
        norm, weight = [], []
        for g in group:
            level = np.median(g["norm"][g["ok"]]) if g["ok"].any() else 1.0
            same = len(g["wave"]) == len(wave) and np.allclose(g["wave"], wave, atol=0.02)
            n = g["norm"] / level if same else np.interp(wave, g["wave"], g["norm"] / level)
            e = g["err"] / level if same else np.interp(wave, g["wave"], g["err"] / level)
            ok = g["ok"] if same else np.interp(wave, g["wave"], g["ok"].astype(float)) > 0.99
            norm.append(n)
            weight.append(np.where(ok, 1.0 / e**2, 0.0))
        norm, weight = np.array(norm), np.array(weight)
        total = weight.sum(axis=0)
        jd = float(np.nanmean([g["jd"] for g in group]))
        out.append({"name": f"{arm}, {len(group)} exposures", "arm": arm, "coadd": False, "wave": wave,
                    "norm": np.where(total > 0, (weight * norm).sum(axis=0) / np.maximum(total, 1e-12), 1.0),
                    "err": np.clip(np.where(total > 0, 1.0 / np.sqrt(np.maximum(total, 1e-12)), 1.0), 1e-4, 1.0),
                    "ok": total > 0, "jd": jd,
                    "utc": Time(jd, format="jd", scale="utc").isot[:16] if np.isfinite(jd) else "",
                    "snr": float(np.sqrt(np.nansum([g["snr"] ** 2 for g in group])))})
    return out


def correlation(templates, teff, feh, width, wave, flux, velocities):
    depth = 1.0 - flux
    return np.array([np.corrcoef(depth, 1.0 - templates.at(teff, feh, width, wave, v))[0, 1] for v in velocities])


def peak(velocities, values):
    i = int(np.clip(np.argmax(values), 1, len(values) - 2))
    a, b, _ = np.polyfit(velocities[i - 1:i + 2], values[i - 1:i + 2], 2)
    return (-b / (2 * a) if a < 0 else velocities[i]), float(values[i])


def residuals(params, templates, teff, feh, width, wave, flux, err, two):
    v1, v2, share = params
    model = templates.at(teff, feh, width, wave, v1)
    if two:
        model = (1 - share) * model + share * templates.at(teff, feh, width, wave, v2)
    x = (wave - wave.mean()) / (np.ptp(wave) / 2)
    design = np.vander(x, 4) * model[:, None]                       # a gentle curve for the continuum
    coef, *_ = np.linalg.lstsq(design / err[:, None], flux / err, rcond=None)
    return (flux - design @ coef) / err


def fit_spectrum(templates, teff, wave, flux, err, v1_start, v2_start):
    """Best one-star and two-star fits over a small grid of metal content and line width."""
    best_one = best_two = None
    for feh in (-1.0, -0.5, -0.25, 0.0, 0.25):
        for width in (0.0, 10.0, 20.0, 30.0):
            args = (templates, teff, feh, width, wave, flux, err)
            one = least_squares(residuals, [v1_start, v2_start, 0.0], args=args + (False,), x_scale=[1, 1, 0.05])
            chi_one = float((one.fun**2).sum())
            if best_one is None or chi_one < best_one["chi2"]:
                best_one = {"chi2": chi_one, "v": float(one.x[0]), "feh": feh, "width": width}
            lo = [v1_start - 60, v2_start - 60, 0.02]
            hi = [v1_start + 60, v2_start + 60, 0.98]
            two = least_squares(residuals, [v1_start, v2_start, 0.4], args=args + (True,), x_scale=[1, 1, 0.05],
                                bounds=(lo, hi))
            chi_two = float((two.fun**2).sum())
            if best_two is None or chi_two < best_two["chi2"]:
                dof = max(len(wave) - 7, 1)
                cov = np.linalg.pinv(two.jac.T @ two.jac) * chi_two / dof      # widened for the misfit
                best_two = {"chi2": chi_two, "v1": float(two.x[0]), "v2": float(two.x[1]),
                            "share": float(two.x[2]), "errors": np.sqrt(np.diag(cov)), "feh": feh, "width": width}
    return best_one, best_two


def choose_temperature(templates, spectra, say):
    """The model temperature that matches the lines best, from the blue arm."""
    blue = [s for s in spectra if s["arm"] == "B"] or spectra
    wave, flux, _ = fitting_region(blue[0])
    velocities = np.arange(-400, 400.1, 8.0)
    best = None
    for teff in np.arange(4500, 7501, 500):
        for feh in (-0.75, -0.25):
            value = correlation(templates, teff, feh, 10.0, wave, flux, velocities).max()
            if best is None or value > best[0]:
                best = (value, teff)
    say(f"  Temperature that matches the lines best: about {best[1]:.0f} K")
    return float(best[1])


def analyse(spec, templates, teff):
    """Everything for one spectrum: correlation curve, one-star and two-star fits, verdict."""
    wave, flux, err = fitting_region(spec)
    velocities = np.arange(-500, 500.1, 2.0)
    curve = correlation(templates, teff, -0.5, 10.0, wave, flux, velocities)
    v_main, height = peak(velocities, curve)
    # what one star alone would give: the model compared with itself, moved to the main peak
    grid = np.linspace(wave.min(), wave.max(), len(wave))
    alone = correlation(templates, teff, -0.5, 10.0, grid, templates.at(teff, -0.5, 10.0, grid, 0.0), velocities)
    alone = np.interp(velocities, velocities + v_main, alone) * height
    excess = np.where(np.abs(velocities - v_main) > MIN_SEPARATION, curve - alone, -np.inf)
    v_other, _ = peak(velocities, excess)
    extra = float(excess.max())
    floor = float(np.std((curve - alone)[np.abs(velocities - v_main) > 250])) if (np.abs(velocities - v_main) > 250).any() else 0.05

    one, two = fit_spectrum(templates, teff, wave, flux, err, v_main, v_other)
    dof = max(len(wave) - 7, 1)
    better = one["chi2"] / max(two["chi2"], 1e-9)
    apart = abs(two["v1"] - two["v2"])
    double = (better > 1.25 and 0.08 < two["share"] < 0.92 and apart > MIN_SEPARATION
              and extra > max(0.2 * height, 5 * floor))
    return {"velocities": velocities, "curve": curve, "alone": alone, "one": one, "two": two, "double": double,
            "gain": dof * (better - 1.0),          # how much better two stars fit, in units of the noise
            "apart": apart,
            "better": better, "extra": extra, "height": height, "chi_one": one["chi2"] / dof,
            "chi_two": two["chi2"] / dof, "wave": wave, "flux": flux}


def what_could_show(spec, result, templates, teff, seed=1):
    """Test what this spectrum could have revealed, by adding a second star to copies of it.

    The copies are built from the one-star model plus the spectrum's own leftover
    noise, shuffled in short blocks. Copies with no second star show how much the
    two-star fit improves on noise alone; a second star counts as found when the
    improvement beats every one of those by a margin and the fit puts it at the
    right separation. Returns (bar, improvements of the blank copies, {(share, separation): found}).
    """
    rng = np.random.default_rng(seed)
    sel = region_mask(spec)
    wave, flux, err = spec["wave"][sel], spec["norm"][sel], spec["err"][sel]
    feh, width, v0 = result["one"]["feh"], result["one"]["width"], result["one"]["v"]
    x = (wave - wave.mean()) / (np.ptp(wave) / 2)
    curve = np.vander(x, 4)
    design = curve * templates.at(teff, feh, width, wave, v0)[:, None]
    coef, *_ = np.linalg.lstsq(design / err[:, None], flux / err, rcond=None)
    continuum, leftover = curve @ coef, flux - design @ coef
    blocks = np.array_split(np.arange(len(leftover)), max(len(leftover) // 20, 1))

    def trial(share, apart):
        lines = (1 - share) * templates.at(teff, feh, width, wave, v0) \
            + share * templates.at(teff, feh, width, wave, v0 + apart)
        noise = np.concatenate([leftover[blocks[i]] for i in rng.permutation(len(blocks))])
        copy = dict(spec)
        copy["norm"] = spec["norm"].copy()
        copy["norm"][sel] = continuum * lines + noise
        return analyse(copy, templates, teff)

    blank = np.array([trial(0.0, 0.0)["gain"] for _ in range(TEST_BLANKS)])
    bar = 1.2 * float(blank.max())
    found = {}
    for share in TEST_SHARES:
        for apart in TEST_SEPARATIONS:
            hits = 0
            for number in range(TEST_TRIALS):
                r = trial(share, apart if number % 2 == 0 else -apart)
                hits += bool(r["double"] or (r["gain"] > bar and abs(r["apart"] - apart) < 25))
            found[(share, apart)] = hits
    return bar, blank, found


# ----------------------------------------------------------------------------
# Picture
# ----------------------------------------------------------------------------
def draw(path, title, spec, result, templates, teff, double, v_a, v_b):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
    v, curve = result["velocities"], result["curve"]
    centre = 0.5 * (v_a + v_b) if double else v_a
    axes[0].plot(v, curve, color=BLUE, lw=2, label=f"LAMOST spectrum, {spec['utc'][:10]}")
    axes[0].plot(v, result["alone"], color=INK2, lw=1.2, ls=(0, (4, 3)), label="what a single star would give")
    axes[0].set_xlim(centre - 350, centre + 350)
    axes[0].set_ylim(min(-0.05, curve.min() - 0.02), curve.max() * 1.32)
    marks = (("Star A", v_a), ("Star B", v_b)) if double else (("The star", v_a),)
    for name, vel in marks:
        top = float(np.interp(vel, v, curve))
        axes[0].text(vel, top + 0.05 * curve.max(), f"{name}\n{vel:+.0f} km/s", ha="center", va="bottom",
                     color=INK, fontsize=10)
    axes[0].set_xlabel("Velocity along the line of sight (km/s)", color=INK2)
    axes[0].set_ylabel("Match with a model spectrum", color=INK2)
    axes[0].set_title(title, loc="left", color=INK, fontsize=13)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK2)

    wave, flux = result["wave"], result["flux"]
    one, two = result["one"], result["two"]
    x = (wave - wave.mean()) / (np.ptp(wave) / 2)

    def fitted(model):
        design = np.vander(x, 4) * model[:, None]
        coef, *_ = np.linalg.lstsq(design, flux, rcond=None)
        return design @ coef

    model_one = fitted(templates.at(teff, one["feh"], one["width"], wave, one["v"]))
    model_two = fitted((1 - two["share"]) * templates.at(teff, two["feh"], two["width"], wave, two["v1"])
                       + two["share"] * templates.at(teff, two["feh"], two["width"], wave, two["v2"]))
    lo, hi = (5160.0, 5192.0) if spec["arm"] == "B" else (6400.0, 6440.0)
    shift = 1 + centre / C
    sel = (wave > lo * shift) & (wave < hi * shift)
    axes[1].plot(wave[sel], flux[sel], color=INK, lw=1.3, label="observed")
    if double:
        axes[1].plot(wave[sel], model_two[sel], color=BLUE, lw=1.8, label="model with two stars")
        axes[1].plot(wave[sel], model_one[sel], color=ORANGE, lw=1.3, ls=(0, (4, 3)), label="model with one star")
    else:
        axes[1].plot(wave[sel], model_one[sel], color=BLUE, lw=1.8, label="model with one star")
    axes[1].set_xlabel("Wavelength in air (Å)", color=INK2)
    axes[1].set_ylabel("Brightness (continuum = 1)", color=INK2)
    axes[1].set_title("A stretch of the spectrum with the fitted model", loc="left", color=INK, fontsize=13)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left", ncol=3, labelcolor=INK2)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Velocities, and one star or two, from LAMOST medium-resolution spectra.")
    p.add_argument("files", nargs="*", help="LAMOST medium-resolution FITS files")
    p.add_argument("--obsid", nargs="*", default=[], help="LAMOST spectrum numbers to download first")
    p.add_argument("--teff", type=float, default=None, help="temperature of the star, K")
    p.add_argument("--gamma", type=float, default=None, help="the system's own velocity, km/s, if known")
    p.add_argument("--outdir", default=None)
    args = p.parse_args()
    if not args.files and not args.obsid:
        p.error("give a FITS file or --obsid NUMBER")

    say = Tee()
    paths = [Path(f) for f in args.files]
    for obsid in args.obsid:
        paths.append(download_spectrum(obsid, args.outdir or "."))
    templates = Templates(load_model(say))

    for path in paths:
        say.lines = []
        outdir = Path(args.outdir) if args.outdir else path.parent
        outdir.mkdir(parents=True, exist_ok=True)
        stem = path.name.replace(".fits.gz", "").replace(".fits", "")
        say(f"\n{path.name}")
        spectra, info = read_spectrum(path, say)
        say(f"  {info['desig']}  (ObsID {info['obsid']}), {len(spectra)} spectra: "
            + ", ".join(f"{s['name']} (signal-to-noise {s['snr']:.0f})" for s in spectra))
        faint = np.nanmedian([s["snr"] for s in spectra]) < FAINT
        if faint and len(spectra) > len({s["arm"] for s in spectra}):
            spectra = stack_exposures(spectra)
            say(f"  Faint (signal-to-noise under {FAINT:.0f} per exposure), so the exposures of each arm are averaged: "
                + ", ".join(f"{s['name']} (signal-to-noise {s['snr']:.0f})" for s in spectra))
        teff = args.teff or choose_temperature(templates, spectra, say)
        if not 3600 <= teff <= 7900:
            sys.exit("The model spectra cover 3600 to 7900 K only.")

        results = [analyse(s, templates, teff) for s in spectra]
        votes = sum(r["double"] for r in results)
        double = votes * 2 > len(results) or (votes > 0 and all(r["double"] for s, r in zip(spectra, results)
                                                                 if s["arm"] == "B"))
        rows = []
        say("\nMEASUREMENTS")
        for s, r in zip(spectra, results):
            two, one = r["two"], r["one"]
            if double:
                # star A is the one giving more of the light
                first = two["share"] <= 0.5
                v_a, v_b = (two["v1"], two["v2"]) if first else (two["v2"], two["v1"])
                e_a, e_b = (two["errors"][0], two["errors"][1]) if first else (two["errors"][1], two["errors"][0])
                share_b = two["share"] if first else 1 - two["share"]
                say(f"  {s['name']:<12} {s['utc']}  star A {v_a:+7.1f} +/- {e_a:.1f}   star B {v_b:+7.1f} +/- {e_b:.1f} km/s   "
                    f"B gives {100 * share_b:.0f}% of the light   (two stars fit {r['better']:.1f} times better than one)")
                rows.append({"spectrum": s["name"], "arm": s["arm"], "utc": s["utc"], "jd": s["jd"], "v_a": v_a,
                             "e_a": e_a, "v_b": v_b, "e_b": e_b, "share_b": share_b, "feh": two["feh"],
                             "extra_width": two["width"], "chi2_one": r["chi_one"], "chi2_two": r["chi_two"]})
            else:
                say(f"  {s['name']:<12} {s['utc']}  velocity {one['v']:+7.1f} km/s   "
                    f"(a second star improves the fit only {r['better']:.2f} times)")
                rows.append({"spectrum": s["name"], "arm": s["arm"], "utc": s["utc"], "jd": s["jd"], "v_a": one["v"],
                             "e_a": np.nan, "v_b": np.nan, "e_b": np.nan, "share_b": np.nan, "feh": one["feh"],
                             "extra_width": one["width"], "chi2_one": r["chi_one"], "chi2_two": r["chi_two"]})
        table = pd.DataFrame(rows)
        table.to_csv(outdir / f"{stem}_velocities.csv", index=False)

        say("\nRESULT")
        scatter = lambda v: max(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 2.0)   # never claim better than 2 km/s
        epochs = []
        if double:
            v_a, v_b, share = table.v_a.mean(), table.v_b.mean(), table.share_b.mean()
            say(f"  DOUBLE-LINED: two sets of lines, {abs(v_a - v_b):.0f} km/s apart.")
            say(f"  Star A {v_a:+.1f} km/s, star B {v_b:+.1f} km/s (each good to about "
                f"{max(scatter(table.v_a), scatter(table.v_b)):.0f} km/s).")
            say(f"  Star B gives about {100 * share:.0f}% of the light, if the two stars have the same temperature.")
            say(f"  Midpoint of the two velocities: {0.5 * (v_a + v_b):+.1f} km/s. For two equal stars that is the")
            say("  system's own velocity.")
            if args.gamma is not None:
                q = (args.gamma - v_a) / (v_b - args.gamma)
                say(f"  With the system's velocity at {args.gamma:+.1f} km/s, star B has {q:.2f} times the mass of star A"
                    if q > 0 else
                    f"  A system velocity of {args.gamma:+.1f} km/s does not lie between the two stars; check it.")
            # one row per star for the whole visit: its exposures are minutes apart
            for label, col in (("A", "v_a"), ("B", "v_b")):
                epochs.append({"JD": round(float(table.jd.mean()), 4), "RV": round(float(table[col].mean()), 1),
                               "RV_ERR": round(max(scatter(table[col]), 3.0), 1),
                               "SOURCE": f"LAMOST medium resolution spectrum {info['obsid']}, two-star fit",
                               "STAR": label})
        else:
            best = sorted(range(len(spectra)), key=lambda i: (spectra[i]["arm"] != "B", -np.nan_to_num(spectra[i]["snr"])))[0]
            say(f"  Testing what this spectrum could have shown ({spectra[best]['name']}; a minute or two) ...")
            bar, blank, could = what_could_show(spectra[best], results[best], templates, teff)
            r = results[best]
            hint = r["gain"] > bar and r["apart"] > MIN_SEPARATION and 0.08 < r["two"]["share"] < 0.92
            say(f"  {'ONE SET OF LINES, WITH A HINT OF A SECOND' if hint else 'SINGLE-LINED'}: "
                f"one set of lines, at {table.v_a.mean():+.1f} km/s (good to about {scatter(table.v_a):.0f} km/s).")
            say(f"  Adding a second star to the model improves the fit by {r['gain']:.0f} (in units of the noise). Noise alone,")
            say(f"  with no second star, gave {np.median(blank):.0f} typically and up to {blank.max():.0f} in {len(blank)} trials"
                + (f"; the bar for a hint is {bar:.0f}." if not hint else "."))
            if hint:
                say(f"  The fit puts a second star at {r['two']['v2']:+.0f} km/s with {100 * r['two']['share']:.0f}% of the light. "
                    "That is more than noise")
                say("  usually gives but far from certain: look at the picture, and get another spectrum.")
            say("  What would have shown: a second star was added to copies of this spectrum and looked for.")
            say("  Found in how many of " + str(TEST_TRIALS) + " copies:")
            say("      share of the light   " + "   ".join(f"{a:.0f} km/s apart" for a in TEST_SEPARATIONS))
            for share in TEST_SHARES:
                say(f"      {100 * share:>10.0f}%          "
                    + "   ".join(f"{could[(share, a)]:>7d} of {TEST_TRIALS}" for a in TEST_SEPARATIONS))
            say(f"  Closer than about {MIN_SEPARATION:.0f} km/s, two stars merge into one set of lines whatever their brightness.")
            say("  So this says only that no such star stood apart at this moment; it may still be there.")
            epochs.append({"JD": round(float(table.jd.mean()), 4), "RV": round(float(table.v_a.mean()), 1),
                           "RV_ERR": round(max(scatter(table.v_a), 3.0), 1),
                           "SOURCE": f"LAMOST medium resolution spectrum {info['obsid']}", "STAR": "1"})
        pd.DataFrame(epochs).to_csv(outdir / f"{stem}_rv_epochs.csv", index=False)

        # picture from the best blue spectrum (most lines), or the best there is
        order = sorted(range(len(spectra)), key=lambda i: (spectra[i]["arm"] != "B", -np.nan_to_num(spectra[i]["snr"])))
        i = order[0]
        title = (f"{info['desig'] or stem}: two sets of lines, {abs(table.v_a.mean() - table.v_b.mean()):.0f} km/s apart"
                 if double else f"{info['desig'] or stem}: one set of lines")
        draw(outdir / f"{stem}_double_lines.png", title, spectra[i], results[i], templates, teff, double,
             table.v_a.iloc[i], table.v_b.iloc[i] if double else np.nan)
        (outdir / f"{stem}_double_lines.txt").write_text("\n".join(say.lines) + "\n")
        say(f"\n  Saved {stem}_double_lines.png, {stem}_velocities.csv, {stem}_rv_epochs.csv in {outdir}/")


if __name__ == "__main__":
    main()
