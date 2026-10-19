#!/usr/bin/env python
"""
system_view.py - a picture of the system that joint_orbit.py describes.

It reads the output folder of a joint_orbit.py run and draws one figure:

  1. The two stars to scale, coloured as their temperatures would look to the
     eye, with the Sun alongside for comparison.
  2. The orbit seen from above, for the most probable period, eccentricity and
     companion mass, with a dot at equal time steps to show where the
     companion hurries and where it lingers.
  3. The eclipse as seen from Earth: how much of the star the companion covers.
  4. A wide map of the sky around the star, with constellations.
  5. A close-up map of the same spot.

Usage
-----
Run it in the folder that holds joint_orbit.py:

    python system_view.py joint_output
    python system_view.py joint_149762453

The picture is saved in that folder as system_view.png.

The first run downloads a star catalogue and constellation figures (about
7 MB, from the d3-celestial project on GitHub) into a folder called sky_data
next to this script. Later runs reuse them. With --no-sky the two maps are
left out and nothing is downloaded.

What the picture is, and is not
-------------------------------
* It shows the single most probable value of each quantity in summary.txt
  (the last block under RESULTS). The real ranges are wide; the panel at the
  lower right lists them.
* The companion's temperature is not measured. It is estimated from the
  fitted mass, assuming an ordinary star of that mass.
* The direction of the orbit's long axis follows from how fast the companion
  crossed the star. Its mirror image fits the eclipse equally well. If
  joint_orbit.py had velocities that settle which is right, the picture uses
  that; otherwise it draws closest approach before the eclipse.
* Star colours are what a glowing body of that temperature looks like to the
  eye: real stars differ far less in colour than artists usually paint them.
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
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

NEEDED = ('joint_orbit.py', 'gaia_tools.py', 'stellar_model.py')
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import joint_orbit as jo
except ImportError as exc:
    here = Path(__file__).resolve().parent
    missing = [f for f in NEEDED if not (here / f).exists()]
    sys.exit(f"Could not import {exc.name or 'a module'}: {exc}\n"
             + (f"Missing from {here}: {', '.join(missing)}" if missing else
                f"(all of {', '.join(NEEDED)} are in {here}; if the module named above is a package, "
                f"pip install it)"))

# ---- colours: one dark surface, text in two weights, one accent for the target ----
SURFACE = "#14161a"
SKY = "#0c1018"
INK = "#f0efec"
INK2 = "#c3c2b7"
MUTED = "#7d7f86"
ACCENT = "#3987e5"
FIGURE_LINE = "#4f6288"       # constellation figures
GRID = "#2a303b"

SKY_URL = "https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/"
SKY_FILES = ("constellations.lines.json", "constellations.json", "stars.6.json",
             "stars.8.json", "starnames.json")
SUN_T, MERCURY_AU, RSUN_AU = 5772.0, 0.387, 0.00465047


# ----------------------------------------------------------------------------
# Reading joint_orbit.py's output
# ----------------------------------------------------------------------------
def read_catalogue(text):
    """summary.txt's CATALOGUE VALUES block (joint_orbit.py) as a dict of catalogue and Gaia DR3
    values, named as in star.json and gaia.json; values written n/a are left out. {} if absent."""
    if "CATALOGUE VALUES" not in text:
        return {}
    block = text.split("CATALOGUE VALUES", 1)[1].split("\n\n", 1)[0]
    val = r"([-+\d.]+|n/a)"
    pats = {
        "tic": r"TIC (\d+): RA", "ra": rf"RA {val} deg", "dec": rf"Dec {val} deg", "tmag": rf"Tmag {val}",
        "teff": rf"Teff {val} K \(combined", "source_id": r"Gaia DR3 (\d+|n/a): parallax",
        "parallax": rf"parallax {val} \+/-", "parallax_error": rf"parallax \S+ \+/- {val} mas",
        "ruwe": rf"RUWE {val},", "astrometric_excess_noise": rf"excess noise {val} mas",
        "phot_g_mean_mag": rf", G {val},", "bp_rp": rf"BP-RP {val}",
        "radial_velocity": rf"velocity: {val} \+/-", "radial_velocity_error": rf"velocity: \S+ \+/- {val} km/s",
        "rv_nb_transits": rf"from {val} visits", "rv_method_used": rf"method {val},",
        "rv_amplitude_robust": rf"robust amplitude {val} km/s",
    }
    out = {}
    for key, pat in pats.items():
        m = re.search(pat, block)
        if m and m.group(1) != "n/a":
            out[key] = int(m.group(1)) if key in ("tic", "source_id") else float(m.group(1))
    return out


STAR_KEYS = ("tic", "ra", "dec", "tmag", "teff")


def read_run(folder):
    """Collect the most probable values from summary.txt. Summaries from before joint_orbit.py wrote
    its CATALOGUE VALUES block need star.json (and gaia.json) beside them as well."""
    summary = (folder / "summary.txt")
    if not summary.exists():
        sys.exit(f"{folder} does not look like a joint_orbit.py output folder (no summary.txt).")
    text = summary.read_text()
    cat = read_catalogue(text)
    if cat:
        star = {k: cat[k] for k in STAR_KEYS if k in cat}
        gaia = {k: v for k, v in cat.items() if k not in STAR_KEYS}
        line = re.search(r"TIC \d+: radius ([\d.]+) Rsun, mass ([\d.]+) Msun", text)
        if line:                                  # the catalogue's values, before any correction
            star.update(rad=float(line.group(1)), mass=float(line.group(2)))
    else:
        if not (folder / "star.json").exists():
            sys.exit(f"{folder}/summary.txt has no CATALOGUE VALUES block and there is no star.json: "
                     "rerun the current joint_orbit.py.")
        print(f"  ({folder}/summary.txt is from an older joint_orbit.py: reading star.json and gaia.json too)")
        star = json.loads((folder / "star.json").read_text(encoding="utf-8-sig"))
        gaia = json.loads((folder / "gaia.json").read_text()) if (folder / "gaia.json").exists() else {}

    results = text.split("RESULTS", 1)[-1].split("ECCENTRICITY", 1)[0]
    number = r"([\d.]+)"

    def last(label, unit):
        found = re.findall(rf"{label}\s+{number} {unit}\s+\(68%: {number} to {number};\s+95%: "
                           rf"{number} to {number}\)", results)
        if not found:
            sys.exit(f"Could not find the {label} in summary.txt.")
        return [float(v) for v in found[-1]]       # value, 68% low, 68% high, 95% low, 95% high

    main = re.search(rf"MAIN STAR: mass {number} Msun, radius {number} Rsun", text)
    if main:                                      # joint_orbit.py corrected the star for its companion's light
        star = dict(star, mass=float(main.group(1)), rad=float(main.group(2)))
    run = {"star": star, "gaia": gaia, "period": last("period", "days"),
           "mass": last("mass", "Msun"), "radius": last("radius", "Rsun")}
    ecc = re.search(rf"Eccentricity {number} \(68%: {number} to {number}", text)
    run["ecc"] = [float(v) for v in ecc.groups()] if ecc else [0.0, 0.0, 0.0]
    fast = re.search(rf"crossed {number} times faster", text)
    run["speed_factor"] = float(fast.group(1)) if fast else 1.0
    side = re.search(r"Closest approach comes (before|after) the eclipse", text)
    run["closest"] = side.group(1) if side else None       # known only if velocities settled it
    # joint_orbit.py also names one orbit that respects every test, including the companion's
    # own eclipse; medians taken one at a time need not, so that orbit is drawn when given
    single = re.search(rf"Representative orbit \(the median eccentricity, at its most probable speed\): eccentricity {number}, crossing {number} times faster "
                       rf"than circular, closest approach {number} days (before|after)", text)
    if single:
        run["ecc"][0] = float(single.group(1))
        run["speed_factor"] = float(single.group(2))
        tail = text[single.end():single.end() + 40]
        run["closest"] = None if "or after" in tail else single.group(4)
    # The stars' light and temperatures, as joint_orbit.py fitted them (from its isochrone when it used
    # one), so the model eclipse drawn here is the one it fitted. Summaries from before joint_orbit.py
    # wrote this line fall back on light ~ mass^4.5 and main-sequence temperatures.
    run["light2"], run["teff2"] = None, None
    run["isochrone"] = "STARS: isochrone" in text
    temps = re.search(rf"temperatures: eclipsed star {number} K .*?companion {number} K .*?"
                      rf"companion's TESS-band light share {number}", text)
    if temps:
        star = dict(star, teff=float(temps.group(1)))
        run["star"] = star
        run["teff2"], run["light2"] = float(temps.group(2)), float(temps.group(3))
    else:
        share = re.search(r"companion's share of the (?:catalogue's|TESS-band) light \(([\d.]+)%", text)
        if share:
            run["light2"] = float(share.group(1)) / 100
        print(f"  ({folder}/summary.txt has no 'temperatures:' line - rerun the current joint_orbit.py; "
              "until then the companion's " + ("temperature is a rough estimate" if share else
              "temperature and light are rough estimates") + " from its mass)")
    # The eclipse joint_orbit.py actually fitted (radius ratio, impact parameter, crossing speed in
    # stellar radii per day, the eclipsed star's share of the light). Drawn as is, so the model curve
    # is the fitted one; medians of mass, radius and speed taken one at a time need not reproduce it.
    shape = re.search(rf"BEST-FIT ECLIPSE: radius ratio {number}, impact parameter {number} stellar radii, "
                      rf"crossing speed {number} stellar radii per day, eclipsed star's share of the light "
                      rf"{number}, length {number} hours", text)
    run["shape"] = dict(zip(("k", "impact", "speed_r1", "dilution", "hours"),
                            (float(v) for v in shape.groups()))) if shape else None
    if not shape:
        print(f"  ({folder}/summary.txt has no BEST-FIT ECLIPSE line - rerun the current joint_orbit.py; until "
              "then the eclipse drawn is rebuilt from the medians and may not match the fit exactly)")
    # One system that all the evidence favours (joint_orbit.py's MOST PROBABLE SYSTEM): when given,
    # every number drawn - orbit, both stars, the eclipse - comes from it, so they belong together.
    run["system"] = False
    if "MOST PROBABLE SYSTEM" in text:
        block = text.split("MOST PROBABLE SYSTEM", 1)[1].split("\n\n", 1)[0]
        orb = re.search(rf"period {number} d, eccentricity {number}, crossing {number} times faster than circular, "
                        rf"closest approach {number} days (before or after|before|after)", block)
        stars = re.search(rf"main star: mass {number} Msun, radius {number} Rsun, {number} K; companion: mass "
                          rf"{number} Msun, radius {number} Rsun, {number} K", block)
        ecl = re.search(rf"radius ratio {number}, impact parameter {number} stellar radii, crossing speed {number} "
                        rf"stellar radii per day, eclipsed star's share of the light {number}, length {number} hours",
                        block)
        if orb and stars and ecl:
            star = dict(star, mass=float(stars.group(1)), rad=float(stars.group(2)), teff=float(stars.group(3)))
            run["star"] = star
            run["period"][0], run["ecc"][0] = float(orb.group(1)), float(orb.group(2))
            run["speed_factor"] = float(orb.group(3))
            run["closest"] = None if orb.group(5) == "before or after" else orb.group(5)
            run["mass"][0], run["radius"][0] = float(stars.group(4)), float(stars.group(5))
            run["teff2"] = float(stars.group(6))
            run["shape"] = dict(zip(("k", "impact", "speed_r1", "dilution", "hours"),
                                    (float(v) for v in ecl.groups())))
            run["light2"] = 1.0 - run["shape"]["dilution"]
            run["system"] = True
    if not run["system"]:
        print(f"  ({folder}/summary.txt has no MOST PROBABLE SYSTEM block - rerun the current joint_orbit.py; "
              "until then the orbit and stars drawn are medians, which need not belong to one system)")
    known = re.search(rf"KNOWN ECLIPSE in \S+: centre JD {number}, deepest about {number}%, "
                      rf"{number}% averaged over {number} hours", text)
    run["depth"] = float(known.group(2)) / 100 if known else None
    run["hours"] = float(known.group(4)) if known else None
    return run


# ----------------------------------------------------------------------------
# Physics for the picture
# ----------------------------------------------------------------------------
def star_temperature(mass):
    """Rough surface temperature of an ordinary star of a given mass (solar masses)."""
    m = [0.08, 0.10, 0.16, 0.23, 0.44, 0.57, 0.70, 0.82, 0.88, 1.00, 1.08, 1.18, 1.33, 1.50, 2.0]
    t = [2500, 2800, 3050, 3200, 3550, 3850, 4440, 5040, 5270, 5770, 5930, 6150, 6550, 7200, 9000]
    return float(np.interp(mass, m, t))


def glow_colour(temperature):
    """Colour of a glowing body at this temperature, as red, green, blue from 0 to 1."""
    t = float(np.clip(temperature, 1700.0, 25000.0))
    if t < 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t < 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t < 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    big = np.array([x / y, 1.0, (1 - x - y) / y])
    rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415],
                    [0.0557, -0.2040, 1.0570]]) @ big
    rgb = np.clip(rgb, 0, None)
    rgb = rgb / rgb.max()
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def companion_teff(run, geo=None):
    """The companion's temperature: as joint_orbit.py fitted it (summary.txt), else estimated
    from its mass for an ordinary star."""
    if run.get("teff2"):
        return run["teff2"]
    return star_temperature(geo["m2"] if geo else run["mass"][0])


def companion_light(run, m2=None):
    """The companion's share of the TESS-band light of the pair."""
    if run.get("light2") is not None:
        return run["light2"]
    m2 = run["mass"][0] if m2 is None else m2
    q = (m2 / run["star"]["mass"]) ** 4.5
    return q / (1 + q)


def orbit_geometry(run):
    """Size, shape and orientation of the most probable orbit.

    With the fitted eclipse from summary.txt (run["shape"]), the radius ratio, impact parameter,
    light share and crossing speed (in stellar radii per day) are the fitted ones, so the eclipse
    drawn has the fitted length and shape."""
    star = run["star"]
    period, m2, e = run["period"][0], run["mass"][0], run["ecc"][0]
    a_au = ((star["mass"] + m2) * (period / 365.25) ** 2) ** (1 / 3)
    shape = run.get("shape")
    factor = run["speed_factor"]
    # how fast the companion crossed the star fixes where along the orbit the eclipse happens
    sin_w = np.clip((factor * np.sqrt(1 - e**2) - 1) / e, -1, 1) if e > 0 else 0.0
    omega = float(np.arcsin(sin_w))                         # closest approach before the eclipse...
    if run.get("closest") == "after":
        omega = np.pi - omega                               # ...or its mirror image, after it
    if shape:
        k, impact = shape["k"], shape["impact"]
        light2 = 1.0 - shape["dilution"]
    else:
        k = run["radius"][0] / star["rad"]
        light2 = companion_light(run, m2)
        dilution = 1.0 - light2
        impact = None
        if run["depth"]:
            z, hidden = jo.blocked_light(k, *jo.limb_darkening(star["teff"]))
            depth = dilution * hidden
            if run["depth"] < depth.max():
                start = int(np.argmax(depth))                          # depth falls outward from here
                impact = float(np.interp(run["depth"], depth[start:][::-1], z[start:][::-1]))
            else:
                impact = 0.0
    r_eclipse = a_au * (1 - e**2) / (1 + e * np.sin(omega))
    # The eclipse's shape in time depends only on the radius ratio, the impact parameter and the
    # crossing speed in stellar radii per day. With the fitted shape, the light curve is worked out in
    # units of the star radius that makes this orbit cross at the fitted speed, so the model curve is
    # the fitted one whatever the orbit; the stars are still drawn at their median radii.
    r1_light = star["rad"] * RSUN_AU
    if shape:
        v_conj = 2 * np.pi * a_au / period * (1 + e * np.sin(omega)) / np.sqrt(1 - e**2)   # AU per day
        r1_light = v_conj / shape["speed_r1"]
        ratio = r1_light / (star["rad"] * RSUN_AU)
        if abs(ratio - 1) > 0.1:
            print(f"  (the fitted eclipse crosses {shape['speed_r1']:.1f} stellar radii per day; on the orbit drawn "
                  f"that matches a {r1_light / RSUN_AU:.2f} Rsun star, against the median {star['rad']:.2f} Rsun - "
                  "the light curve uses the former, the pictures the latter)")
    tilt = None
    if impact is not None:
        tilt = float(np.degrees(np.arcsin(min(impact * r1_light / r_eclipse, 1.0))))
    return {"a": a_au, "e": e, "omega": omega, "k": k, "impact": impact, "tilt": tilt,
            "period": period, "m2": m2, "r_eclipse": r_eclipse, "light2": light2, "speed_factor": factor,
            "r1_light": r1_light}


def orbit_path(a, e, omega, phase):
    """Companion's position relative to the star at orbital phases 0..1 (0 = closest approach).

    Earth lies far below the picture, so the eclipse happens directly below the star.
    """
    mean = 2 * np.pi * np.asarray(phase, float)
    ecc_anom = mean.copy()
    for _ in range(60):
        ecc_anom = ecc_anom - (ecc_anom - e * np.sin(ecc_anom) - mean) / (1 - e * np.cos(ecc_anom))
    true = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(ecc_anom / 2), np.sqrt(1 - e) * np.cos(ecc_anom / 2))
    r = a * (1 - e * np.cos(ecc_anom))
    return r * np.cos(true + omega), -r * np.sin(true + omega)


# ----------------------------------------------------------------------------
# Drawing: stars and orbit
# ----------------------------------------------------------------------------
def draw_disc(ax, x, y, radius, temperature, zorder=3):
    """A star's disc, darker toward its edge as real stars are."""
    n = 240
    g = (np.arange(n) + 0.5) / n * 2 - 1
    xx, yy = np.meshgrid(g, g)
    rr = np.hypot(xx, yy)
    mu = np.sqrt(np.clip(1 - rr**2, 0, 1))
    shade = 1 - 0.6 * (1 - mu)
    img = np.zeros((n, n, 4))
    img[..., :3] = glow_colour(temperature)[None, None, :] * shade[..., None]
    img[..., 3] = np.clip((1 - rr) * n / 3, 0, 1)                  # soft edge
    ax.imshow(img, extent=(x - radius, x + radius, y - radius, y + radius), zorder=zorder,
              interpolation="bilinear")


def bare(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=8)


def panel_stars(ax, bar, run, geo):
    star = run["star"]
    t2 = companion_teff(run, geo)
    bodies = [("Sun, for scale", 1.0, SUN_T, ["5,772 K"]),
              ("The star", star["rad"], star["teff"],
               [f"{star['teff']:,.0f} K", f"{star['rad']:.2f} × Sun's radius",
                f"{star['mass']:.2f} × Sun's mass"]),
              ("The companion", run["radius"][0], t2,
               [f"about {round(t2, -2):,.0f} K" + ("" if run.get("teff2") else "*"), f"{run['radius'][0]:.2f} × Sun's radius",
                f"{run['mass'][0]:.2f} × Sun's mass"])]
    biggest = max(b[1] for b in bodies)
    slot = 2.3 * biggest                                   # equal room for each, so labels never collide
    bare(ax, "The two stars, to scale")
    for i, (name, radius, temp, lines) in enumerate(bodies):
        x = (i - 1) * slot
        draw_disc(ax, x, 0.0, radius, temp)
        ax.text(x, -1.25 * biggest, name, color=INK, ha="center", va="top", fontsize=10.5,
                fontweight="bold")
        ax.text(x, -1.62 * biggest, "\n".join(lines), color=INK2, ha="center", va="top",
                fontsize=8.5, linespacing=1.5)
    ax.set_xlim(-1.5 * slot, 1.5 * slot)
    ax.set_ylim(-biggest * 3.0, biggest * 1.25)
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.text(0.0, 0.0, ("temperatures from the fitted isochrone" if run.get("isochrone") else
             "temperatures as fitted by joint_orbit.py") if run.get("teff2") else
            "*estimated from the fitted mass, assuming an ordinary star",
            transform=ax.transAxes, color=MUTED, fontsize=7.5, va="bottom")

    # ---- temperature strip ----
    temps = [SUN_T, star["teff"], t2]
    lo, hi = max(min(temps) - 1800, 2300), max(temps) + 1800
    grid = np.linspace(lo, hi, 400)
    bar.imshow(np.array([glow_colour(t) for t in grid])[None, :, :], extent=(lo, hi, 0, 1), aspect="auto")
    bar.set_facecolor(SURFACE)
    bar.set_yticks([])
    bar.set_xlim(lo, hi)
    bar.set_ylim(0, 1)
    bar.tick_params(colors=INK2, labelsize=8, length=3)
    for side in bar.spines.values():
        side.set_visible(False)
    bar.set_xlabel("Surface temperature (K) and the colour it gives", color=INK2, fontsize=8.5)
    order = np.argsort(temps)
    for rank, i in enumerate(order):
        height = 1.25 + 0.55 * (rank % 2)
        bar.plot([temps[i], temps[i]], [1.0, height], color=INK2, lw=0.8, clip_on=False)
        bar.text(temps[i], height + 0.05, ("Sun", "Star", "Companion")[i], color=INK, fontsize=8.5,
                 ha="center", va="bottom", clip_on=False)


def panel_orbit(ax, run, geo):
    star = run["star"]
    a, e, omega = geo["a"], geo["e"], geo["omega"]
    bare(ax, "The orbit, seen from above")
    x_all, y_all = orbit_path(a, e, omega, np.linspace(0, 1, 800))
    ax.plot(x_all, y_all, color=INK2, lw=1.4, zorder=2)

    steps = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 50])
    step = steps[np.argmin(np.abs(geo["period"] / steps - 60))]
    ticks = np.arange(0, geo["period"], step) / geo["period"]
    ax.plot(*orbit_path(a, e, omega, ticks), ".", color=INK, ms=3.2, zorder=3)

    r1 = star["rad"] * RSUN_AU
    ex, ey = 0.0, -geo["r_eclipse"]
    span = max(np.ptp(x_all), np.ptp(y_all))
    # At about 270 points across the picture, can the discs be seen at their true size?
    to_scale = r1 / (1.4 * span) * 270 > 1.5
    for x, y, radius, temp, z in ((0, 0, r1, star["teff"], 5),
                                  (ex, ey, run["radius"][0] * RSUN_AU, companion_teff(run, geo), 6)):
        if to_scale:
            draw_disc(ax, x, y, radius, temp, zorder=z)
        else:
            ax.plot(x, y, "o", ms=4.5 if z == 5 else 3.5, color=glow_colour(temp), zorder=z)
    cx, cy = 0.5 * (x_all.max() + x_all.min()), 0.5 * (y_all.max() + y_all.min())
    half = 0.70 * span
    left, right, bottom, top = cx - half, cx + half, cy - 1.12 * half, cy + 0.88 * half
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_anchor("NW")

    # Mercury's orbit for scale, if it is not wildly bigger or smaller than the picture
    mercury = 0.15 * half < MERCURY_AU < 1.6 * half
    if mercury:
        ax.add_patch(Circle((0, 0), MERCURY_AU, fill=False, ec=MUTED, lw=0.9, ls=(0, (1, 4)), zorder=0))

    # closest and farthest points, and the two bodies
    px, py = orbit_path(a, e, omega, [0.0, 0.5])
    near, far = a * (1 - e), a * (1 + e)
    out = lambda x, y, d: (d * np.sign(x - cx) if abs(x - cx) > 0.05 * half else d,
                           d * np.sign(y - cy) if abs(y - cy) > 0.05 * half else d)
    line = dict(arrowstyle="-", color=MUTED, lw=0.7)
    marks = [((px[0], py[0]), f"closest: {near:.2f} AU\n({near / r1:.0f} star radii)"),
             ((px[1], py[1]), f"farthest: {far:.2f} AU")]
    if e < 0.02:                                           # a circle has no closest or farthest point
        marks = [((px[0], py[0]), f"always {a:.2f} AU apart\n({a / r1:.0f} star radii)")]
    for (x, y), text in marks:
        dx, dy = out(x, y, 16)
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2, fontsize=8.5,
                    ha="left" if dx > 0 else "right", va="bottom" if dy > 0 else "top", arrowprops=line)
    side = -1 if px[0] > 0 else 1                         # keep clear of the closest-approach label
    ax.annotate("the star", (0, 0), xytext=(side * 18, 12), textcoords="offset points", color=INK2,
                fontsize=8.5, ha="left" if side > 0 else "right", arrowprops=line)
    ax.annotate("companion at\nthe eclipse", (ex, ey), xytext=(side * 18, -16), textcoords="offset points",
                color=INK2, fontsize=8.5, va="top", ha="left" if side > 0 else "right", arrowprops=line)

    # line of sight
    ax.plot([0, 0], [ey - 0.04 * half, bottom + 0.14 * half], color=ACCENT, lw=1.0, ls=(0, (2, 3)), zorder=1)
    ax.add_patch(FancyArrowPatch((0, bottom + 0.15 * half), (0, bottom + 0.05 * half), color=ACCENT,
                                 arrowstyle="-|>", mutation_scale=11, lw=1.0))
    ax.text(-side * 0.04 * half, bottom + 0.07 * half, "to Earth", color=INK, fontsize=9, va="center",
            ha="right" if side > 0 else "left")

    # scale bar, in whichever lower corner is farther from the line of sight
    nice = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    length = nice[np.argmin(np.abs(nice - 0.35 * half))]
    x0 = left + 0.07 * half if 0 > cx else right - 0.07 * half - length
    y0 = bottom + 0.06 * half
    ax.plot([x0, x0 + length], [y0, y0], color=INK2, lw=1.5, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0 + 0.03 * half, f"{length:g} AU", color=INK2, fontsize=8.5, ha="center")
    note = f"One dot every {step:g} day{'s' if step != 1 else ''}. "
    if e >= 0.02 and not run.get("closest"):
        note = "The mirror image of this orbit fits equally well. " + note
    note += "Both stars are drawn to scale." if to_scale else "The stars are too small to draw to scale here."
    if mercury:
        note += "\nDotted circle: the size of Mercury's orbit."
    ax.text(0.0, -0.035, note, transform=ax.transAxes, color=MUTED, fontsize=7.5, va="top")


def panel_eclipse(ax, run, geo):
    star = run["star"]
    bare(ax, "The eclipse, seen from Earth")
    k, b = geo["k"], geo["impact"]
    draw_disc(ax, 0, 0, 1.0, star["teff"], zorder=3)
    if b is not None:
        ax.plot([-2.6, 2.6], [b, b], color=MUTED, lw=0.9, ls=(0, (4, 4)), zorder=2)
        draw_disc(ax, 0, b, k, companion_teff(run, geo), zorder=4)
        ax.add_patch(Circle((0, b), k, fill=False, ec=SURFACE, lw=1.2, zorder=5))
        ax.add_patch(FancyArrowPatch((1.5, b), (2.5, b), color=INK2, arrowstyle="-|>", mutation_scale=10,
                                     lw=1.0, zorder=2))
        lines = [f"At mid-eclipse the companion hides {100 * run['depth']:.1f}% of the light."]
        if run["hours"]:
            lines.append(f"The whole crossing takes about {run['hours']:g} hours.")
        if geo["tilt"] is not None:
            lines.append(f"Orbit tilted {geo['tilt']:.1f}° from exactly edge-on.")
        ax.text(0.0, 0.0, "\n".join(lines), transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom",
                linespacing=1.5)
        top = max(1.0, b + k)
    else:
        top = 1.0
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(top - 5.4 + 0.5, top + 0.5)
    ax.set_aspect("equal")
    ax.set_anchor("NW")


# ----------------------------------------------------------------------------
# Drawing: the sky
# ----------------------------------------------------------------------------
def fetch_sky(folder):
    """Download the star catalogue and constellation figures once; return them."""
    folder.mkdir(exist_ok=True)
    for name in SKY_FILES:
        path = folder / name
        if path.exists() and path.stat().st_size > 1000:
            continue
        print(f"  downloading {name} ...")
        try:
            import requests

            reply = requests.get(SKY_URL + name, timeout=120)
            reply.raise_for_status()
            path.write_bytes(reply.content)
        except ImportError:
            from urllib.request import urlopen

            path.write_bytes(urlopen(SKY_URL + name, timeout=120).read())
    load = lambda name: json.loads((folder / name).read_text(encoding="utf-8"))

    def stars(name):
        rows = [(f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][1], f["properties"]["mag"],
                 float(f["properties"]["bv"] or 0.6), f["id"]) for f in load(name)["features"]]
        return np.array(rows, float)

    return {"lines": load("constellations.lines.json")["features"],
            "names": load("constellations.json")["features"], "bright": stars("stars.6.json"),
            "faint": stars("stars.8.json"), "starnames": load("starnames.json")}


def project(ra, dec, ra0, dec0):
    """Flatten the sky around (ra0, dec0). Returns x (east to the left), y (north up), and
    the angle from the centre in degrees."""
    a, d = np.radians(np.asarray(ra, float) - ra0), np.radians(np.asarray(dec, float))
    d0 = np.radians(dec0)
    cosc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a)
    scale = 2.0 / (1.0 + np.clip(cosc, -0.999, 1.0))
    x = -scale * np.cos(d) * np.sin(a)
    y = scale * (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a))
    return np.degrees(x), np.degrees(y), np.degrees(np.arccos(np.clip(cosc, -1, 1)))


def edge_crossing(x, y, half, edge):
    """Where a curve first crosses the left or bottom edge of the map, or None."""
    for i in range(len(x) - 1):
        if edge == "left" and (x[i] + half) * (x[i + 1] + half) < 0:
            t = (-half - x[i]) / (x[i + 1] - x[i])
            yy = y[i] + t * (y[i + 1] - y[i])
            if abs(yy) < half:
                return yy
        if edge == "bottom" and (y[i] + half) * (y[i + 1] + half) < 0:
            t = (-half - y[i]) / (y[i + 1] - y[i])
            xx = x[i] + t * (x[i + 1] - x[i])
            if abs(xx) < half:
                return xx
    return None


class LabelPlacer:
    """Puts text on a map, trying a few nearby spots so labels do not sit on each other."""

    def __init__(self, ax, half, side_points):
        self.ax, self.half, self.unit, self.taken = ax, half, 2 * half / side_points, []

    def _box(self, x, y, text, size, ha):
        w, h = 0.70 * size * len(text) * self.unit, 1.3 * size * self.unit
        x0 = x - w / 2 if ha == "center" else (x if ha == "left" else x - w)
        return x0, x0 + w, y - h / 2, y + h / 2

    def reserve(self, x0, x1, y0, y1):
        self.taken.append((x0, x1, y0, y1))

    def place(self, spots, text, size, force=False, **style):
        """`spots` is a list of (x, y, ha) to try in order."""
        edge = 0.97 * self.half
        for x, y, ha in spots:
            x0, x1, y0, y1 = self._box(x, y, text, size, ha)
            fits = x0 > -edge and x1 < edge and y0 > -edge and y1 < edge
            clash = any(x0 < b[1] and x1 > b[0] and y0 < b[3] and y1 > b[2] for b in self.taken)
            if fits and not clash:
                break
        else:
            if not force:
                return False
            for x, y, ha in spots:                         # accept an overlap, but not the map's edge
                x0, x1, y0, y1 = self._box(x, y, text, size, ha)
                if x0 > -edge and x1 < edge and y0 > -edge and y1 < edge:
                    break
            else:
                return False
        self.taken.append((x0, x1, y0, y1))
        self.ax.text(x, y, text, fontsize=size, ha=ha, va="center", clip_on=True, **style)
        return True


def draw_map(ax, sky, ra0, dec0, half, title, stars, mag_limit, name_limit, ra_step, dec_step,
             target, side_points, box=None):
    """One sky map, `half` degrees from centre to edge. North is up and east is left."""
    ax.set_facecolor(SKY)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(GRID)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=8)
    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    reach = 1.6 * half
    labels = LabelPlacer(ax, half, side_points)

    # ---- grid of sky coordinates ----
    for dec in np.arange(-80, 81, dec_step):
        x, y, sep = project(np.linspace(0, 360, 721), np.full(721, dec), ra0, dec0)
        x, y = np.where(sep < 120, x, np.nan), np.where(sep < 120, y, np.nan)
        ax.plot(x, y, color=GRID, lw=0.6, zorder=1)
        at = edge_crossing(x, y, half, "left")
        if at is not None:
            ax.text(-half * 1.015, at, f"{dec:+.0f}°" if dec else "0°", color=MUTED, fontsize=7.5,
                    ha="right", va="center")
    for ra in np.arange(0, 360, ra_step):
        x, y, sep = project(np.full(341, ra), np.linspace(-85, 85, 341), ra0, dec0)
        x, y = np.where(sep < 120, x, np.nan), np.where(sep < 120, y, np.nan)
        ax.plot(x, y, color=GRID, lw=0.6, zorder=1)
        at = edge_crossing(x, y, half, "bottom")
        if at is not None:
            hours = ra / 15.0
            label = f"{int(hours)}h" if abs(hours - round(hours)) < 1e-6 else \
                f"{int(hours)}h{int(round((hours % 1) * 60)):02d}"
            ax.text(at, -half * 1.02, label, color=MUTED, fontsize=7.5, ha="center", va="top")

    # ---- the target, drawn first so that other labels keep clear of it ----
    ring = 0.035 * half if not box else 0.018 * half
    ax.add_patch(Circle((0, 0), ring, fill=False, ec=ACCENT, lw=1.4, zorder=6))
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ax.plot([dx * ring * 1.35, dx * ring * 2.3], [dy * ring * 1.35, dy * ring * 2.3], color=ACCENT,
                lw=1.4, zorder=6)
    labels.reserve(-2.4 * ring, 2.4 * ring, -2.4 * ring, 2.4 * ring)
    plate = dict(boxstyle="round,pad=0.2", fc=SKY, ec="none", alpha=0.7)
    if box:
        ax.add_patch(Rectangle((-box, -box), 2 * box, 2 * box, fill=False, ec=ACCENT, lw=1.1, zorder=6))
        gap = 0.035 * half
        labels.place([(-box, box + gap, "left"), (-box, -box - gap, "left")], "area of the close-up", 8,
                     force=True, color=INK, zorder=7, bbox=plate)
    else:
        labels.place([(ring * 2.8, -ring * 2.8, "left"), (-ring * 2.8, -ring * 2.8, "right")], target, 9,
                     force=True, color=INK, zorder=7, bbox=plate)

    # ---- constellation figures ----
    figures = []                                                    # (id, projected vertices inside the map)
    for con in sky["lines"]:
        inside = []
        for line in con["geometry"]["coordinates"]:
            pts = np.array(line, float)
            x, y, sep = project(pts[:, 0], pts[:, 1], ra0, dec0)
            ok = sep < max(reach, 60)
            for i in range(len(x) - 1):
                if ok[i] and ok[i + 1]:
                    ax.plot(x[i:i + 2], y[i:i + 2], color=FIGURE_LINE, lw=1.0, zorder=2,
                            solid_capstyle="round")
            keep = (np.abs(x) < 0.9 * half) & (np.abs(y) < 0.9 * half) & ok
            inside += list(zip(x[keep], y[keep]))
        figures.append((con["id"], np.array(inside)))

    # ---- stars ----
    x, y, sep = project(stars[:, 0], stars[:, 1], ra0, dec0)
    show = (sep < reach) & (stars[:, 2] <= mag_limit) & (np.abs(x) < half) & (np.abs(y) < half)
    size = 0.6 + 1.7 * (mag_limit - stars[show, 2]) ** 1.7
    temps = 4600 * (1 / (0.92 * stars[show, 3] + 1.7) + 1 / (0.92 * stars[show, 3] + 0.62))
    colours = [glow_colour(t) for t in np.clip(temps, 2800, 12000)]
    ax.scatter(x[show], y[show], s=size, c=colours, linewidths=0, zorder=4)

    # ---- constellation names: the catalogue's own spot if it is on the map, else the middle
    #      of the part of the figure that is ----
    used = set()
    for entry in sky["names"]:
        px, py, sep = project(entry["geometry"]["coordinates"][0], entry["geometry"]["coordinates"][1],
                              ra0, dec0)
        spot = None
        if sep < reach and abs(px) < 0.86 * half and abs(py) < 0.9 * half:
            spot = (float(px), float(py))
        else:
            own = [(i, v) for i, (cid, v) in enumerate(figures)
                   if cid == entry["id"] and i not in used and len(v) >= 2]
            if own:
                # a constellation in two parts (Serpens): take the part nearest this name
                i, v = min(own, key=lambda item: np.hypot(*(item[1].mean(axis=0) - (px, py))))
                used.add(i)
                spot = tuple(v.mean(axis=0))
        if spot:
            step = 1.5 * 8 * labels.unit
            labels.place([(spot[0], spot[1] + k * step, "center") for k in (0, 1, -1, 2, -2, 3, -3)],
                         entry["properties"]["name"].upper(), 8, force=True, color=INK2, alpha=0.8,
                         zorder=3, bbox=dict(boxstyle="round,pad=0.15", fc=SKY, ec="none", alpha=0.55))

    # ---- star names, brightest first, only where there is room ----
    off = 0.02 * half
    for i in np.where(show & (stars[:, 2] <= name_limit))[0][np.argsort(stars[show & (stars[:, 2] <= name_limit), 2])]:
        entry = sky["starnames"].get(str(int(stars[i, 4])), {})
        label = entry.get("name") or (f"{entry['bayer']} {entry.get('c', '')}".strip()
                                      if entry.get("bayer") else "")
        if label:
            labels.place([(x[i] + off, y[i] + off, "left"), (x[i] - off, y[i] + off, "right"),
                          (x[i] + off, y[i] - off, "left"), (x[i] - off, y[i] - off, "right")],
                         label, 7.5, color=INK2, zorder=5)


def where_text(star):
    """Plain-language notes on finding the star."""
    ra, dec = star["ra"], star["dec"]
    h = ra / 15.0
    t = round(h * 36000)                                   # tenths of a second of time
    a = round(abs(dec) * 3600)                             # seconds of arc
    sign = "+" if dec >= 0 else "−"
    lines = [f"RA {t // 36000:02d}h {t % 36000 // 600:02d}m {t % 600 / 10:04.1f}s",
             f"Dec {sign}{a // 3600:02d}° {a % 3600 // 60:02d}′ {a % 60:02d}″"]
    try:
        from astropy.coordinates import SkyCoord, get_constellation
        import astropy.units as u

        lines.insert(0, f"In {get_constellation(SkyCoord(ra * u.deg, dec * u.deg))}")
    except Exception:
        pass
    # the Sun is opposite the star about this long after the March equinox
    day = (80 + ((h - 12) % 24) / 24 * 365.25) % 365.25
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
              "October", "November", "December"]
    starts = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])
    month = int(np.searchsorted(starts, day, side="right") - 1)
    part = ("early", "mid", "late")[min(int((day - starts[month]) // 10.5), 2)]
    lines.append(f"Highest at midnight in {part} {months[month]}")
    if dec >= 0:
        lines.append(f"Rises for anyone north of {90 - dec:.0f}° S")
    else:
        lines.append(f"Rises for anyone south of {90 + dec:.0f}° N")
    if star.get("tmag"):
        lines.append(f"Magnitude {star['tmag']:.1f} (TESS band):")
        lines.append("too faint for the eye" + (" or binoculars;" if star["tmag"] > 9.5 else ";"))
        lines.append("a small telescope shows it" if star["tmag"] > 9.5 else "binoculars show it")
    return lines


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Draw the system described by a joint_orbit.py run.")
    p.add_argument("folder", help="output folder of a joint_orbit.py run")
    p.add_argument("--no-sky", action="store_true", help="leave out the sky maps (no download needed)")
    p.add_argument("--wide", type=float, default=45.0, help="wide map: degrees from centre to edge")
    p.add_argument("--close", type=float, default=10.0, help="close-up: degrees from centre to edge")
    p.add_argument("--out", default=None, help="file to save (default: system_view.png in the folder)")
    args = p.parse_args()

    folder = Path(args.folder)
    run = read_run(folder)
    geo = orbit_geometry(run)
    star = run["star"]
    name = f"TIC {star.get('tic', '')}".strip()

    sky = None
    if not args.no_sky:
        try:
            sky = fetch_sky(Path(__file__).resolve().parent / "sky_data")
        except Exception as exc:
            print(f"  could not get the sky data ({str(exc)[:100]}); drawing without the maps")

    fig = plt.figure(figsize=(16, 12 if sky else 6.6), facecolor=SURFACE)
    top, height = (0.585, 0.30) if sky else (0.17, 0.60)
    fig.text(0.03, 0.972 if sky else 0.95, f"{name}: the most probable picture", color=INK, fontsize=19,
             va="top")
    fig.text(0.03, 0.938 if sky else 0.885,
             f"A {star['teff']:,.0f} K star and a companion on a {geo['period']:.0f}-day orbit"
             + (f" of eccentricity {geo['e']:.2f}" if geo["e"] > 0 else "")
             + ". Most probable values from joint_orbit.py; the ranges are wide.",
             color=INK2, fontsize=11, va="top")

    ax1 = fig.add_axes([0.03, top + 0.035, 0.32, height - 0.035])
    bar = fig.add_axes([0.06, top - 0.03, 0.26, 0.018 if sky else 0.032])
    panel_stars(ax1, bar, run, geo)
    panel_orbit(fig.add_axes([0.385, top - 0.035, 0.33, height + 0.035]), run, geo)
    panel_eclipse(fig.add_axes([0.75, top - 0.035, 0.22, height + 0.035]), run, geo)

    if sky:
        side = 0.445                                      # of the figure's height
        width = side * 12 / 16
        draw_map(fig.add_axes([0.045, 0.045, width, side]), sky, star["ra"], star["dec"], args.wide,
                 "Where it is in the sky", sky["bright"], 5.3, 2.3, 30, 20, name, side * 12 * 72,
                 box=args.close)
        draw_map(fig.add_axes([0.435, 0.045, width, side]), sky, star["ra"], star["dec"], args.close,
                 "Close-up", sky["faint"], 8.0, 4.6, 7.5 if abs(star["dec"]) < 60 else 30, 5, name,
                 side * 12 * 72)
        x = 0.80
        fig.text(x, 0.045 + side + 0.012, "Finding it", color=INK, fontsize=13, va="bottom")
        fig.text(x, 0.045 + side - 0.01, "\n".join(where_text(star)), color=INK2, fontsize=9.5, va="top",
                 linespacing=1.6)
        fig.text(x, 0.045 + side * 0.49, "The numbers", color=INK, fontsize=13, va="bottom")
        facts = [f"Period  {run['period'][0]:.0f} days  ({run['period'][1]:.0f}–{run['period'][2]:.0f})"]
        if geo["e"] > 0:
            facts.append(f"Eccentricity  {run['ecc'][0]:.2f}  ({run['ecc'][1]:.2f}–{run['ecc'][2]:.2f})")
        facts.append(f"Companion mass  {run['mass'][0]:.2f} Suns  ({run['mass'][1]:.2f}–{run['mass'][2]:.2f})")
        facts.append(f"   radius  {run['radius'][0]:.2f} Suns  ({run['radius'][1]:.2f}–{run['radius'][2]:.2f})")
        facts.append(f"Separation  {geo['a'] * (1 - geo['e']):.2f} to {geo['a'] * (1 + geo['e']):.2f} AU"
                     if geo["e"] >= 0.02 else f"Separation  {geo['a']:.2f} AU")
        if run["gaia"].get("parallax"):
            pc = 1000.0 / run["gaia"]["parallax"]
            facts.append(f"Distance  {pc:,.0f} parsecs")
            facts.append(f"   = {round(pc * 3.2616, -1):,.0f} light-years")
        facts += ["", "Brackets: 68% ranges.", "Maps: north up, east left.",
                  f"Wide map {2 * args.wide:.0f}° across,", f"close-up {2 * args.close:.0f}° across."]
        fig.text(x, 0.045 + side * 0.49 - 0.022, "\n".join(facts), color=INK2, fontsize=9.5, va="top",
                 linespacing=1.6)

    out = Path(args.out) if args.out else folder / "system_view.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
