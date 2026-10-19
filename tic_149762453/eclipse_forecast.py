#!/usr/bin/env python
"""
eclipse_forecast.py - when will each star's next eclipses happen, will TESS be
watching, and which of them can you see from your own site?

Give it one or more joint_orbit.py folders. For each star it lists:

1. The TESS sectors still to come that include the star (from the tess-point
   package's pointing plans), with the predicted eclipses that fall inside each.
2. The eclipses you can observe from your site (default Washington, DC): the
   star must be above --min-alt with the Sun below --twilight. Each eclipse is
   rated FULL (the whole eclipse plus --baseline hours either side),
   WHOLE ECLIPSE (all of the eclipse, less baseline), or PARTIAL (ingress,
   egress or the middle only).

Usage
-----
    python eclipse_forecast.py joint_332229223 joint_149762453
    python eclipse_forecast.py joint_* --days 730
    python eclipse_forecast.py joint_149762453 --lat 34.05 --lon -118.24 --tz America/Los_Angeles
    python eclipse_forecast.py joint_149762453 --site 34.05,-118.24 --min-alt 25 --baseline 0.75

Options
-------
--lat/--lon (or --site LAT,LON), --elevation (meters), --tz (time zone name;
default America/New_York, or worked out from the position if the
timezonefinder package is installed), --start (YYYY-MM-DD, default today),
--days (how far ahead, default 365), --min-alt (degrees, default 30),
--twilight (Sun altitude, degrees, default -12), --baseline (hours of
out-of-eclipse data wanted either side, default 1.0), --min-coverage (the
smallest share of the eclipse worth listing, default 0.3), --duration (hours;
overrides the eclipse length worked out from the fit), --out (folder for
the results, default the current folder).

What it reads from each folder
------------------------------
star.json (position), summary.txt (the eclipse time and its uncertainty),
candidate_periods.csv (the period and its uncertainty). The eclipse's length
and depth come from the fitted orbit (system_view.py and system_movie.py must
sit beside this script); if that fails, from the eclipse lengths in
summary.txt. A folder with no period yet (one eclipse only) still gets its
TESS sectors listed.

Times
-----
The ephemeris is in BJD (TDB). Times at your site are converted for the light's
travel time across Earth's orbit (up to 8 minutes either way) and printed in
your local time zone. Each prediction's uncertainty is
sqrt(sigma_T0^2 + (N sigma_P)^2), N being the number of orbits since the
measured eclipse. It is added to the time you need to watch.

Output
------
eclipse_forecast.csv   every listed ground opportunity, all stars, in date order
tess_upcoming.csv      the coming TESS sectors and the eclipses predicted in them
eclipse_forecast.ics   a calendar file (ground opportunities and TESS sectors);
                       open it in Calendar or import it into Google Calendar
and a summary on the screen.

Things to know
--------------
* The pointing plans come from tess-point. Update it now and then
  (pip install -U tess-point): sectors are added as they are announced and can
  be changed. A star "in" a sector can still fall in a gap between CCDs or in a
  sector's mid-point data download gap (about a day).
* Sector start and end dates are estimated as halfway between neighboring
  sectors' midpoints, good to a day or so.
* The eclipse length is the model's from first to last contact. The fitted
  eccentricity is uncertain, so the length is uncertain by several minutes too;
  --baseline covers that.
"""

import argparse
import json
import re
import sys
import warnings
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DC = (38.8977, -77.0365)


# ----------------------------------------------------------------------------
# Reading a joint_orbit.py folder
# ----------------------------------------------------------------------------
def locate_folder(given):
    raw = str(given).strip().strip('"').strip("'")
    path = Path(raw).expanduser()
    if path.is_file():
        path = path.parent
    for place in (path, Path.cwd() / path, HERE / path):
        if place.is_dir():
            return place.resolve()
    sys.exit(f"Could not find the folder '{raw}' (current folder: {Path.cwd()}).")


def read_star(folder):
    """Position, ephemeris and its uncertainties from one folder."""
    out = {"folder": folder}
    path = folder / "star.json"
    if not path.exists():
        sys.exit(f"No star.json in {folder}.")
    star = json.loads(path.read_text(encoding="utf-8-sig"))
    out["tic"] = str(star.get("tic", folder.name))
    out["ra"], out["dec"] = float(star["ra"]), float(star["dec"])
    out["tmag"] = star.get("tmag")
    text = (folder / "summary.txt").read_text(errors="replace") if (folder / "summary.txt").exists() else ""
    out["text"] = text
    found = re.findall(r"KNOWN ECLIPSE in \S+: cent(?:re|er) JD ([\d.]+)", text)
    out["t0"] = float(found[-1]) if found else None
    err = re.findall(r"mid-eclipse uncertainty ([\d.]+) minutes", text)
    out["t0_err"] = float(err[-1]) / 1440 if err else 2.0 / 1440       # a cautious 2 minutes if not given
    out["P"] = out["P_err"] = None
    cand = folder / "candidate_periods.csv"
    if cand.exists():
        c = pd.read_csv(cand).sort_values("probability", ascending=False)
        if len(c):
            out["P"], out["P_err"] = float(c.period_days.iloc[0]), float(c.plus_minus.iloc[0])
            out["P_prob"] = float(c.probability.iloc[0])
            out["n_candidates"] = int((c.probability > 0.01).sum())
    return out


def eclipse_shape(star):
    """Half-length (days), depth and the companion's own eclipse, from the fitted orbit."""
    shape = {"half": None, "depth": None, "secondary": None, "source": None}
    if star["P"] is not None and star["t0"] is not None:
        try:
            sys.path.insert(0, str(HERE))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import system_movie as sm
                import system_view as sv
                run = sv.read_run(star["folder"])
                run["period"][0] = star["P"]
                geo = sv.orbit_geometry(run)
                orbit = sm.Orbit(run, geo, star["P"], star["t0"])
            tt = np.linspace(-1.0, 1.0, 8001)
            model = orbit.brightness(star["t0"] + tt)
            near = np.abs(tt) < 0.5
            lost = np.clip(1.0 - model, 0, None) * near
            shift = float(np.sum(tt * lost) / lost.sum()) if lost.sum() > 0 else 0.0
            model = orbit.brightness(star["t0"] + shift + tt)
            mid = len(tt) // 2
            dipping = model < 0.9995
            if dipping[mid]:
                lo = mid - int(np.argmin(dipping[mid::-1])) + 1
                hi = mid + int(np.argmin(dipping[mid:])) - 1
                shape.update(half=0.5 * (tt[hi] - tt[lo]), depth=1.0 - float(model[mid]),
                             source="the fitted orbit")
                # the companion's own eclipse, somewhere else in the orbit
                ph = np.linspace(0, star["P"], 40001)
                full = orbit.brightness(star["t0"] + shift + ph)
                away = (ph > 3 * shape["half"]) & (ph < star["P"] - 3 * shape["half"])
                if away.any() and (1 - full[away].min()) > 1e-4:
                    j = np.flatnonzero(away)[np.argmin(full[away])]
                    shape["secondary"] = (float(ph[j]), float(1 - full[j]), float(geo["e"]))
        except Exception as exc:                       # noqa: BLE001 - fall back below
            shape["error"] = f"{type(exc).__name__}: {exc}"
    if shape["half"] is None:
        m = re.findall(r"eclipse lengths from ([\d.]+) to ([\d.]+) hours", star["text"])
        if m:
            a, b = (float(v) for v in m[-1])
            shape.update(half=0.25 * (a + b) / 24, source="summary.txt's eclipse lengths")
    if shape["depth"] is None:
        m = re.findall(r"KNOWN ECLIPSE in \S+: .*?deepest about ([\d.]+)%", star["text"])
        if m:
            shape["depth"] = float(m[-1]) / 100
    return shape


# ----------------------------------------------------------------------------
# TESS
# ----------------------------------------------------------------------------
def tess_sectors(ra, dec, start_jd):
    """Coming sectors that include the star: (sector, camera, ccd, begin JD, end JD)."""
    try:
        from tess_stars2px import TESS_Spacecraft_Pointing_Data, tess_stars2px_function_entry
    except ImportError:
        return None
    pdata = TESS_Spacecraft_Pointing_Data()
    mids = {}
    for s, m in zip(pdata.sectors, pdata.midtimes):
        s, m = int(s), float(m)
        if s < 1000:
            mids[s] = m + 2457000.0 if m < 1e5 else m
    out = tess_stars2px_function_entry(np.array([0]), np.array([ra]), np.array([dec]))
    rows = []
    for s, cam, ccd in zip(out[3], out[4], out[5]):
        s = int(s)
        if s not in mids:
            continue
        m = mids[s]
        begin = 0.5 * (mids[s - 1] + m) if s - 1 in mids and 0 < m - mids[s - 1] < 40 else m - 13.7
        end = 0.5 * (m + mids[s + 1]) if s + 1 in mids and 0 < mids[s + 1] - m < 40 else m + 13.7
        if end > start_jd:
            rows.append((s, int(cam), int(ccd), begin, end))
    latest = max(mids.values())
    return sorted(set(rows)), latest


# ----------------------------------------------------------------------------
# Time conversions
# ----------------------------------------------------------------------------
def bjd_to_utc(bjd, coord, site):
    """BJD (TDB) at the solar system's barycenter -> UTC Time at the site."""
    from astropy.time import Time
    t = Time(np.atleast_1d(bjd), format="jd", scale="tdb", location=site)
    est = t - t.light_travel_time(coord, "barycentric")
    est = t - est.light_travel_time(coord, "barycentric")
    return est.utc


def with_error(value, err, least=4):
    """'value +/- err', both to the decimal place of err's second significant figure."""
    if not err or not np.isfinite(err) or err <= 0:
        return f"{value:.{least}f}"
    places = int(min(max(least, 1 - np.floor(np.log10(err))), 8))
    return f"{value:.{places}f} +/- {err:.{places}f}"


def jd_to_date(jd):
    from astropy.time import Time
    return Time(jd, format="jd").iso[:10]


# ----------------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------------
def ground_events(star, shape, site, zone, args, start_jd, end_jd):
    """Every predicted eclipse in the window, with how much of it can be watched."""
    import astropy.units as u
    from astropy.coordinates import AltAz, SkyCoord, get_body
    from astropy.time import TimeDelta

    P, t0, half = star["P"], star["t0"], shape["half"]
    base = args.baseline / 24
    coord = SkyCoord(star["ra"] * u.deg, star["dec"] * u.deg)
    k = np.arange(np.ceil((start_jd - t0) / P), np.floor((end_jd - t0) / P) + 1).astype(int)
    if len(k) == 0:
        return []
    mids_bjd = t0 + k * P
    sig = np.sqrt(star["t0_err"] ** 2 + (k * star["P_err"]) ** 2)          # days
    mids = bjd_to_utc(mids_bjd, coord, site)

    step = 3 / 1440
    reach = half + base + 2 * sig.max()
    offsets = np.arange(-reach, reach + step / 2, step)
    grid = mids[:, None] + TimeDelta(offsets[None, :], format="jd")
    flat = grid.ravel()
    frame = AltAz(obstime=flat, location=site)
    alt = coord.transform_to(frame).alt.deg.reshape(grid.shape)
    sun = get_body("sun", flat, location=site).transform_to(frame).alt.deg.reshape(grid.shape)
    ok = (alt > args.min_alt) & (sun < args.twilight)

    t_mid = mids
    moon = get_body("moon", t_mid, location=site)
    sun_mid = get_body("sun", t_mid, location=site)
    illum = 0.5 * (1 - np.cos(sun_mid.separation(moon).rad))
    moon_sep = moon.separation(coord).deg
    moon_alt = moon.transform_to(AltAz(obstime=t_mid, location=site)).alt.deg

    def local(t):
        return t.to_datetime().replace(tzinfo=timezone.utc).astimezone(zone)

    events = []
    for i in range(len(k)):
        s = sig[i]
        need_e = np.abs(offsets) <= half + 2 * s                  # the eclipse itself, with timing slack
        need_b = np.abs(offsets) <= half + base + 2 * s           # ... and baseline
        inside = np.abs(offsets) <= half
        cover = ok[i][inside].mean() if inside.any() else 0.0
        if cover < args.min_coverage:
            continue
        if ok[i][need_b].all():
            rating = "FULL"
        elif ok[i][need_e].all():
            # how much baseline is there on each side?
            pre = offsets[(offsets < -half - 2 * s) & (offsets >= -reach)]
            okp = ok[i][(offsets < -half - 2 * s)]
            before = step * np.argmin(okp[::-1]) if not okp.all() else len(pre) * step
            okq = ok[i][(offsets > half + 2 * s)]
            after = step * np.argmin(okq) if not okq.all() else okq.size * step
            rating = f"WHOLE ECLIPSE ({before * 24:.1f} h before, {after * 24:.1f} h after)"
        else:
            parts = []
            if ok[i][np.argmin(np.abs(offsets + half))]:
                parts.append("ingress")
            if ok[i][np.argmin(np.abs(offsets))]:
                parts.append("middle")
            if ok[i][np.argmin(np.abs(offsets - half))]:
                parts.append("egress")
            rating = "PARTIAL (" + (" + ".join(parts) if parts else "part of the eclipse") + ")"
        watch = np.flatnonzero(ok[i])
        mid_t = mids[i]
        ingress = mid_t - TimeDelta(half, format="jd")
        egress = mid_t + TimeDelta(half, format="jd")
        w_from = mids[i] + TimeDelta(offsets[watch[0]], format="jd")
        w_to = mids[i] + TimeDelta(offsets[watch[-1]], format="jd")
        in_ecl = np.abs(offsets) <= half
        events.append({
            "tic": star["tic"], "orbit": int(k[i]), "rating": rating,
            "local_date": local(mid_t).strftime("%a %Y-%m-%d"),
            "ingress_local": local(ingress).strftime("%H:%M"),
            "mid_local": local(mid_t).strftime("%H:%M"),
            "egress_local": local(egress).strftime("%H:%M"),
            "tz": local(mid_t).strftime("%Z"),
            "plus_minus_min": round(float(s * 1440), 1),
            "plus_minus_days": float(f"{s:.2g}"),
            "coverage_pct": round(100 * float(cover)),
            "observable_from_local": local(w_from).strftime("%H:%M"),
            "observable_to_local": local(w_to).strftime("%H:%M"),
            "alt_mid_deg": round(float(alt[i][np.argmin(np.abs(offsets))]), 1),
            "alt_min_in_eclipse_deg": round(float(alt[i][in_ecl].min()), 1),
            "moon_illum_pct": round(100 * float(illum[i])),
            "moon_sep_deg": round(float(moon_sep[i])),
            "moon_up": bool(moon_alt[i] > 0),
            "mid_bjd_tdb": float(mids_bjd[i]),
            "mid_utc": mid_t.iso[:16],
            "ingress_utc_jd": float(ingress.jd), "egress_utc_jd": float(egress.jd),
            "depth_pct": round(100 * shape["depth"], 1) if shape["depth"] else None,
            "hours": round(48 * half, 2),
        })
    return events


# ----------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------
def ics_time(jd):
    from astropy.time import Time
    return Time(jd, format="jd").to_datetime().strftime("%Y%m%dT%H%M%SZ")


def ics_text(text):
    """Escape a calendar text value."""
    return str(text).replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def write_ics(path, ground, tess_rows, baseline):
    from astropy.time import Time
    now = Time.now().to_datetime().strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//eclipse_forecast.py//EN", "CALSCALE:GREGORIAN"]
    for e in ground:
        desc = (f"{e['rating']}. Mid-eclipse {e['mid_local']} {e['tz']} (+/- {e['plus_minus_min']} min), "
                f"ingress {e['ingress_local']}, egress {e['egress_local']}. "
                f"Depth about {e['depth_pct']}%. Altitude at mid-eclipse {e['alt_mid_deg']} deg. "
                f"Moon {e['moon_illum_pct']}% lit, {e['moon_sep_deg']} deg away. "
                f"Mid-eclipse BJD_TDB {with_error(e['mid_bjd_tdb'], e['plus_minus_days'])}.")
        lines += ["BEGIN:VEVENT", f"UID:tic{e['tic']}-orbit{e['orbit']}@eclipse-forecast", f"DTSTAMP:{now}",
                  f"DTSTART:{ics_time(e['ingress_utc_jd'] - baseline / 24)}",
                  f"DTEND:{ics_time(e['egress_utc_jd'] + baseline / 24)}",
                  f"SUMMARY:TIC {e['tic']} eclipse - {e['rating'].split(' (')[0]}",
                  f"DESCRIPTION:{ics_text(desc)}", "END:VEVENT"]
    for r in tess_rows:
        d0 = Time(r["begin_jd"], format="jd").to_datetime().strftime("%Y%m%d")
        d1 = Time(r["end_jd"] + 1, format="jd").to_datetime().strftime("%Y%m%d")
        lines += ["BEGIN:VEVENT", f"UID:tic{r['tic']}-s{r['sector']}@eclipse-forecast", f"DTSTAMP:{now}",
                  f"DTSTART;VALUE=DATE:{d0}", f"DTEND;VALUE=DATE:{d1}",
                  f"SUMMARY:TESS sector {r['sector']} observes TIC {r['tic']}",
                  "DESCRIPTION:" + ics_text(f"Camera {r['camera']}, CCD {r['ccd']}. Predicted eclipses: "
                                            f"{r['eclipses'] or 'none'}. Dates estimated to about a day."),
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    # calendar lines should not exceed 75 octets; fold long ones
    folded = []
    for line in lines:
        while len(line.encode()) > 74:
            folded.append(line[:74])
            line = " " + line[74:]
        folded.append(line)
    path.write_text("\r\n".join(folded) + "\r\n")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Coming TESS sectors and ground eclipse opportunities.")
    p.add_argument("folders", nargs="+", help="joint_orbit.py output folders")
    p.add_argument("--lat", type=float, default=None, help=f"site latitude, degrees (default {DC[0]}, Washington DC)")
    p.add_argument("--lon", type=float, default=None, help=f"site longitude, degrees east (default {DC[1]})")
    p.add_argument("--site", default=None, help="LAT,LON in one go, e.g. 38.90,-77.04")
    p.add_argument("--elevation", type=float, default=20.0, help="meters (default 20)")
    p.add_argument("--tz", default=None, help="time zone, e.g. America/Chicago")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default today)")
    p.add_argument("--days", type=float, default=365.0, help="how far ahead to look (default 365)")
    p.add_argument("--min-alt", type=float, default=30.0, help="lowest useful altitude, degrees (default 30)")
    p.add_argument("--twilight", type=float, default=-12.0, help="Sun must be below this altitude (default -12)")
    p.add_argument("--baseline", type=float, default=1.0, help="hours wanted either side of the eclipse (default 1)")
    p.add_argument("--min-coverage", type=float, default=0.3, help="smallest share of the eclipse to list (default 0.3)")
    p.add_argument("--duration", type=float, default=None, help="eclipse length in hours (overrides the fit)")
    p.add_argument("--out", default=".", help="folder for the output files (default: current folder)")
    args = p.parse_args()

    import astropy.units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time, TimeDelta
    from astropy.utils import iers
    from astropy.utils.exceptions import AstropyWarning
    from zoneinfo import ZoneInfo
    iers.conf.auto_download = False
    warnings.simplefilter("ignore", AstropyWarning)
    warnings.filterwarnings("ignore", message=".*(dubious year|ERFA|leap second|IERS).*")

    # ---- the site ------------------------------------------------------------------------------
    if args.site:
        try:
            args.lat, args.lon = (float(v) for v in args.site.replace(" ", "").split(","))
        except ValueError:
            sys.exit("--site should look like 38.90,-77.04")
    lat = DC[0] if args.lat is None else args.lat
    lon = DC[1] if args.lon is None else args.lon
    tz_name = args.tz
    if tz_name is None:
        if args.lat is None and args.lon is None:
            tz_name = "America/New_York"
        else:
            try:
                from timezonefinder import TimezoneFinder
                tz_name = TimezoneFinder().timezone_at(lat=lat, lng=lon)
            except ImportError:
                tz_name = None
            if not tz_name:
                tz_name = "UTC"
                print("Note: times are in UTC. Give --tz (e.g. America/Denver) for local time, "
                      "or pip install timezonefinder to have it worked out.")
    zone = ZoneInfo(tz_name)
    site = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=args.elevation * u.m)
    start = Time(args.start) if args.start else Time(Time.now().iso[:10])
    end = start + TimeDelta(args.days, format="jd")
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Site: {lat:.4f}, {lon:.4f} ({tz_name}); {start.iso[:10]} to {end.iso[:10]}")
    print(f"Watchable: star above {args.min_alt:g} deg, Sun below {args.twilight:g} deg; "
          f"FULL means the eclipse plus {args.baseline:g} h either side.\n")

    all_ground, all_tess = [], []
    latest_plan = None
    for given in args.folders:
        star = read_star(locate_folder(given))
        shape = eclipse_shape(star)
        if args.duration:
            shape.update(half=args.duration / 48, source="--duration")
        head = f"TIC {star['tic']}  (RA {star['ra']:.4f}, Dec {star['dec']:+.4f}"
        head += f", T = {star['tmag']:.2f})" if star.get("tmag") else ")"
        print("=" * len(head) + "\n" + head + "\n" + "=" * len(head))
        if star["P"] is not None and star["t0"] is not None:
            print(f"Ephemeris: mid-eclipse BJD_TDB {with_error(star['t0'], star['t0_err'])} "
                  f"+ N x {with_error(star['P'], star['P_err'], 6)} d")
            print(f"           (+/- {star['t0_err'] * 1440:.1f} min on the epoch, "
                  f"+/- {star['P_err'] * 1440:.2f} min per orbit)")
            if star.get("n_candidates", 1) > 1:
                print(f"  NOTE: {star['n_candidates']} periods are still open; this uses the most probable "
                      f"({100 * star['P_prob']:.0f}%).")
            if shape["half"]:
                d = f", depth about {100 * shape['depth']:.1f}%" if shape["depth"] else ""
                print(f"Eclipse: {48 * shape['half']:.2f} h first to last contact{d} (from {shape['source']})")
            if shape["secondary"]:
                off, dep, e = shape["secondary"]
                print(f"  The companion's own eclipse, in the representative orbit (e = {e:.2f}), would come "
                      f"{off:.2f} d after each eclipse,\n  about {100 * dep:.2f}% deep. Its time depends on e and "
                      "the orbit's orientation, so it is uncertain by a day or more.")
            elif shape["half"] and shape["source"] == "the fitted orbit":
                print("  In the representative orbit the companion's own eclipse misses (the stars do not "
                      "overlap\n  half an orbit later), so there is no second eclipse to plan for.")
        else:
            print("No period yet (one eclipse only), so no eclipses can be predicted; TESS sectors only.")

        # ---- TESS ----------------------------------------------------------------------------------
        got = tess_sectors(star["ra"], star["dec"], start.jd)
        print("\nTESS, coming sectors:")
        if got is None:
            print("  (install tess-point to see these: pip install tess-point)")
        else:
            rows, latest_plan = got
            if not rows:
                print(f"  none planned (the plans run to about {jd_to_date(latest_plan)})")
            for s, cam, ccd, b, e_ in rows:
                ecl = []
                if star["P"] is not None and star["t0"] is not None:
                    for kk in range(int(np.ceil((b - star["t0"]) / star["P"])),
                                    int(np.floor((e_ - star["t0"]) / star["P"])) + 1):
                        when = star["t0"] + kk * star["P"]
                        sg = np.hypot(star["t0_err"], kk * star["P_err"]) * 1440
                        ecl.append(f"BJD {with_error(when, sg / 1440)} = {Time(when, format='jd').iso[:16]} "
                                   f"(+/- {sg:.1f} min)")
                now_flag = "  <- under way" if b <= start.jd else ""
                print(f"  Sector {s}: {jd_to_date(b)} to {jd_to_date(e_)}, camera {cam}, CCD {ccd}{now_flag}")
                if star["P"] is not None:
                    print("      " + ("eclipses: " + "; ".join(ecl) if ecl else "no eclipse predicted in this sector"))
                all_tess.append({"tic": star["tic"], "sector": s, "camera": cam, "ccd": ccd,
                                 "begin_jd": b, "end_jd": e_, "begin": jd_to_date(b), "end": jd_to_date(e_),
                                 "eclipses": "; ".join(ecl)})

        # ---- the ground --------------------------------------------------------------------------------
        if star["P"] is None or star["t0"] is None:
            print()
            continue
        if not shape["half"]:
            print("\nCould not work out the eclipse's length; give it with --duration (hours).\n")
            continue
        events = ground_events(star, shape, site, zone, args, start.jd, end.jd)
        n_total = int(np.floor((end.jd - star["t0"]) / star["P"]) - np.ceil((start.jd - star["t0"]) / star["P"]) + 1)
        print(f"\nFrom your site: {len(events)} of the {n_total} eclipses in this period can be watched "
              f"(at least {100 * args.min_coverage:.0f}% of the eclipse)")
        for ev in events:
            moon = (f"Moon {ev['moon_illum_pct']}% lit, {ev['moon_sep_deg']} deg away"
                    + ("" if ev["moon_up"] else " (down)"))
            print(f"  {ev['local_date']}  mid {ev['mid_local']} {ev['tz']} (+/- {ev['plus_minus_min']} min)  "
                  f"{ev['rating']}")
            print(f"      BJD_TDB {with_error(ev['mid_bjd_tdb'], ev['plus_minus_days'])}")
            print(f"      ingress {ev['ingress_local']}, egress {ev['egress_local']}; watchable "
                  f"{ev['observable_from_local']}-{ev['observable_to_local']}; altitude {ev['alt_mid_deg']:.0f} deg "
                  f"at mid (lowest {ev['alt_min_in_eclipse_deg']:.0f}); {moon}")
        print()
        all_ground += events

    # ---- files --------------------------------------------------------------------------------------
    if all_ground:
        g = pd.DataFrame(all_ground).sort_values("ingress_utc_jd")
        g.drop(columns=["ingress_utc_jd", "egress_utc_jd"]).to_csv(out_dir / "eclipse_forecast.csv", index=False)
        if len(args.folders) > 1:
            print("ALL STARS, BY DATE")
            for r in g.itertuples():
                print(f"  {r.local_date}  {r.mid_local} {r.tz}  TIC {r.tic}  {r.rating.split(' (')[0]}"
                      f"  (alt {r.alt_mid_deg:.0f} deg, depth {r.depth_pct}%)")
            print()
        print(f"Saved {out_dir / 'eclipse_forecast.csv'}")
    if all_tess:
        pd.DataFrame(all_tess).to_csv(out_dir / "tess_upcoming.csv", index=False)
        print(f"Saved {out_dir / 'tess_upcoming.csv'}")
    if all_ground or all_tess:
        write_ics(out_dir / "eclipse_forecast.ics",
                  sorted(all_ground, key=lambda e: e["ingress_utc_jd"]), all_tess, args.baseline)
        print(f"Saved {out_dir / 'eclipse_forecast.ics'} (open it to add the events to your calendar)")
    if latest_plan is not None:
        print(f"TESS plans in tess-point run to about {jd_to_date(latest_plan)}; "
              "pip install -U tess-point picks up newly announced sectors.")


if __name__ == "__main__":
    main()
