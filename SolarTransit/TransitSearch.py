"""
Search for aircraft which crossed the apparent position of the Sun.

Every ADS-B track in the region is interpolated to a fine time grid, converted into apparent
horizontal coordinates as seen from the observing site, and compared with the apparent position
of the Sun. The candidates are ranked by the minimum angular separation, and the uncertainty of
each match is estimated from the interpolation error, the unknown difference between the
barometric and the geometric altitude, and the accuracy of the site elevation.

"""

# The MIT License

# Copyright (c) 2026 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function, division, absolute_import, unicode_literals

import os
import datetime

import numpy as np
import pandas as pd

from SolarTransit.ADSBData import loadRegion
from SolarTransit.Config import (SITE_LAT, SITE_LON, NOMINAL_TIME, SEARCH_HALF_WINDOW,
    SEARCH_TIME_STEP, MAX_SEPARATION_DEG, BARO_ALTITUDE_SIGMA, ADSB_POSITION_SIGMA,
    SITE_ELEVATION_SIGMA, MIN_POINTS_PER_FLIGHT, MAX_USABLE_GAP, RESULTS_DIR, SUBSET_DATA_DIR,
    siteElevation)
from SolarTransit.Conversions import latLonAlt2ECEF, ECEF2AltAz, angularSeparation, slantRange
from SolarTransit.Ephemeris import siteLocation, eclipseCircumstances
from SolarTransit.Refraction import RefractionTable
from SolarTransit.Trajectory import buildTracks


### CONSTANTS ###

# Time margin used when selecting the reports of a flight, so that the interpolation covers the
# whole search window (s)
TRACK_TIME_MARGIN = 300.0

# Time step at which the ephemeris is computed, and then interpolated to the search grid (s)
EPHEMERIS_TIME_STEP = 1.0

# Estimated residual error of the refraction model (deg). The refraction itself is about 0.12 deg
# at an elevation of 6 deg, and the difference between a standard and a real atmosphere is on the
# order of ten per cent of that.
REFRACTION_MODEL_SIGMA = 0.015

### ###



def sunEphemerisOnGrid(times_grid, location, refraction_table):
    """ Compute the apparent position of the Sun and the Moon on the search time grid.

    The ephemeris is computed on a coarse grid and interpolated onto the fine search grid, which
    is much faster and introduces an error far below a milliarcsecond, because the apparent motion
    of the Sun is smooth and slow.

    Arguments:
        times_grid: [ndarray of datetime] Times of the search grid.
        location: [EarthLocation] Location of the observing site.
        refraction_table: [RefractionTable] Table used to apply the refraction.

    Return:
        eph: [dict] Apparent azimuth and altitude of the Sun and the Moon, their angular radii and
            the obscuration of the solar disk, all on the search grid.

    """

    t_beg = times_grid[0]
    t_end = times_grid[-1]

    n_coarse = int((t_end - t_beg).total_seconds()/EPHEMERIS_TIME_STEP) + 2

    times_coarse = np.array([t_beg + datetime.timedelta(seconds=i*EPHEMERIS_TIME_STEP)
        for i in range(n_coarse)])

    circ = eclipseCircumstances(times_coarse, location)

    # Seconds since the beginning of the window
    t_coarse = np.array([(t - t_beg).total_seconds() for t in times_coarse])
    t_fine = np.array([(t - t_beg).total_seconds() for t in times_grid])

    eph = {}

    for key in ['sun_azim', 'sun_alt', 'sun_radius', 'moon_azim', 'moon_alt', 'moon_radius',
        'obscuration']:

        eph[key] = np.interp(t_fine, t_coarse, circ[key])


    # Apply the refraction to the geometric altitudes
    eph['sun_alt_apparent'] = refraction_table.apparentElevationAstronomical(eph['sun_alt'])
    eph['moon_alt_apparent'] = refraction_table.apparentElevationAstronomical(eph['moon_alt'])

    return eph



def apparentTrack(track, t_rel, site_ecef, site_elevation_msl, refraction_table):
    """ Compute the apparent horizontal coordinates of an aircraft as seen from the site.

    The reported barometric altitude is converted into a geometric height, the position is
    transformed into the horizontal frame, and the refraction of a target at a finite distance is
    applied.

    Arguments:
        track: [FlightTrack] Interpolated track of the flight.
        t_rel: [ndarray] Times at which the track is evaluated, in seconds relative to the
            reference time of the track.
        site_ecef: [tuple] ECEF coordinates of the observer (m).
        site_elevation_msl: [float] Height of the observer above sea level (m).
        refraction_table: [RefractionTable] Table used to apply the refraction.

    Return:
        (t_rel, azim, alt_apparent, ecef): [tuple] Times which are inside the span of the reports
            and above the horizon, the apparent azimuth and altitude (deg), and the ECEF
            coordinates of the aircraft at those times.

    """

    x, y, z = track.positionsECEF(t_rel)

    # Times outside the span of the reports give NaN
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    t_rel = t_rel[valid]
    x, y, z = x[valid], y[valid], z[valid]

    if len(t_rel) == 0:
        return t_rel, np.array([]), np.array([]), (x, y, z)


    # Geometric horizontal coordinates of the aircraft
    azim, alt_geometric = ECEF2AltAz(site_ecef, (x, y, z))

    # Only the part of the track above the horizon can be seen
    above_horizon = alt_geometric > 0

    t_rel = t_rel[above_horizon]
    x, y, z = x[above_horizon], y[above_horizon], z[above_horizon]
    azim = azim[above_horizon]
    alt_geometric = alt_geometric[above_horizon]

    if len(t_rel) == 0:
        return t_rel, azim, alt_geometric, (x, y, z)


    # Height of the aircraft above sea level, needed for the refraction
    r_site = np.sqrt(site_ecef[0]**2 + site_ecef[1]**2 + site_ecef[2]**2)
    height = np.sqrt(x**2 + y**2 + z**2) - r_site + site_elevation_msl

    # An aircraft is inside the atmosphere, so its light is refracted less than the light of an
    # astronomical source at the same elevation
    alt_apparent = refraction_table.apparentElevationTarget(alt_geometric, height)

    return t_rel, azim, alt_apparent, (x, y, z)



def searchTransits(df, location, site_ecef, site_elevation_msl, refraction_table, times_grid,
    max_separation=MAX_SEPARATION_DEG, min_points=MIN_POINTS_PER_FLIGHT):
    """ Find all aircraft which passed close to the apparent position of the Sun.

    Arguments:
        df: [pandas.DataFrame] ADS-B reports in the region.
        location: [EarthLocation] Location of the observing site.
        site_ecef: [tuple] ECEF coordinates of the observer (m).
        site_elevation_msl: [float] Height of the observer above sea level (m).
        refraction_table: [RefractionTable] Table used to apply the refraction.
        times_grid: [ndarray of datetime] Times of the search grid.

    Keyword arguments:
        max_separation: [float] Report the aircraft which came closer than this to the centre of
            the Sun (deg).
        min_points: [int] Flights with fewer reports than this are skipped.

    Return:
        (candidates, eph, tracks): [tuple] List of dictionaries describing the candidates, the
            ephemeris on the search grid, and the list of the tracks which were searched.

    """

    time_ref = times_grid[0]

    # Relative times of the search grid
    t_grid = np.array([(t - time_ref).total_seconds() for t in times_grid])

    print("Computing the ephemeris...")
    eph = sunEphemerisOnGrid(times_grid, location, refraction_table)

    print("Building the flight tracks...")
    tracks = buildTracks(df, time_ref, min_points=min_points)

    print("    {:d} tracks with at least {:d} reports".format(len(tracks), min_points))

    print("Searching for transits...")

    candidates = []

    for track in tracks:

        # Only search inside the span of the reports of the flight
        mask = (t_grid >= track.t_beg) & (t_grid <= track.t_end)

        if not np.any(mask):
            continue

        # Apparent position of the aircraft on the sky
        t_sel, azim, alt_apparent, (x, y, z) = apparentTrack(track, t_grid[mask], site_ecef,
            site_elevation_msl, refraction_table)

        if len(t_sel) == 0:
            continue

        # Angular separation from the centre of the apparent solar disk
        idx_grid = np.searchsorted(t_grid, t_sel)
        idx_grid = np.clip(idx_grid, 0, len(t_grid) - 1)

        sep = angularSeparation(azim, alt_apparent, eph['sun_azim'][idx_grid],
            eph['sun_alt_apparent'][idx_grid])

        i_min = int(np.argmin(sep))

        if sep[i_min] > max_separation:
            continue


        # Distance from the observer to the aircraft at the moment of the closest approach
        dist = float(slantRange(site_ecef, (x[i_min], y[i_min], z[i_min])))

        # Uncertainty of the match. All terms are converted into an angle at the distance of the
        # aircraft.
        cos_alt = np.cos(np.radians(alt_apparent[i_min]))

        err_interp_rms, err_interp_max = track.interpolationError()

        sigma_terms = {
            'baro': np.degrees(BARO_ALTITUDE_SIGMA*cos_alt/dist),
            'site_elevation': np.degrees(SITE_ELEVATION_SIGMA*cos_alt/dist),
            'adsb_position': np.degrees(ADSB_POSITION_SIGMA/dist),
            'interpolation': np.degrees((err_interp_rms if np.isfinite(err_interp_rms) else 0.0)
                /dist),
            'refraction': REFRACTION_MODEL_SIGMA,
            }

        sigma_total = np.sqrt(sum(v**2 for v in sigma_terms.values()))

        gap, cadence_median = track.sampleCadence(t_sel[i_min])

        # Angular size of the silhouette of the aircraft
        ang_size = np.degrees(track.wingspan/dist)*3600

        # Time spent inside the solar disk
        sun_radius = eph['sun_radius'][idx_grid][i_min]
        inside = sep < sun_radius
        transit_duration = float(np.sum(inside))*SEARCH_TIME_STEP

        candidates.append({
            'callsign': track.callsign,
            'flight_number': track.flight_number,
            'icao_address': track.icao_address,
            'tail_number': track.tail_number,
            'aircraft_type_icao': track.aircraft_type,
            'airline_iata': track.airline,
            'departure_airport_icao': track.departure,
            'arrival_airport_icao': track.arrival,
            'time_min_sep': time_ref + datetime.timedelta(seconds=float(t_sel[i_min])),
            'min_sep_deg': float(sep[i_min]),
            'sigma_deg': float(sigma_total),
            'sun_radius_deg': float(sun_radius),
            'inside_disk': bool(sep[i_min] < sun_radius),
            'transit_duration_s': transit_duration,
            'apparent_alt_deg': float(alt_apparent[i_min]),
            'apparent_azim_deg': float(azim[i_min]),
            'sun_alt_deg': float(eph['sun_alt_apparent'][idx_grid][i_min]),
            'sun_azim_deg': float(eph['sun_azim'][idx_grid][i_min]),
            'obscuration': float(eph['obscuration'][idx_grid][i_min]),
            'altitude_baro_ft': float(track.baroAltitude(t_sel[i_min])),
            'slant_range_km': dist/1000,
            'sample_gap_s': gap,
            'cadence_median_s': cadence_median,
            'interp_err_rms_m': err_interp_rms,
            'interp_err_max_m': err_interp_max,
            'wingspan_m': track.wingspan,
            'ang_size_arcsec': float(ang_size),
            'n_reports': int(len(track.t_data)),
            'reliable_gap': bool(np.isfinite(gap) and (gap <= MAX_USABLE_GAP)),
            'flight_id': track.flight_id,
            'sigma_baro_deg': float(sigma_terms['baro']),
            'sigma_interp_deg': float(sigma_terms['interpolation']),
            })


    # Rank the candidates by the minimum separation
    candidates = sorted(candidates, key=lambda c: c['min_sep_deg'])

    return candidates, eph, tracks



def printCandidates(candidates, n_print=10):
    """ Print a summary table of the candidates.

    Arguments:
        candidates: [list of dict] Ranked candidates.

    Keyword arguments:
        n_print: [int] Number of candidates which are printed.

    """

    print()
    print("{:<4s} {:<9s} {:<8s} {:<6s} {:<12s} {:>9s} {:>8s} {:>7s} {:>8s} {:>7s} {:>6s}".format(
        "Rank", "Callsign", "Reg", "Type", "Time (UTC)", "Sep (deg)", "+/-", "FL", "Dist km",
        "Gap s", "Disk"))

    print("-"*104)

    for i, cand in enumerate(candidates[:n_print]):

        print("{:<4d} {:<9s} {:<8s} {:<6s} {:<12s} {:9.4f} {:8.4f} {:7.0f} {:8.1f} {:7.1f} "
            "{:>6s}".format(
            i + 1,
            str(cand['callsign'])[:9],
            str(cand['tail_number'])[:8],
            str(cand['aircraft_type_icao'])[:6],
            cand['time_min_sep'].strftime("%H:%M:%S.%f")[:12],
            cand['min_sep_deg'],
            cand['sigma_deg'],
            cand['altitude_baro_ft']/100,
            cand['slant_range_km'],
            cand['sample_gap_s'],
            "yes" if cand['inside_disk'] else "no",
            ))



if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS ###

    arg_parser = argparse.ArgumentParser(description="""Find the aircraft which crossed the
        apparent position of the Sun, as seen from the observing site. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-i', '--input', metavar='PARQUET_PATH', type=str,
        default=os.path.join(SUBSET_DATA_DIR, 'adsb_region.parquet'),
        help="Path of the regional ADS-B subset. It is downloaded if it does not exist.")

    arg_parser.add_argument('-w', '--window', metavar='MINUTES', type=float,
        default=SEARCH_HALF_WINDOW.total_seconds()/60,
        help="Half width of the search window around the nominal time.")

    arg_parser.add_argument('-s', '--sep', metavar='DEGREES', type=float,
        default=MAX_SEPARATION_DEG,
        help="Report the aircraft which came closer than this to the centre of the Sun.")

    arg_parser.add_argument('-n', '--nprint', metavar='COUNT', type=int, default=10,
        help="Number of candidates which are printed.")

    cml_args = arg_parser.parse_args()

    #########################


    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)


    half_window = datetime.timedelta(minutes=cml_args.window)

    t_beg = NOMINAL_TIME - half_window
    t_end = NOMINAL_TIME + half_window

    print("Searching from {:s} to {:s} UTC".format(str(t_beg), str(t_end)))


    ### Load the data ###

    df = loadRegion(t_beg - datetime.timedelta(seconds=TRACK_TIME_MARGIN),
        t_end + datetime.timedelta(seconds=TRACK_TIME_MARGIN), cache_path=cml_args.input)

    ### ###


    ### Set up the site ###

    elevation_wgs84 = siteElevation()

    location = siteLocation(SITE_LAT, SITE_LON, elevation_wgs84)

    site_ecef = latLonAlt2ECEF(np.radians(SITE_LAT), np.radians(SITE_LON), elevation_wgs84)

    print("Site: {:.6f} N, {:.6f} E, {:.1f} m (WGS84)".format(SITE_LAT, SITE_LON, elevation_wgs84))

    # Height above sea level, used for the refraction. The difference between the ellipsoidal and
    # the orthometric height is negligible for the refraction.
    site_elevation_msl = elevation_wgs84

    print("Building the refraction table...")
    refraction_table = RefractionTable(site_elevation_msl)

    ### ###


    ### Search ###

    n_grid = int((t_end - t_beg).total_seconds()/SEARCH_TIME_STEP) + 1

    times_grid = np.array([t_beg + datetime.timedelta(seconds=i*SEARCH_TIME_STEP)
        for i in range(n_grid)])

    candidates, eph, tracks = searchTransits(df, location, site_ecef, site_elevation_msl,
        refraction_table, times_grid, max_separation=cml_args.sep)

    print("Found {:d} aircraft within {:.2f} deg of the centre of the Sun".format(len(candidates),
        cml_args.sep))

    ### ###


    ### Save the results ###

    printCandidates(candidates, n_print=cml_args.nprint)

    if candidates:

        df_cand = pd.DataFrame(candidates)

        csv_path = os.path.join(RESULTS_DIR, 'candidates.csv')
        df_cand.to_csv(csv_path, index=False)

        print()
        print("Ranked candidates written to: {:s}".format(csv_path))
        print("Run Utils/PlotTransits.py to plot the best candidates.")

    ### ###
