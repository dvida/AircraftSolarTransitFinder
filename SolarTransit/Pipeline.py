"""
The complete analysis in one call.

Given a site, a time and a time window, this downloads the ADS-B telemetry, looks up the elevation
of the site, searches for aircraft in front of the Sun, writes the ranked table and draws the
plots. The individual steps are also available on their own, in the other modules of the package.

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

from SolarTransit.Config import (NOMINAL_TIME, SEARCH_HALF_WINDOW, MAX_SEPARATION_DEG,
    RESULTS_DIR, SUBSET_DATA_DIR, RAW_DATA_DIR)
from SolarTransit.Ephemeris import eclipseCircumstances
from SolarTransit.PlotTransits import plotCandidates, DEFAULT_VIEW, VIEW_MODES
from SolarTransit.Site import ObservingSite
from SolarTransit.TransitSearch import runSearch, writeCandidates, printCandidates


### CONSTANTS ###

# Format in which the time of the event is given on the command line
TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"]

### ###



def parseTime(time_str):
    """ Parse the time of the event, which is given in UTC.

    Arguments:
        time_str: [str] Time, e.g. 2026-08-12 18:30:00.

    Return:
        dt: [datetime] The parsed time.

    """

    for time_format in TIME_FORMATS:

        try:
            return datetime.datetime.strptime(time_str.strip(), time_format)

        except ValueError:
            continue


    raise ValueError("The time '{:s}' could not be parsed. Use one of: {:s}".format(time_str,
        ", ".join(TIME_FORMATS)))



def summarizeSun(site, nominal_time):
    """ Print where the Sun was at the time of the event, and how much of it was covered.

    Arguments:
        site: [ObservingSite] The observing site.
        nominal_time: [datetime] Time of the event, UTC.

    Return:
        circ: [dict] Circumstances at the time of the event.

    """

    circ = eclipseCircumstances(np.array([nominal_time]), site.astropyLocation())

    print("The Sun at {:s} UTC: altitude {:.2f} deg, azimuth {:.2f} deg".format(str(nominal_time),
        circ['sun_alt'][0], circ['sun_azim'][0]))

    if circ['obscuration'][0] > 0.001:
        print("    the solar disk was {:.1f}% covered by the Moon".format(
            100*circ['obscuration'][0]))

    return circ



def runPipeline(site, nominal_time, half_window=SEARCH_HALF_WINDOW,
    max_separation=MAX_SEPARATION_DEG, results_dir=RESULTS_DIR, subset_path=None,
    data_dir=RAW_DATA_DIR, make_plots=True, n_plot=3, view=DEFAULT_VIEW, n_print=10,
    df_adsb=None):
    """ Run the whole analysis, from the download of the data to the plots.

    Arguments:
        site: [ObservingSite] The observing site.
        nominal_time: [datetime] Middle of the searched time window, UTC.

    Keyword arguments:
        half_window: [timedelta] Half width of the searched time window.
        max_separation: [float] Report the aircraft which came closer than this to the centre of
            the Sun (deg).
        results_dir: [str] Directory into which the table and the plots are written.
        subset_path: [str] Path of the regional subset of the telemetry. It is written on the
            first run and read on the later ones.
        data_dir: [str] Directory in which the raw hourly files are cached.
        make_plots: [bool] Draw the plots. True by default.
        n_plot: [int] Number of the best candidates which are plotted.
        view: [str] Orientation of the view of the Sun, one of the keys of VIEW_MODES.
        n_print: [int] Number of candidates which are printed.
        df_adsb: [pandas.DataFrame] ADS-B reports to search. If given, nothing is downloaded, so
            telemetry from any other source can be used.

    Return:
        results: [dict] The candidates, the path of the table, the paths of the plots and the
            circumstances at the time of the event.

    """

    if subset_path is None:
        subset_path = os.path.join(SUBSET_DATA_DIR, 'adsb_region.parquet')


    ### Search ###

    candidates, df_adsb, refraction_table = runSearch(site, nominal_time, half_window,
        max_separation=max_separation, subset_path=subset_path, data_dir=data_dir,
        df_adsb=df_adsb)

    ### ###


    circ = summarizeSun(site, nominal_time)

    printCandidates(candidates, n_print=n_print)

    results = {
        'site': site,
        'nominal_time': nominal_time,
        'candidates': candidates,
        'circumstances': circ,
        'csv_path': None,
        'plot_paths': [],
        }

    if not candidates:
        print()
        print("No aircraft came within {:.2f} deg of the centre of the Sun. Try a wider time "
            "window.".format(max_separation))

        return results


    ### Write the table ###

    csv_path = writeCandidates(candidates, results_dir)

    results['csv_path'] = csv_path

    print()
    print("Ranked candidates written to: {:s}".format(csv_path))

    ### ###


    ### Plot ###

    if make_plots:

        df_cand = pd.DataFrame(candidates).head(n_plot)

        results['plot_paths'] = plotCandidates(df_cand, df_adsb, site, nominal_time, view=view,
            results_dir=results_dir, refraction_table=refraction_table)

    ### ###


    ### The verdict ###

    best = candidates[0]

    print()

    if best['inside_disk']:
        print("The best candidate crossed the disk of the Sun:")

    else:
        print("No aircraft crossed the disk. The closest one passed {:.2f} deg from the limb "
            "of the Sun:".format(best['min_sep_deg'] - best['sun_radius_deg']))


    print("    {:s} ({:s}, {:s}) at {:s} UTC".format(str(best['callsign']),
        str(best['tail_number']), str(best['aircraft_type_icao']),
        best['time_min_sep'].strftime("%H:%M:%S.%f")[:-4]))

    print("    {:.4f} +/- {:.4f} deg from the centre of the Sun, whose radius was {:.4f} "
        "deg".format(best['min_sep_deg'], best['sigma_deg'], best['sun_radius_deg']))

    print("    flight level {:.0f}, {:.0f} km away, {:s} to {:s}".format(
        best['altitude_baro_ft']/100, best['slant_range_km'], str(best['departure_airport_icao']),
        str(best['arrival_airport_icao'])))

    # A match which is interpolated over a long gap between the reports is not trustworthy
    if not best['reliable_gap']:
        print("    the closest approach is interpolated over a gap of {:.0f} s between the ADS-B "
            "reports, so the time is uncertain".format(best['sample_gap_s']))

    ### ###

    return results



def main():
    """ Run the whole analysis from the command line. """

    import argparse


    ### COMMAND LINE ARGUMENTS ###

    description = (
"""Find the aircraft which crossed the apparent disk of the Sun, as seen from a given place at a
given time.

Everything is done in one call: the elevation of the site is looked up, the ADS-B telemetry is
downloaded and cached, the search is run, the ranked table is written and the plots are drawn.

Example:
    solar-transit --lat 41.6488 --lon -0.8891 --time "2026-08-12 18:30:00" --window 12
""")

    arg_parser = argparse.ArgumentParser(description=description,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-a', '--lat', metavar='LATITUDE', type=float,
        help="Latitude of the site (deg, +north), WGS84. Taken from the configuration if it is "
             "not given.")

    arg_parser.add_argument('-o', '--lon', metavar='LONGITUDE', type=float,
        help="Longitude of the site (deg, +east), WGS84. Taken from the configuration if it is "
             "not given.")

    arg_parser.add_argument('-e', '--height', metavar='METERS', type=float,
        help="Ground elevation of the site above the WGS84 ellipsoid (m). Looked up in a digital "
             "elevation model if it is not given.")

    arg_parser.add_argument('-t', '--time', metavar='UTC', type=str,
        help="Time of the event in UTC, e.g. \"2026-08-12 18:30:00\". Taken from the "
             "configuration if it is not given.")

    arg_parser.add_argument('-w', '--window', metavar='MINUTES', type=float,
        default=SEARCH_HALF_WINDOW.total_seconds()/60,
        help="Half width of the searched time window. Use a wide window if the time of the event "
             "is only known to the minute.")

    arg_parser.add_argument('-s', '--sep', metavar='DEGREES', type=float,
        default=MAX_SEPARATION_DEG,
        help="Report the aircraft which came closer than this to the centre of the Sun.")

    arg_parser.add_argument('-n', '--nplot', metavar='COUNT', type=int, default=3,
        help="Number of the best candidates which are plotted.")

    arg_parser.add_argument('--nprint', metavar='COUNT', type=int, default=10,
        help="Number of candidates which are printed.")

    arg_parser.add_argument('-v', '--view', metavar='ORIENTATION', type=str, default=DEFAULT_VIEW,
        choices=sorted(VIEW_MODES.keys()),
        help="Orientation of the view of the Sun:\n"
             "    naked_eye - as seen with the unaided eye\n"
             "    flipped   - flipped vertically, as in the eyepiece of a telescope (default)\n"
             "    mirrored  - mirrored left to right, as through a star diagonal\n"
             "    inverted  - upside down and mirrored, as in an inverting telescope")

    arg_parser.add_argument('--noplots', action="store_true",
        help="Only write the table of candidates, do not draw the plots.")

    arg_parser.add_argument('-d', '--datadir', metavar='DIR', type=str, default=RAW_DATA_DIR,
        help="Directory in which the raw hourly ADS-B files are cached.")

    arg_parser.add_argument('-r', '--resultsdir', metavar='DIR', type=str, default=RESULTS_DIR,
        help="Directory into which the table and the plots are written.")

    arg_parser.add_argument('-i', '--input', metavar='PARQUET_PATH', type=str,
        help="Path of the regional subset of the telemetry. It is written on the first run and "
             "read on the later ones.")

    cml_args = arg_parser.parse_args()

    #########################


    # The site and the time may be given on the command line, otherwise they are taken from the
    # configuration
    site = ObservingSite.fromConfig()

    if cml_args.lat is not None:
        site.lat = cml_args.lat

    if cml_args.lon is not None:
        site.lon = cml_args.lon

    if cml_args.height is not None:
        site.ground_elevation = cml_args.height
        site.elevation_source = 'given'

    # The cached elevation belongs to the configured site, so it cannot be used for another one
    if (cml_args.lat is not None) or (cml_args.lon is not None):
        site.ground_elevation = cml_args.height


    nominal_time = NOMINAL_TIME if cml_args.time is None else parseTime(cml_args.time)

    runPipeline(site, nominal_time, half_window=datetime.timedelta(minutes=cml_args.window),
        max_separation=cml_args.sep, results_dir=cml_args.resultsdir,
        subset_path=cml_args.input, data_dir=cml_args.datadir, make_plots=not cml_args.noplots,
        n_plot=cml_args.nplot, view=cml_args.view, n_print=cml_args.nprint)



if __name__ == "__main__":

    main()
