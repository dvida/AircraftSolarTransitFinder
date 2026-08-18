"""
Fetching and caching of ADS-B telemetry from the Contrails.org API, which serves Spire Aviation
data.
Includes:
    - Downloading of hourly global parquet files
    - Local caching of the raw files
    - Extraction of a geographic subset around the observing site

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
import requests

from SolarTransit.Config import (ADSB_API_URL, RAW_DATA_DIR, SUBSET_DATA_DIR, getApiKey,
    BOX_LAT_MIN, BOX_LAT_MAX, BOX_LON_MIN, BOX_LON_MAX)


# Columns which are kept in the regional subset
SUBSET_COLUMNS = ['timestamp', 'latitude', 'longitude', 'altitude_baro', 'collection_type',
    'icao_address', 'flight_id', 'callsign', 'tail_number', 'flight_number', 'aircraft_type_icao',
    'airline_iata', 'departure_airport_icao', 'arrival_airport_icao']



def hourString(dt):
    """ Format a datetime into the hourly string expected by the ADS-B API.

    Arguments:
        dt: [datetime] Time in UTC.

    Return:
        hour_str: [str] Time formatted as YYYY-MM-DDTHH.

    """

    return dt.strftime("%Y-%m-%dT%H")



def hoursInRange(dt_beg, dt_end):
    """ List all hourly slots which cover the given time range.

    Arguments:
        dt_beg: [datetime] Beginning of the range, UTC.
        dt_end: [datetime] End of the range, UTC.

    Return:
        hours: [list of datetime] Hourly slots, truncated to full hours.

    """

    hour_beg = dt_beg.replace(minute=0, second=0, microsecond=0)

    hours = []
    hour = hour_beg

    while hour <= dt_end:
        hours.append(hour)
        hour += datetime.timedelta(hours=1)

    return hours



def downloadHour(dt, data_dir=RAW_DATA_DIR, overwrite=False, timeout=900):
    """ Download one hour of global ADS-B telemetry and cache it locally.

    The API returns the complete global hour as an Apache Parquet file, which is on the order of
    60 MB. Existing files are not downloaded again.

    Arguments:
        dt: [datetime] Time in UTC. Only the hour is used.

    Keyword arguments:
        data_dir: [str] Directory where the raw files are cached.
        overwrite: [bool] Download the file even if it is already cached. False by default.
        timeout: [float] Download timeout in seconds.

    Return:
        file_path: [str] Path to the cached parquet file.

    """

    hour_str = hourString(dt)
    file_path = os.path.join(data_dir, hour_str + ".pq")

    # Skip the download if the file is already cached
    if os.path.isfile(file_path) and (not overwrite) and (os.path.getsize(file_path) > 0):
        print("Using the cached file: {:s}".format(file_path))
        return file_path


    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


    print("Downloading ADS-B telemetry for {:s} UTC...".format(hour_str))

    headers = {'x-api-key': getApiKey()}

    resp = requests.get(ADSB_API_URL, params={'date': hour_str}, headers=headers, timeout=timeout,
        stream=True)

    if resp.status_code != 200:
        raise RuntimeError("The ADS-B API returned status {:d}: {:s}".format(resp.status_code,
            resp.text[:200]))


    # Write to a temporary file first, so that an interrupted download does not poison the cache
    tmp_path = file_path + ".part"

    total_bytes = 0

    with open(tmp_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024*1024):

            if chunk:
                f.write(chunk)
                total_bytes += len(chunk)

    os.rename(tmp_path, file_path)

    print("    saved {:.1f} MB to {:s}".format(total_bytes/1e6, file_path))

    return file_path



def loadRegion(dt_beg, dt_end, lat_min=BOX_LAT_MIN, lat_max=BOX_LAT_MAX, lon_min=BOX_LON_MIN,
    lon_max=BOX_LON_MAX, data_dir=RAW_DATA_DIR, cache_path=None):
    """ Load all ADS-B telemetry inside a geographic box and a time range.

    The hourly global files are downloaded if they are not cached yet, and only the rows inside
    the box are kept in memory.

    Arguments:
        dt_beg: [datetime] Beginning of the time range, UTC.
        dt_end: [datetime] End of the time range, UTC.

    Keyword arguments:
        lat_min: [float] Southern edge of the box (deg).
        lat_max: [float] Northern edge of the box (deg).
        lon_min: [float] Western edge of the box (deg).
        lon_max: [float] Eastern edge of the box (deg).
        data_dir: [str] Directory where the raw hourly files are cached.
        cache_path: [str] If given, the subset is written to this parquet file and read back from
            it on subsequent runs.

    Return:
        df: [pandas.DataFrame] Telemetry inside the box, sorted by flight and time.

    """

    # Return the cached subset if it exists
    if (cache_path is not None) and os.path.isfile(cache_path):
        print("Using the cached regional subset: {:s}".format(cache_path))
        return pd.read_parquet(cache_path)


    subsets = []

    for hour in hoursInRange(dt_beg, dt_end):

        file_path = downloadHour(hour, data_dir=data_dir)

        # Read only the columns which are needed
        df_hour = pd.read_parquet(file_path, columns=SUBSET_COLUMNS)

        # Keep only the rows inside the box
        mask = (
              (df_hour['latitude'] >= lat_min) & (df_hour['latitude'] <= lat_max)
            & (df_hour['longitude'] >= lon_min) & (df_hour['longitude'] <= lon_max)
            )

        df_hour = df_hour[mask]

        print("    {:s} UTC: {:d} points inside the box".format(hourString(hour), len(df_hour)))

        subsets.append(df_hour)


    df = pd.concat(subsets, ignore_index=True)

    # Make sure the timestamps are timezone naive UTC, so that they compare cleanly against the
    # datetimes used elsewhere in the code
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)

    # Keep only the rows inside the time range
    df = df[(df['timestamp'] >= pd.Timestamp(dt_beg)) & (df['timestamp'] <= pd.Timestamp(dt_end))]

    # Remove duplicated reports of the same aircraft at the same time (the same target can be
    # reported by both a terrestrial and a satellite receiver)
    df = df.sort_values(['flight_id', 'timestamp', 'collection_type'])
    df = df.drop_duplicates(subset=['flight_id', 'timestamp'], keep='first')

    df = df.reset_index(drop=True)

    print("Total: {:d} points from {:d} flights".format(len(df), df['flight_id'].nunique()))


    # Cache the subset
    if cache_path is not None:

        cache_dir = os.path.dirname(cache_path)

        if cache_dir and (not os.path.exists(cache_dir)):
            os.makedirs(cache_dir)

        df.to_parquet(cache_path, index=False)

        print("Regional subset written to: {:s}".format(cache_path))


    return df



def main():
    """ Download the data around the configured event and extract the region around the site. """

    import argparse

    from SolarTransit.Config import NOMINAL_TIME, SEARCH_HALF_WINDOW


    ### COMMAND LINE ARGUMENTS ###

    arg_parser = argparse.ArgumentParser(description="""Download ADS-B telemetry around the time
        of the event and extract the region around the observing site. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-m', '--margin', metavar='MINUTES', type=float, default=60.0,
        help="Extend the time range by this many minutes on both sides of the search window.")

    arg_parser.add_argument('-o', '--output', metavar='PARQUET_PATH', type=str,
        default=os.path.join(SUBSET_DATA_DIR, 'adsb_region.parquet'),
        help="Path of the regional subset which will be written.")

    cml_args = arg_parser.parse_args()

    #########################


    margin = datetime.timedelta(minutes=cml_args.margin)

    dt_beg = NOMINAL_TIME - SEARCH_HALF_WINDOW - margin
    dt_end = NOMINAL_TIME + SEARCH_HALF_WINDOW + margin

    print("Loading ADS-B data from {:s} to {:s} UTC".format(str(dt_beg), str(dt_end)))

    loadRegion(dt_beg, dt_end, cache_path=cml_args.output)



if __name__ == "__main__":

    main()
