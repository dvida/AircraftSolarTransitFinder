"""
Configuration of the solar transit search.
Includes:
    - Observing site definition
    - Time window of the search
    - ADS-B data source (Contrails.org API which serves Spire data)
    - Search and uncertainty thresholds

Every value below can be overridden without touching this file, by writing a config.json in the
root of the repository. See config.example.json.

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
import json
from datetime import datetime, timedelta


### PATHS ###

# Root directory of the repository
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directory where the raw hourly ADS-B parquet files are cached
RAW_DATA_DIR = os.path.join(REPO_ROOT, 'data', 'raw')

# Directory where the regional subsets are stored
SUBSET_DATA_DIR = os.path.join(REPO_ROOT, 'data')

# Directory where the results (tables, plots) are written
RESULTS_DIR = os.path.join(REPO_ROOT, 'results')

# Optional local configuration, which overrides the values in this file. It is not a part of the
# repository, so an observing site does not have to be published together with the code.
LOCAL_CONFIG_FILE = os.path.join(REPO_ROOT, 'config.json')

### ###


### ADS-B DATA SOURCE ###

# Contrails.org API which proxies Spire Aviation ADS-B telemetry.
# One request returns one hour of global data as an Apache Parquet file.
ADSB_API_URL = "https://api.contrails.org/v1/adsb/telemetry"

# Name of the environment variable holding the API key
API_KEY_ENV_VAR = "CONTRAILS_API_KEY"

# Fallback location of a file containing nothing but the API key. Used if the environment
# variable is not set. The key is never stored in the repository.
API_KEY_FILE = os.path.join(os.path.expanduser("~"), ".config", "solar_transit", "api_key")

### ###


### OBSERVING SITE ###

# Geographic coordinates of the site, WGS84. The values below are only an example, and place the
# site in Zaragoza, which was inside the path of totality of the 2026 August 12 eclipse. Set your
# own site in config.json.
SITE_LAT = 41.6488
SITE_LON = -0.8891

# Height of the observer above the ground (m)
SITE_EYE_HEIGHT = 1.6

# Ground elevation above the WGS84 ellipsoid (m). If None, it is looked up from a digital
# elevation model (see Utils/SiteElevation.py) and cached in the file below.
SITE_GROUND_ELEVATION = None

# Cache file of the looked up site elevation
SITE_ELEVATION_CACHE = os.path.join(SUBSET_DATA_DIR, 'site_elevation.json')

# Assumed uncertainty of the site elevation (m)
SITE_ELEVATION_SIGMA = 30.0

### ###


### SEARCH WINDOW ###

# Time of the event (UTC). The example is the middle of the total solar eclipse of
# 2026 August 12, as seen from Spain.
NOMINAL_TIME = datetime(2026, 8, 12, 18, 30, 0)

# Half width of the searched time window. If the time of the event is only known to the minute,
# it pays to search a wide window and to rank everything that comes close.
SEARCH_HALF_WINDOW = timedelta(minutes=8)

# Time step of the search (s). An airliner crosses the solar disk in about a second, so the
# sampling has to be much finer than that.
SEARCH_TIME_STEP = 0.1

# Size of the box around the site from which the ADS-B data are extracted (deg). At an apparent
# elevation of 6 deg an aircraft at a cruising altitude is 80 - 120 km away, but when the Sun is
# only 2 deg above the horizon the same geometry puts it 350 km away, so the box is generous.
BOX_MARGIN_LAT = 3.5
BOX_MARGIN_LON = 4.5

### ###


### SEARCH THRESHOLDS ###

# Maximum angular separation from the centre of the Sun for an aircraft to be reported (deg).
# The solar radius is about 0.262 deg, so this keeps the near misses and the inner corona.
MAX_SEPARATION_DEG = 2.0

# Assumed uncertainty of the conversion from the barometric to the geometric altitude (m). The
# ADS-B feed only reports the barometric altitude, which is not corrected for the real pressure
# field.
BARO_ALTITUDE_SIGMA = 250.0

# Assumed uncertainty of the reported ADS-B horizontal position (m)
ADSB_POSITION_SIGMA = 100.0

# Flights with fewer reports than this are skipped
MIN_POINTS_PER_FLIGHT = 3

# Maximum interpolation gap which is still considered usable (s). Matches are still reported
# beyond this, but they are flagged as unreliable.
MAX_USABLE_GAP = 30.0

### ###



def loadLocalConfig(file_path=LOCAL_CONFIG_FILE):
    """ Read the local configuration, which overrides the values in this file.

    Arguments:
        file_path: [str] Path of the JSON file with the local configuration.

    Return:
        local: [dict] Contents of the file, or an empty dictionary if there is none.

    """

    if not os.path.isfile(file_path):
        return {}


    with open(file_path) as f:
        return json.load(f)



### Apply the local configuration ###

_local = loadLocalConfig()

SITE_LAT = _local.get('site_lat', SITE_LAT)
SITE_LON = _local.get('site_lon', SITE_LON)
SITE_EYE_HEIGHT = _local.get('site_eye_height', SITE_EYE_HEIGHT)
SITE_GROUND_ELEVATION = _local.get('site_ground_elevation', SITE_GROUND_ELEVATION)

if 'nominal_time' in _local:
    NOMINAL_TIME = datetime.strptime(_local['nominal_time'], "%Y-%m-%d %H:%M:%S")

if 'search_half_window_minutes' in _local:
    SEARCH_HALF_WINDOW = timedelta(minutes=_local['search_half_window_minutes'])

BOX_MARGIN_LAT = _local.get('box_margin_lat', BOX_MARGIN_LAT)
BOX_MARGIN_LON = _local.get('box_margin_lon', BOX_MARGIN_LON)
MAX_SEPARATION_DEG = _local.get('max_separation_deg', MAX_SEPARATION_DEG)

### ###


# Edges of the box from which the ADS-B data are extracted
BOX_LAT_MIN = SITE_LAT - BOX_MARGIN_LAT
BOX_LAT_MAX = SITE_LAT + BOX_MARGIN_LAT
BOX_LON_MIN = SITE_LON - BOX_MARGIN_LON
BOX_LON_MAX = SITE_LON + BOX_MARGIN_LON



def getApiKey():
    """ Read the ADS-B API key from the environment or from the key file of the user.

    The key is deliberately never stored in the repository, so that the repository can be public.

    Return:
        api_key: [str] The API key.

    """

    # First try the environment variable
    api_key = os.environ.get(API_KEY_ENV_VAR)

    if api_key is not None:
        api_key = api_key.strip()

        if api_key:
            return api_key


    # Then try the key file
    if os.path.isfile(API_KEY_FILE):

        with open(API_KEY_FILE) as f:
            api_key = f.read().strip()

        if api_key:
            return api_key


    raise RuntimeError(
        "The ADS-B API key was not found. Either set the {:s} environment variable, or write "
        "the key into {:s}".format(API_KEY_ENV_VAR, API_KEY_FILE)
        )



def siteElevation():
    """ Return the elevation of the observer above the WGS84 ellipsoid (m).

    If the ground elevation is not given in the configuration, it is taken from the digital
    elevation model cache written by Utils/SiteElevation.py.

    Return:
        elevation: [float] Elevation of the observer in meters.

    """

    ground_elevation = SITE_GROUND_ELEVATION

    # Load the cached lookup if the elevation was not set manually
    if ground_elevation is None:

        if not os.path.isfile(SITE_ELEVATION_CACHE):
            raise RuntimeError(
                "The site elevation is not known. Run Utils/SiteElevation.py to look it up, or "
                "set site_ground_elevation in config.json"
                )


        with open(SITE_ELEVATION_CACHE) as f:
            cache = json.load(f)


        # Make sure that the cache belongs to the site which is being used
        if (abs(cache['latitude'] - SITE_LAT) > 1e-4) or (abs(cache['longitude'] - SITE_LON)
            > 1e-4):

            raise RuntimeError("The cached site elevation was computed for {:.6f}, {:.6f}, but "
                "the site is at {:.6f}, {:.6f}. Run Utils/SiteElevation.py again.".format(
                cache['latitude'], cache['longitude'], SITE_LAT, SITE_LON))


        ground_elevation = cache['elevation_wgs84']


    return ground_elevation + SITE_EYE_HEIGHT
