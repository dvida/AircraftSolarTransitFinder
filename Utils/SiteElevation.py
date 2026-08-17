"""
Look up the ground elevation of the observing site in a digital elevation model.

The elevation is queried from the public OpenTopoData API and converted from the orthometric
(geoid) height returned by the DEM to the ellipsoidal (WGS84) height used by the geodetic
routines. The result is cached in a JSON file so that the search does not depend on the network.

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
import sys
import json

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SolarTransit.Config import SITE_LAT, SITE_LON, SITE_ELEVATION_CACHE


# Public digital elevation model API
OPENTOPODATA_URL = "https://api.opentopodata.org/v1/{:s}"

# Datasets which are tried in order. EU-DEM covers Europe at 25 m, SRTM is the global fallback.
DEM_DATASETS = ['eudem25m', 'srtm30m', 'mapzen']



def queryDEM(lat, lon, timeout=30):
    """ Query the ground elevation from the OpenTopoData API.

    Arguments:
        lat: [float] Latitude (deg, +north).
        lon: [float] Longitude (deg, +east).

    Keyword arguments:
        timeout: [float] Request timeout in seconds.

    Return:
        (elevation, dataset): [tuple] Orthometric elevation in meters and the name of the dataset
            it was taken from.

    """

    for dataset in DEM_DATASETS:

        try:
            resp = requests.get(OPENTOPODATA_URL.format(dataset),
                params={'locations': "{:.6f},{:.6f}".format(lat, lon)}, timeout=timeout)

        except requests.RequestException as e:
            print("Query of the {:s} dataset failed: {:s}".format(dataset, str(e)))
            continue


        if resp.status_code != 200:
            print("The {:s} dataset returned status {:d}".format(dataset, resp.status_code))
            continue


        results = resp.json().get('results', [])

        if results and (results[0].get('elevation') is not None):
            return float(results[0]['elevation']), dataset


        print("The {:s} dataset has no data at this location".format(dataset))


    raise RuntimeError("No digital elevation model returned an elevation for this location")



def geoidUndulation(lat, lon):
    """ Compute the height of the geoid above the WGS84 ellipsoid.

    The EGM96 geoid model is used through pyproj, which downloads the grid on the first use. If
    the grid cannot be obtained, None is returned.

    Arguments:
        lat: [float] Latitude (deg, +north).
        lon: [float] Longitude (deg, +east).

    Return:
        undulation: [float or None] Geoid height above the ellipsoid in meters.

    """

    try:
        import pyproj

        # Allow pyproj to download the geoid grid if it is not available locally
        pyproj.network.set_network_enabled(True)

        # Transform from orthometric (EGM96) to ellipsoidal (WGS84) heights
        transformer = pyproj.Transformer.from_crs("EPSG:4326+5773", "EPSG:4979", always_xy=True)

        _, _, h_ellipsoidal = transformer.transform(lon, lat, 0.0)

        # A height of zero above the geoid transforms into the undulation itself
        if abs(h_ellipsoidal) > 200:
            return None

        return h_ellipsoidal


    except Exception as e:
        print("The geoid undulation could not be computed: {:s}".format(str(e)))
        return None



def siteElevationLookup(lat, lon, cache_path=None):
    """ Look up the elevation of the site above the WGS84 ellipsoid and cache the result.

    Arguments:
        lat: [float] Latitude (deg, +north).
        lon: [float] Longitude (deg, +east).

    Keyword arguments:
        cache_path: [str] Path of the JSON cache file which will be written.

    Return:
        info: [dict] Dictionary with the elevations, the dataset name and the geoid undulation.

    """

    elevation_msl, dataset = queryDEM(lat, lon)

    print("DEM {:s}: orthometric elevation {:.1f} m".format(dataset, elevation_msl))

    undulation = geoidUndulation(lat, lon)

    if undulation is None:

        # Fall back to the approximate EGM96 undulation in eastern Spain
        undulation = 50.0
        undulation_source = 'assumed'
        print("Using the assumed geoid undulation of {:.1f} m".format(undulation))

    else:
        undulation_source = 'EGM96'
        print("EGM96 geoid undulation: {:.1f} m".format(undulation))


    elevation_wgs84 = elevation_msl + undulation

    print("Elevation above the WGS84 ellipsoid: {:.1f} m".format(elevation_wgs84))

    info = {
        'latitude': lat,
        'longitude': lon,
        'elevation_msl': elevation_msl,
        'elevation_wgs84': elevation_wgs84,
        'geoid_undulation': undulation,
        'geoid_source': undulation_source,
        'dem_dataset': dataset,
        }


    if cache_path is not None:

        cache_dir = os.path.dirname(cache_path)

        if cache_dir and (not os.path.exists(cache_dir)):
            os.makedirs(cache_dir)

        with open(cache_path, 'w') as f:
            json.dump(info, f, indent=4)

        print("Cached to: {:s}".format(cache_path))


    return info



if __name__ == "__main__":

    import argparse


    ### COMMAND LINE ARGUMENTS ###

    arg_parser = argparse.ArgumentParser(description="""Look up the ground elevation of the
        observing site in a digital elevation model. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-a', '--lat', metavar='LATITUDE', type=float, default=SITE_LAT,
        help="Latitude of the site (deg, +north).")

    arg_parser.add_argument('-o', '--lon', metavar='LONGITUDE', type=float, default=SITE_LON,
        help="Longitude of the site (deg, +east).")

    cml_args = arg_parser.parse_args()

    #########################


    siteElevationLookup(cml_args.lat, cml_args.lon, cache_path=SITE_ELEVATION_CACHE)
