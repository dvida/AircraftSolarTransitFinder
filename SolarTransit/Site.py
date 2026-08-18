"""
The observing site.
Includes:
    - Definition of the site from coordinates given by hand or taken from the configuration
    - Lookup of the ground elevation in a digital elevation model
    - ECEF coordinates and the astropy location of the site
    - The geographic box from which the ADS-B data are extracted

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

import numpy as np

from SolarTransit.Config import (SITE_LAT, SITE_LON, SITE_EYE_HEIGHT, SITE_GROUND_ELEVATION,
    SITE_ELEVATION_CACHE, BOX_MARGIN_LAT, BOX_MARGIN_LON)
from SolarTransit.Conversions import latLonAlt2ECEF
from SolarTransit.Ephemeris import siteLocation
from SolarTransit.SiteElevation import siteElevationLookup, readElevationCache



class ObservingSite(object):
    """ The place from which the Sun was observed.

    The ground elevation may be given directly, or it is looked up in a digital elevation model
    the first time it is needed and cached, so that later runs do not need the network.

    """

    def __init__(self, lat, lon, ground_elevation=None, eye_height=SITE_EYE_HEIGHT,
        elevation_cache=SITE_ELEVATION_CACHE):
        """
        Arguments:
            lat: [float] Latitude of the site (deg, +north), WGS84.
            lon: [float] Longitude of the site (deg, +east), WGS84.

        Keyword arguments:
            ground_elevation: [float] Ground elevation above the WGS84 ellipsoid (m). If None, it
                is looked up in a digital elevation model.
            eye_height: [float] Height of the observer above the ground (m).
            elevation_cache: [str] Path of the JSON file in which the lookup is cached.

        """

        self.lat = float(lat)
        self.lon = float(lon)
        self.eye_height = float(eye_height)
        self.elevation_cache = elevation_cache

        self.ground_elevation = None if ground_elevation is None else float(ground_elevation)

        # Set when the elevation is taken from a digital elevation model
        self.elevation_source = 'given' if ground_elevation is not None else None



    @classmethod
    def fromConfig(cls):
        """ Create the site from the values in the configuration and in config.json.

        Return:
            site: [ObservingSite] The configured site.

        """

        return cls(SITE_LAT, SITE_LON, ground_elevation=SITE_GROUND_ELEVATION)



    def resolveElevation(self, verbose=True):
        """ Make sure that the ground elevation of the site is known.

        The cached lookup is used if it was made for the same coordinates, otherwise the digital
        elevation model is queried and the result is cached.

        Keyword arguments:
            verbose: [bool] Print what was done.

        Return:
            ground_elevation: [float] Ground elevation above the WGS84 ellipsoid (m).

        """

        if self.ground_elevation is not None:
            return self.ground_elevation


        # Use the cache if it holds an entry for this site
        cached = readElevationCache(self.lat, self.lon, cache_path=self.elevation_cache)

        if cached is not None:

            self.ground_elevation = cached['elevation_wgs84']
            self.elevation_source = cached['dem_dataset'] + ' (cached)'

            if verbose:
                print("Site elevation from the cache: {:.1f} m above the WGS84 "
                    "ellipsoid".format(self.ground_elevation))

            return self.ground_elevation


        # Otherwise query the digital elevation model
        info = siteElevationLookup(self.lat, self.lon, cache_path=self.elevation_cache)

        self.ground_elevation = info['elevation_wgs84']
        self.elevation_source = info['dem_dataset']

        return self.ground_elevation



    @property
    def elevation(self):
        """ Elevation of the observer above the WGS84 ellipsoid (m). """

        if self.ground_elevation is None:
            self.resolveElevation()

        return self.ground_elevation + self.eye_height



    def ecef(self):
        """ ECEF coordinates of the observer.

        Return:
            (x, y, z): [tuple of floats] ECEF coordinates in meters.

        """

        return latLonAlt2ECEF(np.radians(self.lat), np.radians(self.lon), self.elevation)



    def astropyLocation(self):
        """ Location of the site as used by the ephemeris.

        Return:
            location: [EarthLocation] Location of the site.

        """

        return siteLocation(self.lat, self.lon, self.elevation)



    def box(self, margin_lat=BOX_MARGIN_LAT, margin_lon=BOX_MARGIN_LON):
        """ Geographic box around the site from which the ADS-B data are extracted.

        Keyword arguments:
            margin_lat: [float] Half height of the box (deg).
            margin_lon: [float] Half width of the box (deg).

        Return:
            (lat_min, lat_max, lon_min, lon_max): [tuple of floats] Edges of the box (deg).

        """

        return (self.lat - margin_lat, self.lat + margin_lat, self.lon - margin_lon,
            self.lon + margin_lon)



    def __repr__(self):

        return "ObservingSite({:.6f} N, {:.6f} E, {:s})".format(self.lat, self.lon,
            "elevation not resolved" if self.ground_elevation is None
            else "{:.1f} m".format(self.ground_elevation))
