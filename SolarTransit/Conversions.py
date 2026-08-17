"""
Geodetic and horizontal coordinate conversions.
Includes:
    - WGS84 geodetic to ECEF conversion
    - ECEF to alt/az conversion
    - Julian date conversion
    - Angular separation on the sphere

The geodesy is taken from the Raspberry Pi Meteor Station library (RMS/Astrometry/Conversions.py,
https://github.com/CroatianMeteorNetwork/RMS), vectorized here for numpy arrays.

"""

# The MIT License

# Copyright (c) 2016 Denis Vida

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

import math
from datetime import datetime

import numpy as np


### CONSTANTS ###

# Julian date of the J2000.0 epoch and the corresponding calendar date
JULIAN_EPOCH = datetime(2000, 1, 1, 12)
JULIAN_EPOCH_JD = 2451545.0


class EARTH_CONSTANTS(object):
    """ Holds Earth's shape and physical parameters. """

    def __init__(self):

        # Earth ellipsoid parameters in meters (source: WGS84, the GPS standard)
        self.EQUATORIAL_RADIUS = 6378137.0
        self.POLAR_RADIUS = 6356752.314245
        self.E = math.sqrt(1.0 - self.POLAR_RADIUS**2/self.EQUATORIAL_RADIUS**2)
        self.RATIO = self.EQUATORIAL_RADIUS/self.POLAR_RADIUS
        self.SQR_DIFF = self.EQUATORIAL_RADIUS**2 - self.POLAR_RADIUS**2


# Initialize Earth shape constants object
EARTH = EARTH_CONSTANTS()

### ###


def datetime2JD(dt):
    """ Convert a datetime object to a Julian date.

    Arguments:
        dt: [datetime] Time in UTC.

    Return:
        jd: [float] Julian date.

    """

    return (dt - JULIAN_EPOCH).total_seconds()/86400.0 + JULIAN_EPOCH_JD



def latLonAlt2ECEF(lat, lon, h):
    """ Convert geographical coordinates to Earth centered - Earth fixed coordinates.

    Vectorized version of the function from RMS/Astrometry/Conversions.py.

    Arguments:
        lat: [float or ndarray] Latitude in radians (+north).
        lon: [float or ndarray] Longitude in radians (+east).
        h: [float or ndarray] Elevation in meters (WGS84).

    Return:
        (x, y, z): [tuple of floats or ndarrays] ECEF coordinates in meters.

    """

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(lat)**2)

    # Calculate ECEF coordinates
    ecef_x = (N + h)*np.cos(lat)*np.cos(lon)
    ecef_y = (N + h)*np.cos(lat)*np.sin(lon)
    ecef_z = ((1 - EARTH.E**2)*N + h)*np.sin(lat)

    return ecef_x, ecef_y, ecef_z



def ecef2LatLonAlt(x, y, z):
    """ Convert Earth centered - Earth fixed coordinates to geographical coordinates.

    Vectorized version of the function from RMS/Astrometry/Conversions.py, which uses Bowring's
    method. The polar special case of the original is not reproduced here, because the search is
    never done near the poles.

    Arguments:
        x: [float or ndarray] ECEF x coordinate (m).
        y: [float or ndarray] ECEF y coordinate (m).
        z: [float or ndarray] ECEF z coordinate (m).

    Return:
        (lat, lon, alt): [tuple] Latitude and longitude in radians, and the WGS84 elevation in
            meters.

    """

    # Calculate the polar eccentricity
    ep = np.sqrt((EARTH.EQUATORIAL_RADIUS**2 - EARTH.POLAR_RADIUS**2)/(EARTH.POLAR_RADIUS**2))

    lon = np.arctan2(y, x)

    p = np.sqrt(x**2 + y**2)

    theta = np.arctan2(z*EARTH.EQUATORIAL_RADIUS, p*EARTH.POLAR_RADIUS)

    lat = np.arctan2(z + (ep**2)*EARTH.POLAR_RADIUS*np.sin(theta)**3,
        p - (EARTH.E**2)*EARTH.EQUATORIAL_RADIUS*np.cos(theta)**3)

    # Get distance from Earth centre to the position given by geographical coordinates, in WGS84
    N = EARTH.EQUATORIAL_RADIUS/np.sqrt(1.0 - (EARTH.E**2)*np.sin(lat)**2)

    alt = p/np.cos(lat) - N

    return lat, lon, alt



def ECEF2AltAz(s_vect, p_vect):
    """ Given two sets of ECEF coordinates, compute alt/az which point from the point S to the
        point P.

    The station vector S has to be a single point, while the target P may be an array of points.

    This is the same computation as in RMS/Astrometry/Conversions.py, except that the altitude is
    measured from the geodetic horizon, i.e. from the plane perpendicular to the normal of the
    WGS84 ellipsoid, and not from the geocentric one. The two verticals differ by up to 0.19 deg
    at this latitude, which is more than the radius of the Sun, and the astronomical positions
    are given in the geodetic frame as well.

    Arguments:
        s_vect: [ndarray] sx, sy, sz - S point ECEF coordinates in meters.
        p_vect: [ndarray] px, py, pz - P point ECEF coordinates in meters. Each component may be
            an array.

    Return:
        (azim, alt): [tuple of floats or ndarrays] Horizontal coordinates in degrees. The azimuth
            is measured from north towards east.

    """

    sx, sy, sz = s_vect
    px, py, pz = p_vect

    # Compute the pointing vector from S to P
    dx = px - sx
    dy = py - sy
    dz = pz - sz

    # Geodetic coordinates of the station, which define the local horizon
    lat, lon, _ = ecef2LatLonAlt(sx, sy, sz)

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # Rotate the pointing vector into the local east, north, up frame
    east = -sin_lon*dx + cos_lon*dy
    north = -sin_lat*cos_lon*dx - sin_lat*sin_lon*dy + cos_lat*dz
    up = cos_lat*cos_lon*dx + cos_lat*sin_lon*dy + sin_lat*dz

    azim = np.degrees(np.arctan2(east, north))%360
    alt = np.degrees(np.arctan2(up, np.sqrt(east**2 + north**2)))

    return azim, alt



def angularSeparation(azim1, alt1, azim2, alt2):
    """ Compute the angular separation between two points on the sky.

    The Vincenty formula is used, which is accurate for both small and large separations.

    Arguments:
        azim1: [float or ndarray] Azimuth of the first point (deg).
        alt1: [float or ndarray] Altitude of the first point (deg).
        azim2: [float or ndarray] Azimuth of the second point (deg).
        alt2: [float or ndarray] Altitude of the second point (deg).

    Return:
        sep: [float or ndarray] Angular separation (deg).

    """

    az1 = np.radians(azim1)
    el1 = np.radians(alt1)
    az2 = np.radians(azim2)
    el2 = np.radians(alt2)

    daz = az2 - az1

    num = np.sqrt((np.cos(el2)*np.sin(daz))**2
        + (np.cos(el1)*np.sin(el2) - np.sin(el1)*np.cos(el2)*np.cos(daz))**2)

    den = np.sin(el1)*np.sin(el2) + np.cos(el1)*np.cos(el2)*np.cos(daz)

    return np.degrees(np.arctan2(num, den))



def slantRange(s_vect, p_vect):
    """ Compute the distance between two ECEF points.

    Arguments:
        s_vect: [ndarray] sx, sy, sz - S point ECEF coordinates in meters.
        p_vect: [ndarray] px, py, pz - P point ECEF coordinates in meters.

    Return:
        dist: [float or ndarray] Distance in meters.

    """

    sx, sy, sz = s_vect
    px, py, pz = p_vect

    return np.sqrt((px - sx)**2 + (py - sy)**2 + (pz - sz)**2)
