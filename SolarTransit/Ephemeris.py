"""
Apparent positions of the Sun and the Moon, and the circumstances of the solar eclipse.
Includes:
    - Topocentric horizontal coordinates of the Sun and the Moon
    - Apparent angular radii of the Sun and the Moon
    - Obscuration of the solar disk by the Moon

The positions are computed as airless (geometric) topocentric places. The refraction is applied
separately, in the same way as it is applied to the aircraft, so that the two are consistent.

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

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, get_body

from SolarTransit.Conversions import angularSeparation


### CONSTANTS ###

# Physical radii of the Sun and the Moon (m)
SUN_RADIUS = 6.957e8
MOON_RADIUS = 1.7374e6

### ###



def siteLocation(lat, lon, elevation):
    """ Create an astropy location of the observing site.

    Arguments:
        lat: [float] Latitude (deg, +north).
        lon: [float] Longitude (deg, +east).
        elevation: [float] Height above the WGS84 ellipsoid (m).

    Return:
        location: [EarthLocation] Location of the site.

    """

    return EarthLocation.from_geodetic(lon=lon*u.deg, lat=lat*u.deg, height=elevation*u.m,
        ellipsoid='WGS84')



def bodyAltAz(body_name, times, location):
    """ Compute the airless topocentric horizontal coordinates and the angular radius of a body.

    Arguments:
        body_name: [str] Name of the body, e.g. 'sun' or 'moon'.
        times: [ndarray of datetime] Times in UTC.
        location: [EarthLocation] Location of the observing site.

    Return:
        (azim, alt, ang_radius): [tuple of ndarrays] Azimuth (deg, from north towards east),
            geometric altitude (deg) and the apparent angular radius of the body (deg).

    """

    obs_time = Time(times, scale='utc')

    # Apparent place of the body as seen from the site, corrected for light travel time and
    # aberration by astropy
    body = get_body(body_name, obs_time, location=location)

    # Refraction is not applied here, it is handled by SolarTransit.Refraction
    frame = AltAz(obstime=obs_time, location=location, pressure=0*u.hPa)

    body_altaz = body.transform_to(frame)

    azim = np.atleast_1d(body_altaz.az.deg)
    alt = np.atleast_1d(body_altaz.alt.deg)

    # Angular radius of the body as seen from the site
    distance = np.atleast_1d(body_altaz.distance.to(u.m).value)

    if body_name == 'sun':
        physical_radius = SUN_RADIUS

    elif body_name == 'moon':
        physical_radius = MOON_RADIUS

    else:
        physical_radius = 0.0

    ang_radius = np.degrees(np.arcsin(physical_radius/distance))

    return azim, alt, ang_radius



def sunAltAz(times, location):
    """ Compute the airless topocentric position and the angular radius of the Sun.

    Arguments:
        times: [ndarray of datetime] Times in UTC.
        location: [EarthLocation] Location of the observing site.

    Return:
        (azim, alt, ang_radius): [tuple of ndarrays] Azimuth (deg), geometric altitude (deg) and
            the angular radius (deg).

    """

    return bodyAltAz('sun', times, location)



def moonAltAz(times, location):
    """ Compute the airless topocentric position and the angular radius of the Moon.

    Arguments:
        times: [ndarray of datetime] Times in UTC.
        location: [EarthLocation] Location of the observing site.

    Return:
        (azim, alt, ang_radius): [tuple of ndarrays] Azimuth (deg), geometric altitude (deg) and
            the angular radius (deg).

    """

    return bodyAltAz('moon', times, location)



def diskObscuration(sep, r_sun, r_moon):
    """ Compute the fraction of the solar disk which is covered by the Moon.

    The disks are treated as circles, and the area of their overlap is computed analytically.

    Arguments:
        sep: [float or ndarray] Angular separation of the centres of the disks (deg).
        r_sun: [float or ndarray] Angular radius of the Sun (deg).
        r_moon: [float or ndarray] Angular radius of the Moon (deg).

    Return:
        obscuration: [ndarray] Fraction of the solar disk which is covered, from 0 to 1.

    """

    sep = np.atleast_1d(np.asarray(sep, dtype=np.float64))
    r_sun = np.broadcast_to(np.asarray(r_sun, dtype=np.float64), sep.shape)
    r_moon = np.broadcast_to(np.asarray(r_moon, dtype=np.float64), sep.shape)

    obscuration = np.zeros_like(sep)

    # The disks do not overlap
    no_overlap = sep >= (r_sun + r_moon)

    # The Sun is completely covered
    total = sep <= (r_moon - r_sun)

    # The Moon is completely inside the solar disk
    annular = sep <= (r_sun - r_moon)

    partial = ~(no_overlap | total | annular)

    obscuration[total] = 1.0
    obscuration[annular] = (r_moon[annular]/r_sun[annular])**2

    if np.any(partial):

        d = sep[partial]
        rs = r_sun[partial]
        rm = r_moon[partial]

        # Half angles subtended by the chord of the intersection at the centre of each disk
        alpha = np.arccos(np.clip((d**2 + rs**2 - rm**2)/(2*d*rs), -1.0, 1.0))
        beta = np.arccos(np.clip((d**2 + rm**2 - rs**2)/(2*d*rm), -1.0, 1.0))

        # Area of the lens shaped intersection of the two disks
        area = (rs**2*(alpha - np.sin(2*alpha)/2) + rm**2*(beta - np.sin(2*beta)/2))

        obscuration[partial] = area/(np.pi*rs**2)


    return obscuration



def eclipseCircumstances(times, location):
    """ Compute the circumstances of the solar eclipse at the site.

    Arguments:
        times: [ndarray of datetime] Times in UTC.
        location: [EarthLocation] Location of the observing site.

    Return:
        circ: [dict] Dictionary with the positions of the Sun and the Moon, their angular radii,
            their separation and the obscuration of the solar disk.

    """

    sun_azim, sun_alt, sun_radius = sunAltAz(times, location)
    moon_azim, moon_alt, moon_radius = moonAltAz(times, location)

    sep = angularSeparation(sun_azim, sun_alt, moon_azim, moon_alt)

    obscuration = diskObscuration(sep, sun_radius, moon_radius)

    return {
        'time': times,
        'sun_azim': sun_azim,
        'sun_alt': sun_alt,
        'sun_radius': sun_radius,
        'moon_azim': moon_azim,
        'moon_alt': moon_alt,
        'moon_radius': moon_radius,
        'separation': sep,
        'obscuration': obscuration,
        }



if __name__ == "__main__":

    import argparse
    import datetime

    from SolarTransit.Config import SITE_LAT, SITE_LON, NOMINAL_TIME, siteElevation


    ### COMMAND LINE ARGUMENTS ###

    arg_parser = argparse.ArgumentParser(description="""Print the position of the Sun and the
        circumstances of the eclipse at the observing site. """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-w', '--window', metavar='MINUTES', type=float, default=8.0,
        help="Half width of the printed time window around the nominal time.")

    arg_parser.add_argument('-s', '--step', metavar='SECONDS', type=float, default=60.0,
        help="Time step of the printout.")

    cml_args = arg_parser.parse_args()

    #########################


    elevation = siteElevation()

    location = siteLocation(SITE_LAT, SITE_LON, elevation)

    print("Site: {:.6f} N, {:.6f} E, {:.1f} m (WGS84)".format(SITE_LAT, SITE_LON, elevation))

    # Generate the times of the printout
    n_steps = int(2*cml_args.window*60/cml_args.step) + 1

    times = np.array([NOMINAL_TIME - datetime.timedelta(minutes=cml_args.window)
        + datetime.timedelta(seconds=i*cml_args.step) for i in range(n_steps)])

    circ = eclipseCircumstances(times, location)

    print()
    print("{:<21s} {:>9s} {:>8s} {:>9s} {:>8s} {:>9s} {:>12s}".format("Time (UTC)", "Sun azim",
        "Sun alt", "Moon azim", "Moon alt", "Sep (deg)", "Obscuration"))

    for i, t in enumerate(times):
        print("{:<21s} {:9.3f} {:8.3f} {:9.3f} {:8.3f} {:9.4f} {:11.2f}%".format(str(t),
            circ['sun_azim'][i], circ['sun_alt'][i], circ['moon_azim'][i], circ['moon_alt'][i],
            circ['separation'][i], 100*circ['obscuration'][i]))
