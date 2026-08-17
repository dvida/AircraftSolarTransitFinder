"""
Tests of the geometry, the refraction and the ephemeris.

Run as:
    python Tests/TestGeometry.py

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
import datetime
import unittest

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import AltAz

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SolarTransit.Conversions import (latLonAlt2ECEF, ecef2LatLonAlt, ECEF2AltAz,
    angularSeparation, datetime2JD)
from SolarTransit.Ephemeris import siteLocation, sunAltAz, diskObscuration
from SolarTransit.Refraction import (astronomicalRefraction, bennettRefraction, RefractionTable,
    isaDensity, ISA_RHO0)
from SolarTransit.Trajectory import baroToGeometricHeight, FlightTrack


# Site used in the tests. It is fixed here, and deliberately not taken from the configuration,
# so that the tests do not depend on the site which the user happens to have configured.
TEST_LAT = 41.6488
TEST_LON = -0.8891
TEST_ELEVATION = 250.0

# Time used in the tests, in the middle of the total solar eclipse of 2026 August 12
TEST_TIME = datetime.datetime(2026, 8, 12, 18, 30, 0)



def lowPrecisionSunAltAz(dt, lat, lon):
    """ Compute the position of the Sun with the classical low precision formulae.

    The formulae are accurate to about 0.01 deg, and are used here as a check of the ephemeris
    which is completely independent of astropy.

    Arguments:
        dt: [datetime] Time in UTC.
        lat: [float] Latitude of the site (deg, +north).
        lon: [float] Longitude of the site (deg, +east).

    Return:
        (azim, alt): [tuple of floats] Azimuth and altitude of the Sun (deg).

    """

    # Days since the J2000.0 epoch
    n = datetime2JD(dt) - 2451545.0

    # Mean longitude and mean anomaly of the Sun
    mean_lon = np.radians((280.460 + 0.9856474*n)%360)
    mean_anomaly = np.radians((357.528 + 0.9856003*n)%360)

    # Ecliptic longitude
    ecl_lon = mean_lon + np.radians(1.915)*np.sin(mean_anomaly) \
        + np.radians(0.020)*np.sin(2*mean_anomaly)

    # Obliquity of the ecliptic
    obliquity = np.radians(23.439 - 0.0000004*n)

    # Equatorial coordinates
    ra = np.arctan2(np.cos(obliquity)*np.sin(ecl_lon), np.cos(ecl_lon))
    dec = np.arcsin(np.sin(obliquity)*np.sin(ecl_lon))

    # Local sidereal time
    gmst = (18.697374558 + 24.06570982441908*n)%24
    lst = np.radians((gmst*15 + lon)%360)

    hour_angle = lst - ra

    lat_rad = np.radians(lat)

    alt = np.arcsin(np.sin(lat_rad)*np.sin(dec)
        + np.cos(lat_rad)*np.cos(dec)*np.cos(hour_angle))

    azim = np.arctan2(-np.sin(hour_angle),
        np.tan(dec)*np.cos(lat_rad) - np.sin(lat_rad)*np.cos(hour_angle))

    return np.degrees(azim)%360, np.degrees(alt)



class TestConversions(unittest.TestCase):
    """ Tests of the coordinate conversions. """


    def testECEFRoundTrip(self):
        """ The geodetic coordinates have to survive the conversion to ECEF and back. """

        for lat, lon, h in [(TEST_LAT, TEST_LON, TEST_ELEVATION), (0.0, 0.0, 0.0),
            (60.0, 120.0, 11000.0),
            (-33.5, -70.6, 500.0)]:

            x, y, z = latLonAlt2ECEF(np.radians(lat), np.radians(lon), h)

            lat_back, lon_back, h_back = ecef2LatLonAlt(x, y, z)

            self.assertAlmostEqual(lat, np.degrees(lat_back), places=8)
            self.assertAlmostEqual(lon, np.degrees(lon_back), places=8)
            self.assertAlmostEqual(h, h_back, places=3)



    def testAltAzAgainstProj(self):
        """ The horizontal coordinates have to agree with the ones computed by PROJ.

        Targets are placed at a range of distances and directions around the site, and the
        directions towards them are computed both by the vendored conversion and by the
        topocentric conversion of PROJ.

        The horizontal coordinates of a nearby target are deliberately not compared against the
        ITRS to AltAz transformation of astropy. That transformation is meant for celestial
        sources and displaces a target at a finite distance by several hundred meters, which is
        degrees for a target which is only a few tens of kilometers away. It is harmless for the
        Sun, which is why astropy is still used for the ephemeris.

        """

        from pyproj import Transformer

        transformer = Transformer.from_pipeline(
            "+proj=pipeline +step +proj=topocentric +ellps=WGS84 "
            "+lat_0={:.9f} +lon_0={:.9f} +h_0={:.4f}".format(TEST_LAT, TEST_LON, TEST_ELEVATION))

        site_ecef = latLonAlt2ECEF(np.radians(TEST_LAT), np.radians(TEST_LON), TEST_ELEVATION)

        # Targets at different distances, altitudes and azimuths
        for dist in [10e3, 100e3, 300e3]:
            for azim_true in [0.0, 90.0, 284.2, 350.0]:
                for alt_true in [1.0, 6.2, 45.0]:

                    # Position of the target in the local frame of the site
                    east = dist*np.cos(np.radians(alt_true))*np.sin(np.radians(azim_true))
                    north = dist*np.cos(np.radians(alt_true))*np.cos(np.radians(azim_true))
                    up = dist*np.sin(np.radians(alt_true))

                    # Rotate the local frame into the ECEF frame
                    lat_rad = np.radians(TEST_LAT)
                    lon_rad = np.radians(TEST_LON)

                    dx = (-np.sin(lon_rad)*east - np.sin(lat_rad)*np.cos(lon_rad)*north
                        + np.cos(lat_rad)*np.cos(lon_rad)*up)
                    dy = (np.cos(lon_rad)*east - np.sin(lat_rad)*np.sin(lon_rad)*north
                        + np.cos(lat_rad)*np.sin(lon_rad)*up)
                    dz = np.cos(lat_rad)*north + np.sin(lat_rad)*up

                    target_ecef = (site_ecef[0] + dx, site_ecef[1] + dy, site_ecef[2] + dz)

                    azim, alt = ECEF2AltAz(site_ecef, target_ecef)

                    # The same direction computed by PROJ
                    east_proj, north_proj, up_proj = transformer.transform(*target_ecef)

                    azim_proj = np.degrees(np.arctan2(east_proj, north_proj))%360
                    alt_proj = np.degrees(np.arctan2(up_proj, np.hypot(east_proj, north_proj)))

                    sep = angularSeparation(azim, alt, azim_proj, alt_proj)

                    self.assertLess(sep, 0.001, "The direction towards a target at {:.0f} km, "
                        "azimuth {:.1f} deg and altitude {:.1f} deg differs from PROJ by "
                        "{:.4f} deg".format(dist/1000, azim_true, alt_true, sep))

                    # The conversion also has to return the direction which was put in
                    self.assertLess(angularSeparation(azim, alt, azim_true, alt_true), 0.001)



    def testJulianDate(self):
        """ The Julian date has to match the value computed by astropy. """

        for dt in [datetime.datetime(2000, 1, 1, 12), TEST_TIME,
            datetime.datetime(2026, 8, 12, 18, 28, 30, 500000)]:

            self.assertAlmostEqual(datetime2JD(dt), Time(dt, scale='utc').jd, places=8)



class TestRefraction(unittest.TestCase):
    """ Tests of the refraction model. """


    def testStandardAtmosphere(self):
        """ The standard atmosphere has to reproduce the known densities. """

        self.assertAlmostEqual(float(isaDensity(0.0)[0]), ISA_RHO0, places=3)

        # Density at 11 km, the top of the troposphere
        self.assertAlmostEqual(float(isaDensity(11000.0)[0]), 0.3639, places=3)



    def testAgainstBennett(self):
        """ At an infinite distance the ray tracing has to reproduce Bennett's formula.

        Bennett's formula is defined for a slightly different sea level pressure and temperature
        than the standard atmosphere, so a few per cent of difference is expected.

        """

        for elev in [5.0, 6.2, 10.0, 20.0, 45.0]:

            traced = astronomicalRefraction(elev, 0.0)
            bennett = bennettRefraction(elev)

            self.assertLess(abs(traced - bennett)/bennett, 0.05,
                "The traced refraction of {:.4f} deg at an elevation of {:.1f} deg differs from "
                "Bennett's {:.4f} deg by more than 5 per cent".format(traced, elev, bennett))



    def testRefractionDecreasesWithHeight(self):
        """ A target inside the atmosphere has to be refracted less than an astronomical source.

        This is the effect which has to be modelled, because an aircraft at a cruising altitude is
        above most of the refracting air.

        """

        table = RefractionTable(TEST_ELEVATION)

        elev_geometric = 6.2

        refr_astronomical = (table.apparentElevationAstronomical(elev_geometric)
            - elev_geometric)

        refr_aircraft = float(table.apparentElevationTarget(elev_geometric, 11000.0)[0]
            - elev_geometric)

        self.assertGreater(refr_astronomical, refr_aircraft)

        # The difference at this elevation is a few hundredths of a degree, which is a
        # considerable fraction of the solar radius
        self.assertGreater(refr_astronomical - refr_aircraft, 0.02)
        self.assertLess(refr_astronomical - refr_aircraft, 0.10)



class TestEphemeris(unittest.TestCase):
    """ Tests of the ephemeris and of the eclipse geometry. """


    def testSolarPosition(self):
        """ The position of the Sun has to agree with an independent low precision algorithm.

        The ephemeris of astropy is checked against the classical low precision formulae for the
        position of the Sun, which are accurate to about 0.01 deg. Several sites and times are
        used, so the test does not depend on the site which happens to be configured.

        """

        for lat, lon in [(TEST_LAT, TEST_LON), (0.0, 0.0), (60.0, 25.0), (-33.9, 151.2)]:
            for hours in [0.0, 6.0, 13.5, 21.0]:

                dt = TEST_TIME + datetime.timedelta(hours=hours)

                location = siteLocation(lat, lon, TEST_ELEVATION)

                azim, alt, radius = sunAltAz(np.array([dt]), location)

                azim_ref, alt_ref = lowPrecisionSunAltAz(dt, lat, lon)

                # The Sun is only compared when it is well above the horizon, because the low
                # precision formulae do not include the parallax, which matters near the horizon
                if alt_ref < 10.0:
                    continue


                sep = angularSeparation(float(azim[0]), float(alt[0]), azim_ref, alt_ref)

                self.assertLess(sep, 0.05, "The position of the Sun at {:s} UTC seen from "
                    "{:.1f}, {:.1f} differs from the low precision formulae by {:.4f} "
                    "deg".format(str(dt), lat, lon, sep))

                # Angular radius of the Sun, which varies between 0.262 and 0.271 deg
                self.assertGreater(float(radius[0]), 0.260)
                self.assertLess(float(radius[0]), 0.273)



    def testObscuration(self):
        """ The overlap of the two disks has to be computed correctly. """

        # The disks do not touch
        self.assertAlmostEqual(float(diskObscuration(2.0, 0.5, 0.5)[0]), 0.0, places=6)

        # The Sun is completely covered
        self.assertAlmostEqual(float(diskObscuration(0.0, 0.5, 0.6)[0]), 1.0, places=6)

        # The Moon is entirely inside the solar disk, which is the annular case
        self.assertAlmostEqual(float(diskObscuration(0.0, 1.0, 0.5)[0]), 0.25, places=6)

        # Two equal disks whose centres coincide
        self.assertAlmostEqual(float(diskObscuration(0.0, 0.5, 0.5)[0]), 1.0, places=6)

        # A partial phase has to be between zero and one, and has to grow as the disks approach
        obsc_far = float(diskObscuration(0.9, 0.5, 0.5)[0])
        obsc_near = float(diskObscuration(0.4, 0.5, 0.5)[0])

        self.assertGreater(obsc_near, obsc_far)
        self.assertGreater(obsc_far, 0.0)
        self.assertLess(obsc_near, 1.0)



class TestTrajectory(unittest.TestCase):
    """ Tests of the trajectory interpolation. """


    def testBaroConversion(self):
        """ The barometric altitude has to be converted into a slightly larger geometric height.
        """

        # A cruising altitude of 36000 ft
        height = baroToGeometricHeight(36000.0)

        self.assertGreater(height, 36000*0.3048)
        self.assertLess(height, 36000*0.3048 + 100)



    def testInterpolationOfAStraightTrack(self):
        """ A track flown at a constant speed along a straight line has to be recovered exactly.

        The leave one out test is run on a synthetic track, so that the error of the interpolation
        can be compared with a known answer.

        """

        import pandas as pd

        time_ref = TEST_TIME

        # A synthetic flight which crosses the region at a constant speed and altitude
        n_points = 20
        times = [time_ref + datetime.timedelta(seconds=30*i) for i in range(n_points)]

        lats = 40.0 + 0.02*np.arange(n_points)
        lons = -3.0 + 0.06*np.arange(n_points)

        df_flight = pd.DataFrame({
            'timestamp': pd.to_datetime(times),
            'latitude': lats,
            'longitude': lons,
            'altitude_baro': np.full(n_points, 36000.0),
            'flight_id': 'test',
            'icao_address': 'ABCDEF',
            'callsign': 'TEST123',
            'tail_number': 'N123AB',
            'flight_number': 'TS123',
            'aircraft_type_icao': 'A320',
            'airline_iata': 'TS',
            'departure_airport_icao': 'LEMD',
            'arrival_airport_icao': 'EDDM',
            })

        track = FlightTrack(df_flight, time_ref)

        err_rms, err_max = track.interpolationError()

        # A smooth track has to be interpolated to much better than the size of the aircraft
        self.assertLess(err_rms, 10.0)
        self.assertLess(err_max, 30.0)

        # The wingspan of the type has to be recognized
        self.assertAlmostEqual(track.wingspan, 35.8, places=1)



if __name__ == "__main__":

    unittest.main(verbosity=2)
