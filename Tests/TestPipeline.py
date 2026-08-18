"""
Tests of the observing site and of the command line interface of the pipeline.

Run as:
    python Tests/TestPipeline.py

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
import shutil
import datetime
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SolarTransit.Conversions import latLonAlt2ECEF, ECEF2AltAz
from SolarTransit.Pipeline import parseTime
from SolarTransit.Site import ObservingSite
from SolarTransit.SiteElevation import cacheKey, readElevationCache, writeElevationCache


# Site used in the tests
TEST_LAT = 41.6488
TEST_LON = -0.8891
TEST_GROUND_ELEVATION = 277.3



class TestSite(unittest.TestCase):
    """ Tests of the observing site. """


    def testElevation(self):
        """ The elevation of the observer has to include the height of the eye. """

        site = ObservingSite(TEST_LAT, TEST_LON, ground_elevation=TEST_GROUND_ELEVATION,
            eye_height=1.6)

        self.assertAlmostEqual(site.elevation, TEST_GROUND_ELEVATION + 1.6, places=6)

        # A site whose elevation is given must not need a lookup
        self.assertEqual(site.elevation_source, 'given')



    def testECEF(self):
        """ The site has to give the same ECEF coordinates as the conversion itself. """

        site = ObservingSite(TEST_LAT, TEST_LON, ground_elevation=TEST_GROUND_ELEVATION)

        expected = latLonAlt2ECEF(np.radians(TEST_LAT), np.radians(TEST_LON), site.elevation)

        for value, value_expected in zip(site.ecef(), expected):
            self.assertAlmostEqual(value, value_expected, places=6)


        # A target straight above the site, along the normal of the ellipsoid, has to be at the
        # zenith
        up = latLonAlt2ECEF(np.radians(TEST_LAT), np.radians(TEST_LON), site.elevation + 10000.0)

        _, alt = ECEF2AltAz(site.ecef(), up)

        self.assertAlmostEqual(float(alt), 90.0, places=6)



    def testBox(self):
        """ The box has to be centred on the site. """

        site = ObservingSite(TEST_LAT, TEST_LON, ground_elevation=TEST_GROUND_ELEVATION)

        lat_min, lat_max, lon_min, lon_max = site.box(margin_lat=2.0, margin_lon=3.0)

        self.assertAlmostEqual(lat_min, TEST_LAT - 2.0, places=6)
        self.assertAlmostEqual(lat_max, TEST_LAT + 2.0, places=6)
        self.assertAlmostEqual(lon_min, TEST_LON - 3.0, places=6)
        self.assertAlmostEqual(lon_max, TEST_LON + 3.0, places=6)



    def testElevationCache(self):
        """ The cache has to hold an entry per site, and has to be found again.

        Two sites are written into the same cache, and both have to survive, so that switching
        between sites does not force a new lookup of the elevation.

        """

        cache_dir = tempfile.mkdtemp()

        try:
            cache_path = os.path.join(cache_dir, 'site_elevation.json')

            first = {'latitude': TEST_LAT, 'longitude': TEST_LON, 'elevation_msl': 227.2,
                'elevation_wgs84': TEST_GROUND_ELEVATION, 'geoid_undulation': 50.1,
                'geoid_source': 'EGM96', 'dem_dataset': 'eudem25m'}

            second = dict(first)
            second.update({'latitude': 0.0, 'longitude': 0.0, 'elevation_wgs84': 17.0})

            writeElevationCache(first, cache_path)
            writeElevationCache(second, cache_path)

            # Both sites have to be in the cache
            self.assertAlmostEqual(readElevationCache(TEST_LAT, TEST_LON,
                cache_path=cache_path)['elevation_wgs84'], TEST_GROUND_ELEVATION, places=3)

            self.assertAlmostEqual(readElevationCache(0.0, 0.0,
                cache_path=cache_path)['elevation_wgs84'], 17.0, places=3)

            # A site which was never looked up is not in the cache
            self.assertIsNone(readElevationCache(10.0, 10.0, cache_path=cache_path))

            # The site given to the site object has to be taken from the cache, without a lookup
            site = ObservingSite(TEST_LAT, TEST_LON, elevation_cache=cache_path)

            self.assertAlmostEqual(site.resolveElevation(verbose=False), TEST_GROUND_ELEVATION,
                places=3)

            self.assertIn('cached', site.elevation_source)

            # The keys have to be distinct
            with open(cache_path) as f:
                cache = json.load(f)

            self.assertEqual(len(cache), 2)
            self.assertIn(cacheKey(TEST_LAT, TEST_LON), cache)


        finally:
            shutil.rmtree(cache_dir)



class TestPipelineInterface(unittest.TestCase):
    """ Tests of the command line interface of the pipeline. """


    def testParseTime(self):
        """ The time of the event has to be accepted in the documented formats. """

        expected = datetime.datetime(2026, 8, 12, 18, 30, 0)

        for time_str in ["2026-08-12 18:30:00", "2026-08-12T18:30:00", "2026-08-12 18:30",
            "2026-08-12T18:30", "  2026-08-12 18:30:00  "]:

            self.assertEqual(parseTime(time_str), expected)



    def testParseTimeRejectsGarbage(self):
        """ A time which cannot be parsed has to raise an error, and not be guessed. """

        for time_str in ["yesterday", "12/08/2026 18:30", "18:30:00", ""]:

            with self.assertRaises(ValueError):
                parseTime(time_str)



    def testPipelineHelp(self):
        """ The entry points have to be importable and have to expose a main function. """

        from SolarTransit import ADSBData, Ephemeris, PlotTransits, Pipeline, SiteElevation
        from SolarTransit import TransitSearch

        for module in [ADSBData, Ephemeris, PlotTransits, Pipeline, SiteElevation, TransitSearch]:
            self.assertTrue(callable(getattr(module, 'main', None)),
                "{:s} has no main function".format(module.__name__))



if __name__ == "__main__":

    unittest.main(verbosity=2)
