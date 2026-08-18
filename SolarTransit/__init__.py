"""
Find the aircraft which crossed the apparent disk of the Sun, from historical ADS-B data.

The whole analysis is available in one call:

    from SolarTransit import ObservingSite, runPipeline

    site = ObservingSite(41.6488, -0.8891)
    results = runPipeline(site, datetime.datetime(2026, 8, 12, 18, 30))

The individual steps are in the modules of this package: ADSBData downloads the telemetry,
Trajectory interpolates the tracks, Refraction and Ephemeris provide the apparent positions,
TransitSearch does the search and PlotTransits draws the result.

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

__version__ = "1.0.0"

from SolarTransit.Site import ObservingSite
from SolarTransit.Pipeline import runPipeline, parseTime
from SolarTransit.TransitSearch import runSearch, searchTransits, writeCandidates
from SolarTransit.PlotTransits import plotCandidates
from SolarTransit.ADSBData import loadRegion
from SolarTransit.Ephemeris import eclipseCircumstances, sunAltAz, moonAltAz
from SolarTransit.Refraction import RefractionTable
from SolarTransit.Trajectory import buildTracks, FlightTrack


__all__ = [
    'ObservingSite',
    'runPipeline',
    'parseTime',
    'runSearch',
    'searchTransits',
    'writeCandidates',
    'plotCandidates',
    'loadRegion',
    'eclipseCircumstances',
    'sunAltAz',
    'moonAltAz',
    'RefractionTable',
    'buildTracks',
    'FlightTrack',
    ]
