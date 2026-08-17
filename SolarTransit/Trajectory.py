"""
Reconstruction of aircraft trajectories from discretized ADS-B reports.
Includes:
    - Conversion of the barometric altitude to the geometric height
    - Monotone cubic interpolation of the trajectory in ECEF coordinates
    - Estimation of the interpolation error by leaving samples out

The ADS-B feed reports the position of an aircraft about every 30 s. An airliner covers 7 km in
that time, which is far more than the size of the solar disk at the distance of the aircraft, so
the trajectory has to be interpolated and the error of the interpolation has to be known.

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
import scipy.interpolate

from SolarTransit.Conversions import latLonAlt2ECEF


### CONSTANTS ###

# Conversion from feet to meters
FT2M = 0.3048

# Radius used to convert the geopotential height into the geometric height (m)
GEOPOTENTIAL_RADIUS = 6356766.0

# Wingspans of common aircraft types (m). Used to compute the angular size of the silhouette.
WINGSPANS = {
    'A19N': 35.8, 'A20N': 35.8, 'A21N': 35.8, 'A318': 34.1, 'A319': 35.8, 'A320': 35.8,
    'A321': 35.8, 'A332': 60.3, 'A333': 60.3, 'A338': 60.3, 'A339': 64.0, 'A342': 60.3,
    'A343': 60.3, 'A345': 63.5, 'A346': 63.5, 'A359': 64.8, 'A35K': 64.8, 'A388': 79.8,
    'AT43': 24.6, 'AT45': 24.6, 'AT72': 27.1, 'AT75': 27.1, 'AT76': 27.1,
    'B37M': 35.9, 'B38M': 35.9, 'B39M': 35.9, 'B734': 28.9, 'B735': 28.9, 'B736': 34.3,
    'B737': 34.3, 'B738': 35.8, 'B739': 35.8, 'B744': 64.4, 'B748': 68.4, 'B752': 38.1,
    'B753': 38.1, 'B762': 47.6, 'B763': 47.6, 'B764': 51.9, 'B772': 60.9, 'B773': 64.8,
    'B77L': 64.8, 'B77W': 64.8, 'B788': 60.1, 'B789': 60.1, 'B78X': 60.1,
    'BCS1': 35.1, 'BCS3': 35.1, 'C25A': 15.9, 'C25B': 15.9, 'C25C': 17.2, 'C56X': 17.2,
    'CRJ2': 21.2, 'CRJ7': 23.2, 'CRJ9': 24.9, 'CRJX': 26.2,
    'E170': 26.0, 'E175': 26.0, 'E190': 28.7, 'E195': 28.7, 'E290': 33.7, 'E295': 35.1,
    'E50P': 12.5, 'E55P': 14.3, 'F2TH': 19.3, 'FA7X': 26.2, 'GLEX': 28.7, 'GLF5': 28.5,
    'GLF6': 29.0, 'H25B': 15.7, 'LJ45': 14.6, 'MD11': 51.7, 'PC12': 16.3, 'SB20': 24.8,
    'SF34': 21.4,
    }

# Wingspan used when the aircraft type is not known (m)
DEFAULT_WINGSPAN = 35.0

### ###



def baroToGeometricHeight(altitude_baro_ft):
    """ Convert the reported barometric altitude to the geometric height above sea level.

    The barometric altitude is a pressure altitude referenced to the standard pressure of
    1013.25 hPa, and is reported as a geopotential height. It is converted here to a geometric
    height under the assumption of a standard atmosphere. The difference between the pressure
    altitude and the true height caused by the actual weather is not corrected, and is carried
    through the analysis as an uncertainty instead.

    Arguments:
        altitude_baro_ft: [float or ndarray] Barometric altitude (ft).

    Return:
        height: [float or ndarray] Geometric height above sea level (m).

    """

    # Geopotential height in meters
    h_geopotential = np.asarray(altitude_baro_ft, dtype=np.float64)*FT2M

    # Convert the geopotential height into the geometric height
    return GEOPOTENTIAL_RADIUS*h_geopotential/(GEOPOTENTIAL_RADIUS - h_geopotential)



def wingspan(aircraft_type):
    """ Return the wingspan of the given aircraft type.

    Arguments:
        aircraft_type: [str] ICAO aircraft type designator.

    Return:
        span: [float] Wingspan (m).

    """

    if aircraft_type is None:
        return DEFAULT_WINGSPAN

    return WINGSPANS.get(str(aircraft_type).strip().upper(), DEFAULT_WINGSPAN)



class FlightTrack(object):
    """ Interpolated trajectory of a single flight.

    The positions are interpolated in ECEF coordinates using monotone cubic (PCHIP) splines. The
    interpolation is done in the Cartesian frame, so that it does not suffer from the distortion
    of the geographic coordinates, and monotone splines are used because they do not overshoot
    between the widely spaced samples.

    """

    def __init__(self, df_flight, time_ref):
        """
        Arguments:
            df_flight: [pandas.DataFrame] ADS-B reports of a single flight, sorted by time.
            time_ref: [datetime] Reference time to which the interpolation times are relative.

        """

        self.time_ref = time_ref

        # Metadata of the flight, taken from the last report which has it
        self.flight_id = self.firstValid(df_flight, 'flight_id')
        self.icao_address = self.firstValid(df_flight, 'icao_address')
        self.callsign = self.firstValid(df_flight, 'callsign')
        self.tail_number = self.firstValid(df_flight, 'tail_number')
        self.flight_number = self.firstValid(df_flight, 'flight_number')
        self.aircraft_type = self.firstValid(df_flight, 'aircraft_type_icao')
        self.airline = self.firstValid(df_flight, 'airline_iata')
        self.departure = self.firstValid(df_flight, 'departure_airport_icao')
        self.arrival = self.firstValid(df_flight, 'arrival_airport_icao')

        self.wingspan = wingspan(self.aircraft_type)

        # Times of the reports, in seconds relative to the reference time
        self.t_data = (df_flight['timestamp'].to_numpy().astype('datetime64[us]')
            - np.datetime64(time_ref, 'us')).astype(np.float64)/1e6

        self.lat_data = df_flight['latitude'].to_numpy(dtype=np.float64)
        self.lon_data = df_flight['longitude'].to_numpy(dtype=np.float64)
        self.baro_data = df_flight['altitude_baro'].to_numpy(dtype=np.float64)

        # Geometric height above sea level
        self.height_data = baroToGeometricHeight(self.baro_data)

        # ECEF coordinates of the reports
        x, y, z = latLonAlt2ECEF(np.radians(self.lat_data), np.radians(self.lon_data),
            self.height_data)

        self.ecef_data = np.vstack([x, y, z])

        # Interpolators of the ECEF coordinates and of the barometric altitude
        self.interp_ecef = [scipy.interpolate.PchipInterpolator(self.t_data, comp,
            extrapolate=False) for comp in self.ecef_data]

        self.interp_baro = scipy.interpolate.PchipInterpolator(self.t_data, self.baro_data,
            extrapolate=False)

        self.t_beg = self.t_data[0]
        self.t_end = self.t_data[-1]



    @staticmethod
    def firstValid(df_flight, column):
        """ Return the first value of the column which is not empty.

        Arguments:
            df_flight: [pandas.DataFrame] Reports of a single flight.
            column: [str] Name of the column.

        Return:
            value: [str or None] The first non-empty value, or None if there is none.

        """

        if column not in df_flight.columns:
            return None

        values = df_flight[column].dropna()

        values = [str(v).strip() for v in values if str(v).strip() not in ('', 'None', 'nan')]

        if not values:
            return None

        return values[0]



    def positionsECEF(self, t_rel):
        """ Interpolate the position of the aircraft.

        Arguments:
            t_rel: [ndarray] Times in seconds relative to the reference time.

        Return:
            (x, y, z): [tuple of ndarrays] ECEF coordinates (m). Times outside the span of the
                reports give NaN.

        """

        return tuple(interp(t_rel) for interp in self.interp_ecef)



    def baroAltitude(self, t_rel):
        """ Interpolate the barometric altitude of the aircraft.

        Arguments:
            t_rel: [ndarray] Times in seconds relative to the reference time.

        Return:
            altitude: [ndarray] Barometric altitude (ft).

        """

        return self.interp_baro(t_rel)



    def sampleCadence(self, t_rel):
        """ Return the length of the interval between the two reports which bracket the time.

        Arguments:
            t_rel: [float] Time in seconds relative to the reference time.

        Return:
            (gap, cadence_median): [tuple of floats] Length of the bracketing interval and the
                median interval of the whole track, both in seconds.

        """

        cadence_median = float(np.median(np.diff(self.t_data))) if len(self.t_data) > 1 else np.nan

        if (t_rel < self.t_beg) or (t_rel > self.t_end):
            return np.nan, cadence_median

        idx = int(np.searchsorted(self.t_data, t_rel))
        idx = min(max(idx, 1), len(self.t_data) - 1)

        gap = float(self.t_data[idx] - self.t_data[idx - 1])

        return gap, cadence_median



    def interpolationError(self):
        """ Estimate the error of the interpolation by leaving samples out.

        Every interior report is removed in turn, the trajectory is interpolated from the
        remaining reports, and the position of the removed report is compared with the
        interpolated one. This measures how well the interpolation follows the real motion of the
        aircraft, including turns.

        Return:
            (err_rms, err_max): [tuple of floats] Root mean square and maximum position error (m).
                NaN is returned if the track is too short for the test.

        """

        n_points = len(self.t_data)

        if n_points < 5:
            return np.nan, np.nan


        errors = []

        for i in range(1, n_points - 1):

            mask = np.ones(n_points, dtype=bool)
            mask[i] = False

            # Interpolate the trajectory without the left out report
            interp = [scipy.interpolate.PchipInterpolator(self.t_data[mask], comp[mask],
                extrapolate=False) for comp in self.ecef_data]

            pos_interp = np.array([f(self.t_data[i]) for f in interp])

            errors.append(np.linalg.norm(pos_interp - self.ecef_data[:, i]))


        errors = np.array(errors)

        # The leave one out test doubles the length of the interpolated interval, so the error of
        # the interpolation over the real interval is smaller. The scaling of the cubic
        # interpolation error with the interval length is used to correct for this.
        scaling = 1.0/(2.0**3)

        return float(scaling*np.sqrt(np.mean(errors**2))), float(scaling*np.max(errors))



def buildTracks(df, time_ref, min_points=3):
    """ Build interpolated tracks for all flights in the dataframe.

    Arguments:
        df: [pandas.DataFrame] ADS-B reports.
        time_ref: [datetime] Reference time of the interpolation.

    Keyword arguments:
        min_points: [int] Flights with fewer reports than this are skipped.

    Return:
        tracks: [list of FlightTrack] Interpolated tracks.

    """

    tracks = []

    for flight_id, df_flight in df.groupby('flight_id', sort=False):

        df_flight = df_flight.sort_values('timestamp')

        # Remove duplicated times, which would break the interpolation
        df_flight = df_flight.drop_duplicates(subset='timestamp', keep='first')

        if len(df_flight) < min_points:
            continue

        tracks.append(FlightTrack(df_flight, time_ref))


    return tracks
