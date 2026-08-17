"""
Atmospheric refraction for targets at a finite distance.
Includes:
    - International Standard Atmosphere profile
    - Ray tracing through a spherically symmetric atmosphere
    - A lookup table of refraction as a function of the geometric elevation and the height of
      the target

An aircraft at a cruising altitude of 11 km sits above most of the refracting atmosphere, so its
light is bent less than the light of an astronomical object at the same apparent elevation. At an
elevation of 6 deg the difference is about 0.05 deg, which is a fifth of the solar diameter, so
the Sun and the aircraft cannot be compared without treating both consistently.

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
import scipy.integrate
import scipy.interpolate


### CONSTANTS ###

# Sea level values of the International Standard Atmosphere
ISA_T0 = 288.15      # K
ISA_P0 = 101325.0    # Pa
ISA_RHO0 = 1.225     # kg/m3
ISA_G = 9.80665      # m/s2
ISA_R = 287.0528     # J/(kg K), specific gas constant of dry air

# Layers of the standard atmosphere: base height (m), base temperature (K), lapse rate (K/m)
ISA_LAYERS = [
    (0.0,     288.15, -0.0065),
    (11000.0, 216.65,  0.0),
    (20000.0, 216.65,  0.001),
    (32000.0, 228.65,  0.0028),
    (47000.0, 270.65,  0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
    ]

# Refractivity of air at sea level for visible light. The refractivity scales with the density,
# so n - 1 = REFRACTIVITY_CONST*rho.
REFRACTIVITY_SEA_LEVEL = 2.77e-4
REFRACTIVITY_CONST = REFRACTIVITY_SEA_LEVEL/ISA_RHO0

# Height above which the atmosphere is neglected (m)
ATMOSPHERE_TOP = 100000.0

# Mean radius of the Earth used for the spherically symmetric ray tracing (m)
EARTH_MEAN_RADIUS = 6371000.0

### ###



def isaDensity(h):
    """ Compute the air density in the International Standard Atmosphere.

    Arguments:
        h: [float or ndarray] Geopotential height above sea level (m).

    Return:
        rho: [float or ndarray] Air density (kg/m3).

    """

    h = np.atleast_1d(np.asarray(h, dtype=np.float64))

    # Above the modelled atmosphere the density is taken as zero
    rho = np.zeros_like(h)

    # Pressure and temperature at the base of the current layer
    p_base = ISA_P0
    t_base = ISA_T0

    for i, (h_base, t_layer_base, lapse) in enumerate(ISA_LAYERS):

        # Height of the top of the layer
        if i + 1 < len(ISA_LAYERS):
            h_top = ISA_LAYERS[i + 1][0]

        else:
            h_top = ATMOSPHERE_TOP


        mask = (h >= h_base) & (h < h_top)

        if np.any(mask):

            dh = h[mask] - h_base

            # Temperature and pressure inside the layer
            if lapse == 0.0:
                t_layer = t_layer_base
                p_layer = p_base*np.exp(-ISA_G*dh/(ISA_R*t_layer_base))

            else:
                t_layer = t_layer_base + lapse*dh
                p_layer = p_base*(t_layer/t_layer_base)**(-ISA_G/(ISA_R*lapse))


            rho[mask] = p_layer/(ISA_R*t_layer)


        # Propagate the pressure and the temperature to the base of the next layer
        dh_layer = h_top - h_base

        if lapse == 0.0:
            p_base = p_base*np.exp(-ISA_G*dh_layer/(ISA_R*t_base))

        else:
            t_top = t_base + lapse*dh_layer
            p_base = p_base*(t_top/t_base)**(-ISA_G/(ISA_R*lapse))
            t_base = t_top


    return rho



def refractiveIndex(h):
    """ Compute the refractive index of air at the given height.

    Arguments:
        h: [float or ndarray] Height above sea level (m).

    Return:
        n: [float or ndarray] Refractive index.

    """

    return 1.0 + REFRACTIVITY_CONST*isaDensity(h)



def traceRay(elev_apparent, observer_height, r_eval):
    """ Trace a light ray through a spherically symmetric atmosphere.

    The ray is shot from the observer at the given apparent elevation and followed upwards. In a
    spherically symmetric medium the quantity n(r)*r*cos(E) is conserved along the ray, where E
    is the local elevation angle above the horizontal.

    Arguments:
        elev_apparent: [float] Apparent elevation of the ray at the observer (deg).
        observer_height: [float] Height of the observer above sea level (m).
        r_eval: [ndarray] Geocentric radii at which the ray position is evaluated (m).

    Return:
        theta: [ndarray] Geocentric angle between the observer and the ray position (rad).

    """

    r_obs = EARTH_MEAN_RADIUS + observer_height

    # The conserved quantity of the ray
    n_obs = float(refractiveIndex(observer_height)[0])
    invariant = n_obs*r_obs*np.cos(np.radians(elev_apparent))


    def dThetaDr(r, _):
        """ Rate of change of the geocentric angle with the radius. """

        n = refractiveIndex(r - EARTH_MEAN_RADIUS)

        cos_e = invariant/(n*r)

        # Guard against the ray turning around due to numerical noise
        cos_e = np.clip(cos_e, -1.0, 1.0 - 1e-15)

        sin_e = np.sqrt(1.0 - cos_e**2)

        return cos_e/(r*sin_e)


    sol = scipy.integrate.solve_ivp(dThetaDr, (r_obs, r_eval[-1]), [0.0], t_eval=r_eval,
        rtol=1e-10, atol=1e-12, method='DOP853')

    if (not sol.success) or (len(sol.y) == 0):
        raise RuntimeError("The ray tracing failed for the apparent elevation of {:.3f} deg. "
            "Rays which start below the horizontal cannot be traced upwards.".format(
            elev_apparent))

    return sol.y[0]



def geometricFromApparent(elev_apparent, observer_height, target_heights):
    """ Compute the geometric elevation of targets which are seen at the given apparent elevation.

    A single ray is traced from the observer, and the geometric (straight line) elevation of every
    point along it is computed. Those are the elevations at which targets at the corresponding
    heights would be if the atmosphere were not there.

    Arguments:
        elev_apparent: [float] Apparent elevation of the ray at the observer (deg).
        observer_height: [float] Height of the observer above sea level (m).
        target_heights: [ndarray] Heights of the targets above sea level (m). The last entry is
            treated as the top of the atmosphere, and the direction of the ray there is the
            geometric elevation of an astronomical source.

    Return:
        elev_geometric: [ndarray] Geometric elevations of the targets (deg).

    """

    r_obs = EARTH_MEAN_RADIUS + observer_height
    r_eval = EARTH_MEAN_RADIUS + np.asarray(target_heights, dtype=np.float64)

    theta = traceRay(elev_apparent, observer_height, r_eval)

    # Straight line elevation from the observer to the point reached by the ray
    elev_geometric = np.degrees(np.arctan2(r_eval*np.cos(theta) - r_obs, r_eval*np.sin(theta)))

    return elev_geometric



def astronomicalRefraction(elev_apparent, observer_height):
    """ Compute the refraction of an astronomical source, i.e. a source at an infinite distance.

    Arguments:
        elev_apparent: [float] Apparent elevation (deg).
        observer_height: [float] Height of the observer above sea level (m).

    Return:
        refraction: [float] Difference between the apparent and the geometric elevation (deg).

    """

    r_obs = EARTH_MEAN_RADIUS + observer_height
    r_top = EARTH_MEAN_RADIUS + ATMOSPHERE_TOP

    r_eval = np.linspace(r_obs, r_top, 2000)

    theta = traceRay(elev_apparent, observer_height, r_eval)

    n_obs = float(refractiveIndex(observer_height)[0])
    invariant = n_obs*r_obs*np.cos(np.radians(elev_apparent))

    # Local elevation of the ray at the top of the atmosphere, where the refractive index is unity
    cos_e_top = np.clip(invariant/r_top, -1.0, 1.0)
    elev_local_top = np.degrees(np.arccos(cos_e_top))

    # The direction of the ray in the frame of the observer. The local horizontal at the top of
    # the atmosphere is tilted by the geocentric angle.
    elev_out = elev_local_top - np.degrees(theta[-1])

    return elev_apparent - elev_out



class RefractionTable(object):
    """ Lookup table of the apparent elevation as a function of the geometric elevation and the
        height of the target.

    Rays are traced for a grid of apparent elevations. Because a single ray gives the geometric
    elevation of targets at all heights along it, the table is built with one integration per
    elevation. The table is then inverted, so that the apparent elevation can be looked up from
    the geometric one.

    """

    def __init__(self, observer_height, elev_min=0.5, elev_max=30.0, n_elev=119,
        height_max=20000.0, n_heights=41):
        """
        Arguments:
            observer_height: [float] Height of the observer above sea level (m).

        Keyword arguments:
            elev_min: [float] Lowest apparent elevation in the table (deg). Rays which start below
                the horizontal cannot be traced upwards, so this has to be positive.
            elev_max: [float] Highest apparent elevation in the table (deg).
            n_elev: [int] Number of elevation steps.
            height_max: [float] Highest target height in the table (m).
            n_heights: [int] Number of height steps.

        """

        self.observer_height = observer_height

        # Grid of target heights, starting just above the observer. The grid cannot start at the
        # height of the observer itself, because every ray has a zero geometric elevation there.
        self.heights = np.linspace(observer_height + 50.0, height_max, n_heights)

        # Grid of apparent elevations for which the rays are traced
        self.elev_apparent = np.linspace(elev_min, elev_max, n_elev)

        # Geometric elevations of the targets, one row per apparent elevation
        elev_geometric = np.zeros((n_elev, n_heights))

        for i, elev in enumerate(self.elev_apparent):
            elev_geometric[i] = geometricFromApparent(elev, observer_height, self.heights)

        self.elev_geometric = elev_geometric

        # Refraction of an astronomical source for the same grid of apparent elevations
        self.refraction_astronomical = np.array([astronomicalRefraction(elev, observer_height)
            for elev in self.elev_apparent])

        # Interpolator of the astronomical refraction as a function of the geometric elevation
        self.astronomical_interp = scipy.interpolate.interp1d(
            self.elev_apparent - self.refraction_astronomical, self.refraction_astronomical,
            kind='cubic', bounds_error=False, fill_value=(self.refraction_astronomical[0],
            self.refraction_astronomical[-1]))

        # Interpolators of the refraction of a target at a finite height, one per height
        self.target_interps = []

        for j in range(n_heights):

            refraction = self.elev_apparent - elev_geometric[:, j]

            self.target_interps.append(scipy.interpolate.interp1d(elev_geometric[:, j], refraction,
                kind='cubic', bounds_error=False, fill_value=(refraction[0], refraction[-1])))



    def apparentElevationTarget(self, elev_geometric, target_height):
        """ Compute the apparent elevation of a target at a finite height.

        Arguments:
            elev_geometric: [float or ndarray] Geometric elevation of the target (deg).
            target_height: [float or ndarray] Height of the target above sea level (m).

        Return:
            elev_apparent: [float or ndarray] Apparent elevation of the target (deg).

        """

        elev_geometric = np.atleast_1d(np.asarray(elev_geometric, dtype=np.float64))
        target_height = np.broadcast_to(np.asarray(target_height, dtype=np.float64),
            elev_geometric.shape)

        # Evaluate the refraction on the height grid and interpolate between the two nearest
        # heights
        heights_clipped = np.clip(target_height, self.heights[0], self.heights[-1])

        idx = np.clip(np.searchsorted(self.heights, heights_clipped) - 1, 0, len(self.heights) - 2)

        h_lo = self.heights[idx]
        h_hi = self.heights[idx + 1]

        weight = (heights_clipped - h_lo)/(h_hi - h_lo)

        refraction = np.zeros_like(elev_geometric)

        # Interpolate in the height dimension between the two bracketing tables
        for j in np.unique(idx):

            mask = (idx == j)

            r_lo = self.target_interps[j](elev_geometric[mask])
            r_hi = self.target_interps[j + 1](elev_geometric[mask])

            refraction[mask] = r_lo*(1.0 - weight[mask]) + r_hi*weight[mask]


        return elev_geometric + refraction



    def apparentElevationAstronomical(self, elev_geometric):
        """ Compute the apparent elevation of an astronomical source.

        Arguments:
            elev_geometric: [float or ndarray] Geometric elevation of the source (deg).

        Return:
            elev_apparent: [float or ndarray] Apparent elevation of the source (deg).

        """

        elev_geometric = np.asarray(elev_geometric, dtype=np.float64)

        return elev_geometric + self.astronomical_interp(elev_geometric)



def bennettRefraction(elev_apparent):
    """ Refraction of an astronomical source at sea level, using Bennett's approximation.

    Used as an independent check of the ray tracing.

    Arguments:
        elev_apparent: [float or ndarray] Apparent elevation (deg).

    Return:
        refraction: [float or ndarray] Refraction (deg).

    """

    elev_apparent = np.asarray(elev_apparent, dtype=np.float64)

    return (1.0/np.tan(np.radians(elev_apparent + 7.31/(elev_apparent + 4.4))))/60.0
