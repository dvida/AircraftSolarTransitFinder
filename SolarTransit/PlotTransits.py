"""
Plot the candidates found by the transit search.
Includes:
    - Angular separation from the centre of the Sun as a function of time
    - The view of the eclipsed Sun with the silhouette of the aircraft drawn to scale
    - A map of the region with the tracks of the candidates

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

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import matplotlib.patheffects as path_effects

from SolarTransit.ADSBData import loadRegion
from SolarTransit.Config import NOMINAL_TIME, RESULTS_DIR, SUBSET_DATA_DIR, MIN_POINTS_PER_FLIGHT
from SolarTransit.Conversions import ecef2LatLonAlt, angularSeparation, slantRange
from SolarTransit.Ephemeris import eclipseCircumstances
from SolarTransit.Refraction import RefractionTable
from SolarTransit.TransitSearch import apparentTrack
from SolarTransit.Trajectory import buildTracks


### CONSTANTS ###

# Time step used when the tracks are evaluated for the plots (s)
PLOT_TIME_STEP = 0.05

# Half width of the plotted time interval around the closest approach (s)
PLOT_HALF_WINDOW = 90.0

# Colours of the Sun, of the Moon and of the corona
SUN_COLOR = '#ffd447'
MOON_COLOR = '#101014'
CORONA_COLOR = '#fff6d5'

# How the view of the Sun is oriented, given as the sign of the flip of each axis. An astronomical
# telescope without a diagonal turns the image upside down and left to right, while a diagonal
# mirrors it left to right only.
VIEW_MODES = {
    'naked_eye': (-1, -1),
    'flipped': (-1, 1),
    'mirrored': (1, -1),
    'inverted': (1, 1),
    }

VIEW_DESCRIPTIONS = {
    'naked_eye': "as seen with the unaided eye",
    'flipped': "flipped vertically, as in the eyepiece of a telescope",
    'mirrored': "mirrored left to right, as through a star diagonal",
    'inverted': "turned upside down and mirrored, as in an inverting telescope",
    }

# Orientation used by default
DEFAULT_VIEW = 'flipped'

### ###



def planeSilhouette(x_center, y_center, span, direction, aspect=0.95):
    """ Create a simple silhouette of an aircraft, drawn to the given angular size.

    The wings are drawn perpendicular to the direction of the apparent motion and the fuselage
    along it, which is how an aircraft in level flight is seen when it crosses the disk.

    Arguments:
        x_center: [float] Horizontal position of the centre of the aircraft on the plot.
        y_center: [float] Vertical position of the centre of the aircraft on the plot.
        span: [float] Wingspan in the units of the plot.
        direction: [float] Direction of the apparent motion (rad), measured from the horizontal
            axis of the plot.

    Keyword arguments:
        aspect: [float] Ratio of the length of the fuselage to the wingspan.

    Return:
        polygon: [ndarray] Vertices of the silhouette.

    """

    half_span = span/2
    half_length = aspect*span/2

    # One side of the silhouette, traced from the nose to the tail in a frame where the fuselage
    # points along the x axis. The other side is its mirror image.
    side = np.array([
        [ 1.00*half_length,  0.00*half_span],   # nose
        [ 0.75*half_length,  0.07*half_span],
        [ 0.10*half_length,  0.07*half_span],   # root of the leading edge of the wing
        [-0.25*half_length,  1.00*half_span],   # wing tip, leading edge
        [-0.42*half_length,  1.00*half_span],   # wing tip, trailing edge
        [-0.20*half_length,  0.07*half_span],   # root of the trailing edge of the wing
        [-0.70*half_length,  0.06*half_span],
        [-0.88*half_length,  0.34*half_span],   # tailplane tip
        [-1.00*half_length,  0.34*half_span],
        [-0.98*half_length,  0.04*half_span],   # tail
        ])

    # Close the outline by mirroring the traced side
    outline = np.vstack([side, side[::-1]*np.array([1.0, -1.0])])

    # Rotate the silhouette into the direction of the motion
    rot = np.array([[np.cos(direction), -np.sin(direction)],
                    [np.sin(direction),  np.cos(direction)]])

    outline = outline.dot(rot.T)

    outline[:, 0] += x_center
    outline[:, 1] += y_center

    return outline



def drawEclipsedSun(ax, sun_radius, moon_radius, moon_x, moon_y, obscuration, limit,
    corona=True):
    """ Draw the Sun as it looked during the eclipse.

    The Moon is drawn on top of the Sun at its true relative position, so what remains uncovered
    is the crescent which was actually seen. When the eclipse is deep, the corona and the glow of
    the sky around the Sun are added.

    Arguments:
        ax: [matplotlib axis] Axis on which the Sun is drawn.
        sun_radius: [float] Angular radius of the Sun (arcmin).
        moon_radius: [float] Angular radius of the Moon (arcmin).
        moon_x: [float] Horizontal offset of the centre of the Moon from the centre of the Sun
            (arcmin).
        moon_y: [float] Vertical offset of the centre of the Moon (arcmin).
        obscuration: [float] Fraction of the solar disk which is covered.
        limit: [float] Half width of the plotted frame (arcmin).

    Keyword arguments:
        corona: [bool] Draw the corona. True by default.

    """

    # The corona and the sky glow are only visible when almost all of the disk is covered
    if corona and (obscuration > 0.9):

        n_pix = 600

        grid = np.linspace(-limit, limit, n_pix)
        xx, yy = np.meshgrid(grid, grid)

        rr = np.sqrt(xx**2 + yy**2)

        # Brightness of the corona, which falls off steeply with the distance from the limb
        brightness = np.exp(-(rr - sun_radius)/(0.45*sun_radius))
        brightness[rr < sun_radius] = 1.0

        # The corona is only as bright as the fraction of the disk which is covered
        brightness = np.clip(brightness, 0, 1)*min((obscuration - 0.9)/0.1, 1.0)

        rgba = np.zeros((n_pix, n_pix, 4))
        rgba[..., 0] = 1.0
        rgba[..., 1] = 0.96
        rgba[..., 2] = 0.84
        rgba[..., 3] = 0.55*brightness

        ax.imshow(rgba, extent=(-limit, limit, -limit, limit), origin='lower', zorder=1,
            interpolation='bilinear')


    # The Sun, and the Moon drawn on top of it
    ax.add_patch(Circle((0, 0), sun_radius, color=SUN_COLOR, zorder=2))
    ax.add_patch(Circle((moon_x, moon_y), moon_radius, color=MOON_COLOR, zorder=3))

    # Outline of the solar limb, so that the covered part of the disk can still be seen
    ax.add_patch(Circle((0, 0), sun_radius, facecolor='none', edgecolor='#8a8a8a',
        linestyle=':', linewidth=0.8, zorder=4))



def drawPlane(ax, plane_x, plane_y, span_arcmin, i, alpha=1.0):
    """ Draw the silhouette of the aircraft at one point of its apparent track.

    Arguments:
        ax: [matplotlib axis] Axis on which the aircraft is drawn.
        plane_x: [ndarray] Horizontal offsets of the aircraft from the centre of the Sun (arcmin).
        plane_y: [ndarray] Vertical offsets of the aircraft (arcmin).
        span_arcmin: [ndarray] Angular wingspan of the aircraft (arcmin).
        i: [int] Index of the point which is drawn.

    Keyword arguments:
        alpha: [float] Opacity of the silhouette.

    """

    direction = trackDirection(plane_x, plane_y, i)

    outline = planeSilhouette(plane_x[i], plane_y[i], span_arcmin[i], direction)

    ax.add_patch(Polygon(outline, closed=True, facecolor='#0a0a0a', edgecolor='#5fc8ff',
        linewidth=0.5, alpha=alpha, zorder=6))



def trackDirection(plane_x, plane_y, i):
    """ Return the direction of the apparent motion at one point of the track.

    Arguments:
        plane_x: [ndarray] Horizontal offsets of the aircraft from the centre of the Sun (arcmin).
        plane_y: [ndarray] Vertical offsets of the aircraft (arcmin).
        i: [int] Index of the point.

    Return:
        direction: [float] Direction of the motion (rad), measured from the horizontal axis.

    """

    j = min(i + 1, len(plane_x) - 1)
    k = max(i - 1, 0)

    return np.arctan2(plane_y[j] - plane_y[k], plane_x[j] - plane_x[k])



def labelOffset(plane_x, plane_y, i, span, distance=2.2):
    """ Find where to put the time label of a mark, so that it does not sit on the track.

    The label is placed perpendicular to the track, on the side which is further away from the
    centre of the Sun, so that the labels do not fall onto the solar disk.

    Arguments:
        plane_x: [ndarray] Horizontal offsets of the aircraft from the centre of the Sun (arcmin).
        plane_y: [ndarray] Vertical offsets of the aircraft (arcmin).
        i: [int] Index of the point which is labelled.
        span: [float] Angular wingspan of the aircraft (arcmin).

    Keyword arguments:
        distance: [float] Distance of the label from the track, in units of the wingspan.

    Return:
        (x, y): [tuple of floats] Position of the label (arcmin).

    """

    direction = trackDirection(plane_x, plane_y, i)

    # Unit vector perpendicular to the track
    perp = np.array([-np.sin(direction), np.cos(direction)])

    # Point the offset away from the centre of the Sun
    if np.dot(perp, [plane_x[i], plane_y[i]]) < 0:
        perp = -perp

    offset = max(distance*span, 1.2)

    return plane_x[i] + offset*perp[0], plane_y[i] + offset*perp[1]



def drawDirectionArrow(ax, plane_x, plane_y, limit, flip_x=-1, flip_y=-1):
    """ Draw an arrow along the track which shows in which direction the aircraft flew.

    Arguments:
        ax: [matplotlib axis] Axis on which the arrow is drawn.
        plane_x: [ndarray] Horizontal offsets of the aircraft from the centre of the Sun (arcmin).
        plane_y: [ndarray] Vertical offsets of the aircraft (arcmin).
        limit: [float] Half width of the plotted frame (arcmin).

    Keyword arguments:
        flip_x: [int] Sign of the flip of the horizontal axis, as given by VIEW_MODES.
        flip_y: [int] Sign of the flip of the vertical axis.

    """

    inside = np.where((np.abs(plane_x) < 0.98*limit) & (np.abs(plane_y) < 0.98*limit))[0]

    if len(inside) < 2:
        return


    direction = trackDirection(plane_x, plane_y, inside[len(inside)//2])

    # The arrow is drawn near the top of the frame, parallel to the track but away from it, so
    # that it does not cover either the disk of the Sun or the time labels. The top of the frame
    # is at the negative end of the axis when the axis is flipped.
    length = 0.30*limit

    x_mid = 0.0
    y_mid = -0.86*flip_y*limit

    dx = 0.5*length*np.cos(direction)
    dy = 0.5*length*np.sin(direction)

    ax.annotate("", xy=(x_mid + dx, y_mid + dy), xytext=(x_mid - dx, y_mid - dy),
        arrowprops=dict(arrowstyle='-|>', color='#5fc8ff', linewidth=1.6,
        mutation_scale=18, shrinkA=0, shrinkB=0), zorder=8)

    # Direction of the track as it appears on the screen, which is not the direction in the data
    # when an axis is flipped
    rotation = np.degrees(np.arctan2(-flip_y*np.sin(direction), -flip_x*np.cos(direction)))

    # Keep the text upright, no matter in which direction the aircraft flew
    if rotation > 90:
        rotation -= 180

    elif rotation < -90:
        rotation += 180


    ax.text(x_mid, y_mid - 0.045*flip_y*limit, "direction of flight", color='#5fc8ff', fontsize=8,
        ha='center', va='bottom', rotation=rotation, rotation_mode='anchor', zorder=8)



def candidateSeries(track, time_ref, t_center, site_ecef, site_elevation_msl, refraction_table,
    location, half_window=PLOT_HALF_WINDOW, time_step=PLOT_TIME_STEP):
    """ Compute the apparent position of an aircraft and of the Sun around the closest approach.

    Arguments:
        track: [FlightTrack] Track of the flight.
        time_ref: [datetime] Reference time of the track.
        t_center: [float] Time of the closest approach, in seconds relative to the reference time.
        site_ecef: [tuple] ECEF coordinates of the observer (m).
        site_elevation_msl: [float] Height of the observer above sea level (m).
        refraction_table: [RefractionTable] Table used to apply the refraction.
        location: [EarthLocation] Location of the observing site.

    Keyword arguments:
        half_window: [float] Half width of the computed interval around the closest approach (s).
        time_step: [float] Time step (s).

    Return:
        series: [dict] Times, apparent positions of the aircraft, positions and radii of the Sun
            and of the Moon, the separation and the distance to the aircraft.

    """

    t_rel = np.arange(t_center - half_window, t_center + half_window + time_step, time_step)

    t_rel = t_rel[(t_rel >= track.t_beg) & (t_rel <= track.t_end)]

    t_rel, azim, alt, ecef = apparentTrack(track, t_rel, site_ecef, site_elevation_msl,
        refraction_table)

    times = np.array([time_ref + datetime.timedelta(seconds=float(t)) for t in t_rel])

    circ = eclipseCircumstances(times, location)

    sun_alt_apparent = refraction_table.apparentElevationAstronomical(circ['sun_alt'])
    moon_alt_apparent = refraction_table.apparentElevationAstronomical(circ['moon_alt'])

    sep = angularSeparation(azim, alt, circ['sun_azim'], sun_alt_apparent)

    dist = slantRange(site_ecef, ecef)

    return {
        'times': times,
        't_rel': t_rel,
        'azim': azim,
        'alt': alt,
        'sun_azim': circ['sun_azim'],
        'sun_alt': sun_alt_apparent,
        'sun_radius': circ['sun_radius'],
        'moon_azim': circ['moon_azim'],
        'moon_alt': moon_alt_apparent,
        'moon_radius': circ['moon_radius'],
        'obscuration': circ['obscuration'],
        'separation': sep,
        'distance': dist,
        }



def plotSeparation(series_list, labels, output_path, site):
    """ Plot the angular separation from the centre of the Sun as a function of time.

    Arguments:
        series_list: [list of dict] Series computed by candidateSeries.
        labels: [list of str] Labels of the candidates.
        output_path: [str] Path of the image which will be written.
        site: [ObservingSite] The observing site.

    """

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for series, label in zip(series_list, labels):

        # Time in seconds relative to the closest approach
        i_min = int(np.argmin(series['separation']))
        t_zero = series['t_rel'][i_min]

        ax.plot(series['t_rel'] - t_zero, 60*series['separation'], label="{:s} ({:s} UTC)".format(
            label, series['times'][i_min].strftime("%H:%M:%S")))


    # The radius of the solar disk, which is what has to be crossed for a transit
    sun_radius_arcmin = 60*np.mean([s['sun_radius'].mean() for s in series_list])

    ax.axhline(sun_radius_arcmin, color='k', linestyle='--', linewidth=1,
        label="Solar limb ({:.1f}')".format(sun_radius_arcmin))

    ax.set_xlabel("Time from the closest approach (s)")
    ax.set_ylabel("Angular separation from the centre of the Sun (arcmin)")
    ax.set_title("Aircraft passing the apparent position of the Sun\n"
        "{:s}, seen from {:.4f} N, {:.4f} E".format(
        series_list[0]['times'][0].strftime("%Y %B %d"), site.lat, site.lon))

    ax.set_xlim(-60, 60)
    ax.set_ylim(0, 60)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Plot written to: {:s}".format(output_path))



def plotEclipseView(series, label, output_path, n_silhouettes=7, view=DEFAULT_VIEW):
    """ Plot the view of the eclipsed Sun with the aircraft crossing it.

    The Sun and the Moon are drawn at their true relative positions at the moment of the closest
    approach, so the shape of the remaining crescent is the one which was actually seen. The
    aircraft is drawn to scale, using the wingspan of its type and its distance from the observer.

    Arguments:
        series: [dict] Series computed by candidateSeries.
        label: [str] Label of the candidate.
        output_path: [str] Path of the image which will be written.

    Keyword arguments:
        n_silhouettes: [int] Number of silhouettes drawn along the track.
        view: [str] Orientation of the frame, one of the keys of VIEW_MODES. The image is only
            reoriented, the underlying coordinates are not changed.

    """

    i_min = int(np.argmin(series['separation']))

    # The frame is centred on the Sun, with the zenith direction pointing up. The horizontal axis
    # is the azimuth difference compressed by the cosine of the altitude, so that the frame is
    # not distorted.
    cos_alt = np.cos(np.radians(series['sun_alt'][i_min]))

    def toFrame(azim, alt, i):
        """ Convert the horizontal coordinates into the frame centred on the Sun (arcmin). """

        daz = (np.asarray(azim) - series['sun_azim'][i] + 180)%360 - 180

        return 60*daz*cos_alt, 60*(np.asarray(alt) - series['sun_alt'][i])


    sun_radius = 60*series['sun_radius'][i_min]
    moon_radius = 60*series['moon_radius'][i_min]

    moon_x, moon_y = toFrame(series['moon_azim'][i_min], series['moon_alt'][i_min], i_min)

    plane_x, plane_y = toFrame(series['azim'], series['alt'], i_min)

    obscuration = series['obscuration'][i_min]


    # Angular size of the aircraft, drawn to scale
    span_arcmin = 60*np.degrees(series['wingspan']/series['distance'])

    limit = 2.2*sun_radius

    flip_x, flip_y = VIEW_MODES[view]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_facecolor('#05050a')

    drawEclipsedSun(ax, sun_radius, moon_radius, moon_x, moon_y, obscuration, limit)

    # Track of the aircraft
    ax.plot(plane_x, plane_y, color='#5fc8ff', linewidth=0.8, alpha=0.8, zorder=5)

    # Draw the silhouette at several moments while the aircraft is inside the frame
    inside = np.where((np.abs(plane_x) < 0.90*limit) & (np.abs(plane_y) < 0.90*limit))[0]

    if len(inside):

        idx_draw = list(inside[np.linspace(0, len(inside) - 1, n_silhouettes).astype(int)])

        # The moment of the closest approach is always drawn. It replaces the nearest of the
        # evenly spaced marks, so that the two labels do not end up on top of each other.
        idx_draw[int(np.argmin(np.abs(np.array(idx_draw) - i_min)))] = i_min

    else:
        idx_draw = [i_min]


    for i in idx_draw:

        drawPlane(ax, plane_x, plane_y, span_arcmin, i, alpha=1.0 if i == i_min else 0.55)

        # The aircraft crosses the frame in about ten seconds, so the marks are only a few
        # seconds apart and the labels need a tenth of a second. The full hour is in the title.
        ax.annotate(series['times'][i].strftime("%M:%S.%f")[:-5],
            xy=(plane_x[i], plane_y[i]), xytext=labelOffset(plane_x, plane_y, i, span_arcmin[i]),
            textcoords='data', color='#5fc8ff', fontsize=7,
            ha='center', va='center', zorder=8,
            fontweight='bold' if i == i_min else 'normal',
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground='#05050a')])


    # Mark the position at the closest approach
    ax.plot(plane_x[i_min], plane_y[i_min], marker='+', color='#5fc8ff', markersize=8, zorder=7)

    # Arrow which shows in which direction the aircraft flew
    drawDirectionArrow(ax, plane_x, plane_y, limit, flip_x=flip_x, flip_y=flip_y)

    # Orient the frame the way the view is to be shown
    ax.set_xlim(flip_x*limit, -flip_x*limit)
    ax.set_ylim(flip_y*limit, -flip_y*limit)
    ax.set_aspect('equal')

    ax.set_xlabel("Offset from the centre of the Sun, along the azimuth (arcmin){:s}".format(
        ", increasing to the left" if flip_x > 0 else ""))
    ax.set_ylabel("Offset from the centre of the Sun, along the altitude (arcmin){:s}".format(
        ", increasing downwards" if flip_y > 0 else ""))

    ax.set_title("{:s} and the eclipsed Sun, {:s} UTC\n"
        "solar disk {:.1f}% covered, aircraft {:.0f} km away, {:.0f} arcsec across, "
        "{:.1f} arcmin from the centre\n"
        "{:s}, the marks along the track are labelled mm:ss.s UTC".format(label,
        series['times'][i_min].strftime("%H:%M:%S.%f")[:-4], 100*obscuration,
        series['distance'][i_min]/1000, 60*span_arcmin[i_min], 60*series['separation'][i_min],
        VIEW_DESCRIPTIONS[view]), fontsize=10)

    ax.tick_params(colors='#404040')
    ax.grid(alpha=0.12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor='white')
    plt.close(fig)

    print("Plot written to: {:s}".format(output_path))



def plotMap(series_list, labels, output_path, site):
    """ Plot a map of the region with the tracks of the candidates and the direction of the Sun.

    Arguments:
        series_list: [list of dict] Series computed by candidateSeries, with the ground tracks
            stored under the keys lat and lon.
        labels: [list of str] Labels of the candidates.
        output_path: [str] Path of the image which will be written.
        site: [ObservingSite] The observing site.

    """

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(site.lon, site.lat, marker='*', color='red', markersize=14, zorder=5,
        label="Observing site")

    for series, label in zip(series_list, labels):

        i_min = int(np.argmin(series['separation']))

        ax.plot(series['lon'], series['lat'], linewidth=1.2, label="{:s} ({:s} UTC)".format(label,
            series['times'][i_min].strftime("%H:%M:%S")))

        ax.plot(series['lon'][i_min], series['lat'][i_min], marker='o', markersize=5,
            color=ax.lines[-1].get_color())

        # Line from the site towards the aircraft at the moment of the closest approach
        ax.plot([site.lon, series['lon'][i_min]], [site.lat, series['lat'][i_min]],
            color='gray', linestyle=':', linewidth=0.8, zorder=1)


    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title("Ground tracks of the candidates and the lines of sight towards the Sun")

    # Keep the aspect ratio correct at this latitude
    ax.set_aspect(1.0/np.cos(np.radians(site.lat)))

    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print("Plot written to: {:s}".format(output_path))



def plotCandidates(df_cand, df_adsb, site, time_ref, view=DEFAULT_VIEW, results_dir=RESULTS_DIR,
    refraction_table=None):
    """ Plot a set of candidates, given the table of candidates and the ADS-B reports.

    This is the entry point which the pipeline uses, and which the command line interface of this
    module wraps.

    Arguments:
        df_cand: [pandas.DataFrame] Candidates, as written by the transit search.
        df_adsb: [pandas.DataFrame] ADS-B reports which contain the tracks of the candidates.
        site: [ObservingSite] The observing site.
        time_ref: [datetime] Reference time of the tracks.

    Keyword arguments:
        view: [str] Orientation of the view of the Sun, one of the keys of VIEW_MODES.
        results_dir: [str] Directory into which the plots are written.
        refraction_table: [RefractionTable] Table used to apply the refraction. It is computed if
            it is not given.

    Return:
        plot_paths: [list of str] Paths of the plots which were written.

    """

    if not len(df_cand):
        print("No candidates to plot")
        return []


    elevation = site.elevation
    location = site.astropyLocation()
    site_ecef = site.ecef()

    if refraction_table is None:
        print("Building the refraction table...")
        refraction_table = RefractionTable(elevation)


    tracks = {track.flight_id: track for track in buildTracks(df_adsb, time_ref,
        min_points=MIN_POINTS_PER_FLIGHT)}

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)


    series_list = []
    labels = []
    plot_paths = []

    for _, cand in df_cand.iterrows():

        track = tracks.get(cand['flight_id'])

        if track is None:
            print("The track of {:s} was not found in the data".format(str(cand['callsign'])))
            continue


        t_center = (pd.Timestamp(cand['time_min_sep']).to_pydatetime() - time_ref).total_seconds()

        series = candidateSeries(track, time_ref, t_center, site_ecef, elevation,
            refraction_table, location)

        series['wingspan'] = track.wingspan

        # Ground track of the aircraft, needed for the map
        x, y, z = track.positionsECEF(series['t_rel'])

        lat, lon, _ = ecef2LatLonAlt(x, y, z)

        series['lat'] = np.degrees(lat)
        series['lon'] = np.degrees(lon)

        label = "{:s} ({:s}, {:s})".format(str(cand['callsign']), str(cand['tail_number']),
            str(cand['aircraft_type_icao']))

        series_list.append(series)
        labels.append(label)

        # The view of the Sun is plotted separately for every candidate
        view_path = os.path.join(results_dir, "eclipse_view_{:s}.png".format(
            str(cand['callsign'])))

        plotEclipseView(series, label, view_path, view=view)

        plot_paths.append(view_path)


    if not series_list:
        return plot_paths


    separation_path = os.path.join(results_dir, 'separation.png')
    map_path = os.path.join(results_dir, 'map.png')

    plotSeparation(series_list, labels, separation_path, site)
    plotMap(series_list, labels, map_path, site)

    return plot_paths + [separation_path, map_path]



def main():
    """ Plot the candidates found by an earlier run of the transit search. """

    import argparse

    from SolarTransit.Site import ObservingSite


    ### COMMAND LINE ARGUMENTS ###

    arg_parser = argparse.ArgumentParser(description="""Plot the candidates found by the transit
        search. """, formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('callsigns', metavar='CALLSIGN', type=str, nargs='*',
        help="Callsigns which will be plotted. If none are given, the best candidates from the "
             "result table are used.")

    arg_parser.add_argument('-c', '--candidates', metavar='CSV_PATH', type=str,
        default=os.path.join(RESULTS_DIR, 'candidates.csv'),
        help="Path of the table of candidates written by the transit search.")

    arg_parser.add_argument('-i', '--input', metavar='PARQUET_PATH', type=str,
        default=os.path.join(SUBSET_DATA_DIR, 'adsb_region.parquet'),
        help="Path of the regional ADS-B subset.")

    arg_parser.add_argument('-n', '--nplot', metavar='COUNT', type=int, default=3,
        help="Number of the best candidates which are plotted.")

    arg_parser.add_argument('-v', '--view', metavar='ORIENTATION', type=str, default=DEFAULT_VIEW,
        choices=sorted(VIEW_MODES.keys()),
        help="Orientation of the view of the Sun:\n"
             "    naked_eye - as seen with the unaided eye\n"
             "    flipped   - flipped vertically, as in the eyepiece of a telescope (default)\n"
             "    mirrored  - mirrored left to right, as through a star diagonal\n"
             "    inverted  - upside down and mirrored, as in an inverting telescope")

    cml_args = arg_parser.parse_args()

    #########################


    df_cand = pd.read_csv(cml_args.candidates, parse_dates=['time_min_sep'])

    # Select the candidates which will be plotted
    if cml_args.callsigns:
        df_cand = df_cand[df_cand['callsign'].isin(cml_args.callsigns)]

    else:
        df_cand = df_cand.head(cml_args.nplot)


    if not len(df_cand):
        print("No candidates to plot")
        sys.exit(1)


    site = ObservingSite.fromConfig()

    # Load the ADS-B reports which cover the tracks of the selected candidates
    t_beg = df_cand['time_min_sep'].min().to_pydatetime() - datetime.timedelta(minutes=10)
    t_end = df_cand['time_min_sep'].max().to_pydatetime() + datetime.timedelta(minutes=10)

    lat_min, lat_max, lon_min, lon_max = site.box()

    df_adsb = loadRegion(t_beg, t_end, lat_min=lat_min, lat_max=lat_max, lon_min=lon_min,
        lon_max=lon_max, cache_path=cml_args.input)

    plotCandidates(df_cand, df_adsb, site, NOMINAL_TIME, view=cml_args.view)



if __name__ == "__main__":

    main()
