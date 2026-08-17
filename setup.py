from __future__ import print_function, division, absolute_import, unicode_literals

from setuptools import setup, find_packages


setup(
    name='AircraftSolarTransitFinder',
    version='0.1.0',
    description="Find aircraft which crossed the apparent disk of the Sun, using ADS-B data",
    author="Denis Vida",
    license='MIT',
    packages=find_packages(include=['SolarTransit', 'SolarTransit.*']),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'pyarrow',
        'requests',
        'astropy',
        'jplephem',
        'matplotlib',
        'pyproj',
        ],
    python_requires='>=3.10',
    )
