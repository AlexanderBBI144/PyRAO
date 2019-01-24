"""Useful functions for working with coordinates."""
import numpy as np
from scipy.constants import c


def local_sid(mjd, lon):
    """
    Calculate local sidereal time from Greenwich sidereal time.

    Parameters
    ----------
    mjd : float
        Julian date
    lon : float
        Longitude in radians

    Returns
    -------
    local_sidereal_time : float
        Local sidereal time in radians based on J2000 epoch

    """
    jd = mjd + 2400000.5 - 0.125  # Convert Puschino mjd to Greenwich jd
    T = (jd - 2451545.0) / 36525  # Convert jd to J2000 epoch
    sid = 280.46061837\
        + 360.98564736629 * (jd - 2451545.0)\
        + 0.000387933 * T ** 2\
        - T ** 3 / 38710000
    # Calculate local sid by adding longitude to GMST
    return np.radians(sid % 360 + np.degrees(lon))


def za(beams, freqs):
    """
    Calculate zenith angle.

    Parameters
    ----------
    beams : int or array_like
        Indices of beams to return zenith angles for
    freqs : int or array_like
        Indices of frequencies to return zenith angles for

    Returns
    -------
    za : numpy.ndarray
        Zenith angles in radians

    """
    beams = np.array(beams)
    freqs = np.array(freqs)
    nbeams_total = 129
    len_dipole = 384.0

    za = np.zeros((nbeams_total, len(freqs)))
    for i in range(1, nbeams_total):
        za[i] = za[i-1] + c / (len_dipole * freqs * 1e6 * np.cos(za[i-1]))
    return za[beams]


def dej(lat, za):
    r"""
    Calculate declination.

    Declination here is computed as (latitude - zenith angle),
    however it can also be computed using the following equation:

    $$
    \sin{\delta} = \sin{\psi} \sin{h} - \cos{\psi} \cos{h} \cos{A}
    $$

    where

    $\delta$ - declination, positive north, negative south
    $h$ - altitude, positive above the horizon, negative below horizon
    $A$ - azimuth, measured westward from the South,
          other sources often measure from the North
    $\psi$ - observer's latitude

    Parameters
    ----------
    lat : float or numpy.ndarray
        Latitude in radians
    za : float or numpy.ndarray
        Zenith angle in radians

    Returns
    -------
    dej : float or numpy.ndarray
        Declination in radians

    References
    ----------
    1. Jean Meeus, Astronomical Algorithms 2 edition
       Chapter 13, "Transformations of Coordinates"
    2. pavolgaj/AstroAlgorithms4Python

    """
    return lat - za


def ra(az, alt, lat, sid):
    r"""
    Calculate right ascension.

    Right ascension is computed using the following formula:

    $$
    \tan{H} = \frac{\sin{A}} {\cos{A} \sin{\psi} + \tan{h} \cos{\psi}}
    $$
    $$
    \alpha = \theta - H
    $$

    where

    $\alpha$ - right ascension, if obtained from formula it is in radians
    $h$ - altitude, positive above the horizon, negative below horizon
    $A$ - azimuth, measured westward from the South,
          other sources often measure from the North
    $\psi$ - observer's latitude
    $H$ - local hour angle
    $\theta$ - local sidereal time

    Parameters
    ----------
    az : numpy.ndarray
        Azumuth in radians
    alt : numpy.ndarray
        Altitude in radians
    lat : numpy.ndarray
        Latitude in radians
    sid : numpy.ndarray
        Local sidereal time in radians

    Returns
    -------
    ra : numpy.ndarray
        Right ascensions in J2000.0 epoch

    References
    ----------
    1. Jean Meeus, Astronomical Algorithms 2 edition
       Chapter 13, "Transformations of Coordinates"
    2. pavolgaj/AstroAlgorithms4Python

    """
    x = np.sin(az)
    y = np.cos(az) * np.sin(lat) + np.tan(alt) * np.cos(az)

    return sid[:, np.newaxis, np.newaxis] - np.arctan2(x, y)
