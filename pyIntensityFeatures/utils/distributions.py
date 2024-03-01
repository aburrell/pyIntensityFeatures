#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
"""Functions providing data distribtuions."""

import numpy as np


def calc_quadratic(x, c, b, a):
    """Calculate the quadratic value at a specified location.

    Parameters
    ----------
    x : float or array-like
        Location(s) at which the quadratic background will be calculated
    c : float
        Constant term in the quadratic equation
    b : float
        Multipler for the x-term in the quadratic equation
    a : float
        Multiplier for the x-squared term in the quadratic equation

    Returns
    -------
    y : float or array-like
       Amplitude value(s) at the input location(s)

    """
    # Calculate the background
    y = c + b * x + a * x * x

    return y


def gauss(x, amp, mu, sigma, const):
    """Provide results for a simple Gaussian function.

    Parameters
    ----------
    x : (float or array-like)
        Dependent variable
    amp : float
        Amplitude
    mu : float
        Mean, or x-offset
    sigma : float
        Sigma, or exponential scalar
    const : float
        Additive constant

    Returns
    -------
    y : (float or array-like)
        Normal value at `x`

    """
    y = amp * np.exp(-np.power((x - mu), 2.0) / (2.0 * sigma**2)) + const

    return y


def mult_gauss(x, amps, mus, sigmas, const):
    """Provide results for a double-peaked Gaussian function.

    Parameters
    ----------
    x : float or array-like
        Dependent variable
    amps : array-like
        Amplitudes
    mus : array-like
        Means, or x-offsets
    sigmas : array-like
        Sigmas, or exponential scalars
    const : float
        Additive constant

    Returns
    -------
    y : (float or array-like)
        Normal value at `x`

    Raises
    ------
    ValueError
        If the number of input parameters is wrong

    """

    amps = np.asarray(amps)
    mus = np.asarray(mus)
    sigmas = np.asarray(sigmas)

    # Ensure the number of parameters is correct
    if amps.shape != mus.shape or sigmas.shape != amps.shape:
        raise ValueError('Unexpected number of input parameters')

    if np.asarray(x).shape == ():
        y = (amps * np.exp(-np.power((x - mus), 2.0) / (2.0 * sigmas**2))
             + const).sum()
    else:
        y = const
        for i, amp in enumerate(amps):
            y += gauss(x, amp, mus[i], sigmas[i], 0.0)

    return y


def single_gauss_quad(x, c, b, a, amp, mu, sigma):
    """Provide results for a single Gaussian with a quadratic background.

    Parameters
    ----------
    x : (float or array-like)
        Dependent variable
    c : float
        Constant term in the quadratic equation
    b : float
        Multipler for the x-term in the quadratic equation
    a : float
        Multiplier for the x-squared term in the quadratic equation
    amp : float
        Amplitude
    mu : float
        Mean, or x-offset
    sigma : float
        Sigma, or exponential scalar

    Returns
    -------
    y : (float or array-like)
        Normal value at `x`

    """
    # Calculate the value
    y = gauss(x, amp, mu, sigma, c) + calc_quadratic(x, 0.0, b, a)

    return y


def mult_gauss_quad(x, *param):
    """Provide results for a multiple Gaussian with a quadratic background.

    Parameters
    ----------
    x : float or array-like
        Dependent variable
    param : set
        Set containing: constant, quadratic multiplier for x, quadratic
        multiplier for x^2, and Gaussian amplitudes, x offsets, and exponential
        scalers for each Gaussian. The number of each Gaussian group must be
        the same; e.g., there must be two of each amplitude, x offset, and
        exponential scalers, but only one constant and quadratic multipliers.

    Returns
    -------
    y : float or array-like
        Normal value at `x`

    Raises
    ------
    ValueError
        If the correct number of inputs are not supplied to `param`

    """
    amps = list()
    x_offs = list()
    e_scals = list()

    if len(param) == 1:
        params = param[0]
    else:
        params = param

    const = params[0]
    x_const = params[1]
    x2_const = params[2]

    for i in range(int((len(params) - 3) / 3)):
        amps.append(params[3 + i * 3])
        x_offs.append(params[4 + i * 3])
        e_scals.append(params[5 + i * 3])

    amps = np.asarray(amps)
    x_offs = np.asarray(x_offs)
    e_scals = np.asarray(e_scals)

    # Test the input
    if len(params) != len(amps) * 3 + 3:
        raise ValueError('Unexpected number of input parameters')

    # Calculate the double Gaussian value
    y = mult_gauss(x, amps, x_offs, e_scals, const) + calc_quadratic(
        x, 0.0, x_const, x2_const)

    return y
