#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Approximately model an atomic absorption line in a stellar spectrum. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np
import scipy.optimize as op

from oracle import photospheres, synthesis

logger = logging.getLogger("fireworks")


def approximate_atomic_transitions(stellar_parameters, transitions, X_H=False,
    ew_points=None, photosphere_kwargs=None, synthesis_kwargs=None, **kwargs):
    """
    Return functions to approximate the equivalent width of atomic transitions
    given the stellar parameters and logarithmic abundance.

    :param stellar_parameters:
        A multi-row table containing the temperature, surface gravity,
        metallicity(, and microturbulence) for stars in the training set. If the
        microturbulence is not provided, it will be approximated with the Kirby
        et al. (2008) relation for giants (logg < 3.5) and from the Reddy et al. 
        (2003) relationship for dwarfs (logg >= 3.5).

    :type stellar_parameters:
        :class:`np.ndarray`

    :param transitions:
        A table containing the atomic data for each transition to build a model
        for. This table should include columns named: `wavelength`, `species`,
        `excitation_potential`, `log_gf`, and optionally: `VDW_DAMP`.

    :type transitions:
        :class:`~astropy.table.Table`

    :param X_H: [optional]
        Use abundances in X_H format. If set to `False`, then log(X) abundance
        formats are assumed.

    :type X_H:
        bool

    :param ew_points: [optional]
        The equivalent widths (EWs) to sample at each set of stellar parameters
        for each atomic line. If `None` is provided, then 30 EW values between
        (10, 300) milliAngstroms will be sampled for each star.

    :type ew_points:
        list

    :param photosphere_kwargs: [optional]
        Keywords that will be used when generating photospheres.

    :type photosphere_kwargs:
        dict

    :param synthesis_kwargs: [optional]
        Keywords that will be used when running MOOG to determine abundances.

    :type synthesis_kwargs:
        dict
    """

    # Calculate abundances for all lines in all stars.
    ews, abundances = _atomic_line_abundances(stellar_parameters, transitions,
        X_H=X_H, ew_points=ew_points, photosphere_kwargs=photosphere_kwargs,
        synthesis_kwargs=synthesis_kwargs, **kwargs)

    # f(teff, logg, EW) --> logX
    full_output = kwargs.pop("full_output", False)
    results = []
    for i, transition in enumerate(transitions):
        p_opt, p_cov = _approximate_radiative_transfer(transition,
            stellar_parameters, ews[:, i, :], abundances[:, i, :])
        results.append((p_opt, p_cov) if full_output else p_opt)

    return results


def xi_relation(effective_temperature, surface_gravity):
    """
    Estimate microtubulence from relations between effective temperature and
    surface gravity. For giants (logg < 3.5) the relationship employed is from
    Kirby et al. (2008, ) and for dwarfs (logg >= 3.5) the Reddy et al. (2003)
    relation is used.

    :param effective_temperature:
        The effective temperature of the star in Kelvin.

    :type effective_temperature:
        float

    :param surface_gravity:
        The surface gravity of the star.

    :type surface_gravity:
        float

    :returns:
        The estimated microturbulence (km/s) from the given stellar parameters.

    :rtype:
        float
    """

    try:
        _ = len(effective_temperature)

    except TypeError:
        if surface_gravity >= 3.5:
            xi = 1.28 + 3.3e-4 * (effective_temperature - 6000) \
                - 0.64 * (surface_gravity - 4.5)
        else:
            xi = 2.70 - 0.509 * surface_gravity
    else:
        xi = np.nan * np.ones(len(effective_temperature))
        dwarfs = surface_gravity >= 3.5
        xi[dwarfs] = 1.28 + 3.3e-4 * (effective_temperature[dwarfs] - 6000) \
            - 0.64 * (surface_gravity[dwarfs] - 4.5)
        xi[~dwarfs] = 2.70 - 0.509 * surface_gravity[~dwarfs]
    return xi


def _atomic_line_abundances(stellar_parameters, transitions, X_H=False,
    ew_points=None, photosphere_kwargs=None, synthesis_kwargs=None, **kwargs):
    """
    Calculate atomic line abundances at different equivalent width values for
    some set of stellar parameters.

    :param stellar_parameters:
        A multi-row table containing the temperature, surface gravity,
        metallicity(, and microturbulence) for stars in the training set. If the
        microturbulence is not provided, it will be approximated with the Kirby
        et al. (2008) relation for giants (logg < 3.5) and from the Reddy et al. 
        (2003) relationship for dwarfs (logg >= 3.5).

    :type stellar_parameters:
        :class:`np.ndarray`

    :param transitions:
        A table containing the atomic data for each transition to build a model
        for. This table should include columns named: `wavelength`, `species`,
        `excitation_potential`, `log_gf`, and optionally: `VDW_DAMP`.

    :type transitions:
        :class:`~astropy.table.Table`

    :param X_H: [optional]
        Use abundances in X_H format. If set to `False`, then log(X) abundance
        formats are assumed.

    :type X_H:
        bool

    :param ew_points: [optional]
        The equivalent widths (EWs) to sample at each set of stellar parameters
        for each atomic line. If `None` is provided, then 30 EW values between
        (1, 300) milliAngstroms will be sampled for each star.

    :type ew_points:
        list

    :param photosphere_kwargs: [optional]
        Keywords that will be used when generating photospheres.

    :type photosphere_kwargs:
        dict

    :param synthesis_kwargs: [optional]
        Keywords that will be used when running MOOG to determine abundances.

    :type synthesis_kwargs:
        dict

    :returns:
        A 3-length tuple containing the equivalent width at each sampled point
        (an array of shape N_stars, N_transitions, N_ews), the expected atomic
        offset in each point (on the scale of log(EW/lambda)) and the calculated
        abundances. All three arrays have the same shape.
    """
    stellar_parameters = np.atleast_2d(stellar_parameters)
    if stellar_parameters.shape[1] == 3:
        logger.warn("No microturbulence values provided. These will be "\
            "estimated from relations of Reddy et al. (2003) for dwarfs and "\
            "Kirby et al. (2008) for giants.")

        stellar_parameters = np.hstack([
            stellar_parameters,
            xi_relation(*stellar_parameters[:, :2].T).reshape(-1, 1)
        ])

    elif stellar_parameters.shape[1] != 4:
        raise ValueError("stellar parameters must be rows of (teff, logg, feh,"\
            " xi), where xi is optional")

    # Check that the stellar parameters are OK.
    _sp_ranges = kwargs.pop("__valid_stellar_parameter_ranges", [
        (3500, 8000),
        (0, 5.5),
        (-5, 1),
        (0, None)
    ])
    _sp_param = ("effective temperature", "surface gravity", "metallicity",
        "microturbulence")
    for i, (parameter, (lower, upper)) in enumerate(zip(_sp_param, _sp_ranges)):
        if (upper is not None and any(stellar_parameters[:, i] > upper)) \
        or (lower is not None and any(stellar_parameters[:, i] < lower)):
            raise ValueError("a {0} value (index {1}) is outside of the valid "\
                "range ({2}, {3})".format(parameter, i, lower, upper))

    # Generate values for parameters, where necessary.
    if ew_points is None: ew_points = np.linspace(10, 300, 30)
    synthesis_kwargs = synthesis_kwargs or {}
    photosphere_kwargs = photosphere_kwargs or {}

    # Make a copy of the transition table, because we will fuck it up.
    transitions = transitions.copy()
    N_ews, N_stars, N_transitions \
        = map(len, [ew_points, stellar_parameters, transitions])
    abundances = np.nan * np.ones((N_stars, N_transitions, N_ews))
    
    # Some announcements for debugging purposes:
    logger.debug("Building EW-log(X) model from {0} stars and {1} EW points "\
        "per star".format(N_stars, N_ews))
    logger.debug("Range of stellar parameters:\n\tTeff = [{0:.0f}, {1:.0f}]\n\t"
        "logg = [{2:.2f}, {3:.2f}]\n\t[M/H] = [{4:.2f}, {5:.2f}]".format(
            stellar_parameters[:, 0].min(), stellar_parameters[:, 0].max(),
            stellar_parameters[:, 1].min(), stellar_parameters[:, 1].max(),
            stellar_parameters[:, 2].min(), stellar_parameters[:, 2].max()))

    # Any scaling for solar abundances?
    if X_H:
        solar_abundances = photospheres.solar_abundance(transitions["species"])
    else:
        solar_abundances = 0.
    
    for i, sp in enumerate(stellar_parameters):
        logger.debug("At star {0}: {1}".format(i, sp))
        for j, ew in enumerate(ew_points):
            # Update transition table with equivalent_widths before passing to
            # the radiative transfer code.
            transitions["equivalent_width"] = ew
            abundances[i, :, j] = synthesis.moog.atomic_abundances(
                transitions, sp[:3], sp[3],
                photosphere_kwargs=photosphere_kwargs, **synthesis_kwargs) \
                - solar_abundances
                

    # Calculate array of corresponding equivalent widths for each abundance.
    ews = np.tile(ew_points, N_stars * N_transitions).reshape(N_stars,
        N_transitions, -1)
    
    return (ews, abundances)




def _abundance_predictors(ew, wavelength, stellar_parameters):
    teff, logg, fe_h = stellar_parameters[:3]

    return np.array([1,
        np.log(ew/wavelength),
        np.log(ew/wavelength)**2,
        np.log(ew/wavelength)**3,
        np.log(teff),
        np.log(teff)**2,
        np.log(teff)**3,
        np.log(teff) * logg,
        np.log(teff) * fe_h,
        logg * fe_h,
        fe_h,
        logg,
        xi_relation(teff, logg),
        0,
        0,
    ])


def _solve_equivalent_width(abundance, coeffs, wavelength, stellar_parameters,
    tol=1.48e-08, maxiter=500):

    f = lambda ew: (abundance - \
        np.dot(_abundance_predictors(ew, wavelength, stellar_parameters), coeffs))**2

    return op.brent(f, brack=[0, 300], tol=tol, maxiter=maxiter)

def _approximate_radiative_transfer(transition, stellar_parameters, ews,
    abundances):
    
    # Do the fitting for each line.
    # We want some function that returns the approximate equivalent width for a
    # given set of logarithmic abundance and stellar parameters.
    def f(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
        return np.dot([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o], x)

    N = 15
    S, E = abundances.shape

    _x_sp = lambda sps, N=None: np.repeat(sps, E).reshape(-1, E)

    # Prepare the x and y data.
    y = abundances.flatten()
    x = np.zeros((N, S, E))
    x[0] = 1.
    x[1] = np.log(ews/transition["wavelength"])
    x[2] = np.log(ews/transition["wavelength"])**2
    x[3] = np.log(ews/transition["wavelength"])**3
    x[4] = _x_sp(np.log(stellar_parameters[:, 0]))
    x[5] = _x_sp(np.log(stellar_parameters[:, 0]))**2
    x[6] = _x_sp(np.log(stellar_parameters[:, 0]))**3
    x[7] = _x_sp(np.log(stellar_parameters[:, 0]) * stellar_parameters[:, 1])
    x[8] = _x_sp(np.log(stellar_parameters[:, 0]) * stellar_parameters[:, 2])
    x[9] = _x_sp(stellar_parameters[:, 1] * stellar_parameters[:, 2])
    x[10] = _x_sp(stellar_parameters[:, 2])
    x[11] = _x_sp(stellar_parameters[:, 1])
    x[12] = _x_sp(xi_relation(*stellar_parameters[:, :2].T))

    x = x.reshape(x.shape[0], -1)
    ok = ((x[1] < -4.5) * (ews.flatten() > 1)).flatten()  
    x, y = x[:, ok], y[ok]

    p_opt, p_cov = op.curve_fit(f, x, y, p0=np.hstack([1, np.zeros(x.shape[0] - 1)]))

    if True:

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        model = f(x, *p_opt)
        scat = ax.scatter(y, model, c=np.log(ews/transition["wavelength"]).flatten()[ok])
        cbar = plt.colorbar(scat)
        limits = [
            min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ]
        ax.plot(limits, limits, c="#666666", zorder=-100)
        ax.set_xlim(limits)
        ax.set_ylim(limits)

        ax.set_title("{0:.3f}".format(np.abs(model - y).std()))

    return (p_opt, p_cov)

"""

def _tile_transition_values(values, N_stars, N_ews):
    return np.repeat(np.tile(values, N_stars).reshape(N_stars, -1), N_ews)\
        .reshape(N_stars, -1, N_ews)


# Calculate the offsets between different atomic lines (e.g., no radiative
# transfer is needed). Below chi is excitation_potential.
# atomic_offsets = -5040*chi/Teff + np.log(wavelength * np.exp(loggf))
chi = _tile_transition_values(transitions["excitation_potential"], N_stars,
    N_ews)
teff = _tile_stellar_parameters(stellar_parameters[:, 0], N_transitions,
    N_ews)

_ = np.log(transitions["wavelength"] * np.exp(transitions["loggf"]))    
atomic_offsets = -5040.* chi/teff + _tile_transition_values(_, N_stars, N_ews)
"""