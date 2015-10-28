#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Approximately model an atomic absorption line in a stellar spectrum. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np
import scipy.optimize as op
from astropy.table import Table

from oracle import photospheres, synthesis

logger = logging.getLogger("fireworks")

import utils

def _guess_abundance_label_format(species, star_labels):
    """
    Guess the string format of the abundance label columns, based on the atomic
    species we expect to be measureable and the existing labels for each star.

    :param species:
        A list of floating-point representations of transitions.

    :type species:
        iterable

    :param star_labels:
        The list of available string labels for each star.

    :type star_labels:
        list of str

    :returns:
        A two-length tuple containing the guessed string format of the abundance
        label, and what each element is measured relative to in the Sun (H, Fe,
        or None for logarithmic abundances).
    """

    try_these = [
        ("{ELEMENT}_H", "H"),
        ("{Element}_H", "H"),
        ("{ELEMENT}{ion}", None),
        ("{Element}{ion}", None),
        ("{ELEMENT}_FE", "Fe"),
        ("{Element}_Fe", "Fe"),
        ("LOG_{ELEMENT}", None),
        ("LOG_{ELEMENT}{ion}", None),
        ("logeps_{element}{ion}", None),
    ]
    if not len(species): return try_these[0]

    for element_repr, relative_to in try_these:
        for each in set(species):
            element, ion = utils.element(each, True)
            dict_repr = {
                "ELEMENT": element.upper(),
                "Element": element.title(),
                "element": element.lower(),
                "ion": ion
            }
            if element_repr.format(**dict_repr) not in star_labels:
                break
        else:
            break
    else:
        raise ValueError("Could not guess label abundance format. Tried: {0}"\
            .format(", ".join([_[0] for _ in try_these])))

    return (element_repr, relative_to)





class AtomicTransitionModel(object):


    def __init__(self, stars, atomic_data, abundance_label_format=None,
        stellar_parameter_labels=None):
        """
        Initialise the atomic transition model.
        """

        self.stars = stars
        self.atomic_data = atomic_data

        print("CHECK THE FORMAT OF STARS AND ATOMIC DATA")

        self.abundance_label_format = abundance_label_format \
            or _guess_abundance_label_format(
                atomic_data["species"], stars.dtype.names)

        self.stellar_parameter_labels = stellar_parameter_labels \
            or ("TEFF", "LOGG", "PARAM_M_H")



    def calculate_abundances(self, ews=None, mask_stars=None,
        photosphere_kwargs=None, synthesis_kwargs=None, **kwargs):
        """
        Calculate atomic line abundances at different equivalent width values
        for some set of stellar parameters.

        """

        predictors = []
        equivalent_widths = []

        synthesis_kwargs = synthesis_kwargs or {}
        photosphere_kwargs = photosphere_kwargs or {}

        # Need to be calculated for each star.
        for i, star in enumerate(self.stars):
            print(i, len(self.stars))
            if mask_stars is not None and mask_stars[i]: continue

            # Build stellar parameters and generate the EW points.
            stellar_parameters = np.array([star[label] \
                for label in self.stellar_parameter_labels])
            if stellar_parameters.size < 4: 
                # Microturbulence is missing. Use a relation.
                xi = utils.xi_relation(*stellar_parameters[:2])
            else:
                stellar_parameters, xi \
                    = stellar_parameters[:3], stellar_parameters[-1]
                
            # The equivalent widths might be linearly sampled (default), an
            # iterable, or a callable. These equivalent widths will be sampled
            # for all transitions.
            if ews is None:
                ews_rtf = np.linspace(5, 120, 30)
            elif hasattr(ews, '__call__'):
                ews_rtf = np.atleast_1d(ews(star))
            else:
                ews_rtf = np.atleast_1d(list(ews))

            if np.any(ews_rtf < 0):
                raise ValueError("cannot calculate abundance for a negative "\
                    "equivalent width")

            transitions_rtf = Table(np.repeat(self.atomic_data, len(ews_rtf)))
            transitions_rtf["equivalent_width"] \
                = np.tile(ews_rtf, len(self.atomic_data))

            # Extend the predictors
            # f(teff, logg, feh, xi, log(abund)) --> ew
            # Note that these are predictors for all of the transitions, so we
            # will need to keep track of which predictors are associated with
            # each line.

            # (transition_index, teff, logg, feh, xi, log(X))
            predictor = np.repeat(np.hstack([0, stellar_parameters, xi, 0]),
                len(transitions_rtf)).reshape(-1, len(transitions_rtf)).T

            predictor[:,  0] = np.repeat(np.arange(len(self.atomic_data)),
                len(ews_rtf))
            predictor[:, -1] = synthesis.moog.atomic_abundances(
                transitions_rtf, stellar_parameters, xi,
                photosphere_kwargs=photosphere_kwargs, **synthesis_kwargs)
        
            predictors.extend(predictor)
            equivalent_widths.extend(np.tile(ews_rtf, len(self.atomic_data)))

        predictors = np.hstack(predictors).reshape(-1, 6)
        equivalent_widths = np.hstack(equivalent_widths)

        names = ("transition_index", "teff", "logg", "feh", "xi", "log_eps")
        return (predictors, equivalent_widths, names)







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
    if ew_points is None: ew_points = np.linspace(10, 300, 15)
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














    def fit(self, ews=None, photosphere_kwargs=None, synthesis_kwargs=None,
        **kwargs):

        """
        Solve for the coefficients of the atomic line model for each transition.

        :param ews: [optional]
            The equivalent width values (in milliAngstroms) to calculate line
            abundances for. If `None` is given, then 30 values will be linearly
            sampled from 5 mA to 120 mA.

        :type ews:
            float, iterable, or callable

        :param photosphere_kwargs: [optional]
            Keywords that will be used when generating photospheres.

        :type photosphere_kwargs:
            dict

        :param synthesis_kwargs: [optional]
            Keywords that will by the synthesis code when calculating abundances.

        :type synthesis_kwargs:
            dict
        """







    def save(self):
        raise NotImplementedError




        raise NotImplementedError

"""
atomic_lines = fireworks.AtomicLineModel(stars[ok], line_list, 
    abundance_label_format="%(element)s_H",
    stellar_parameter_labels=None)
coefficients = atomic_model.fit(label_vector=None, ews=None, mask=None)

# save the coefficients or the model?
# need: coefficients, atomic info (wavlength, species)



"""

def approximate_atomic_transitions(stellar_parameters, transitions, X_H=True,
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
    if ew_points is None: ew_points = np.linspace(10, 300, 15)
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



if __name__ == "__main__":

    def air2vac(wave,sdssweb=False):
        """
        NAME:
           air2vac
        PURPOSE:
           Convert from air to vacuum wavelengths (See Allende Prieto technical note: http://hebe.as.utexas.edu/apogee/docs/air_vacuum.pdf)
        INPUT:
           wave - air wavelength in \AA
           sdssweb= (False) if True, use the expression from the SDSS website (http://classic.sdss.org/dr7/products/spectra/vacwavelength.html)
        OUTPUT:
           vacuum wavelength in \AA
        HISTORY:
           2014-12-04 - Written - Bovy (IAS)
           2015-04-27 - Updated to CAP note expression - Bovy (IAS)
        """
        return op.brentq(lambda x: vac2air(x,sdssweb=sdssweb)-wave,
                               wave-20,wave+20.)

    def vac2air(wave,sdssweb=False):
        """
        NAME:
           vac2air
        PURPOSE:
           Convert from vacuum to air wavelengths (See Allende Prieto technical note: http://hebe.as.utexas.edu/apogee/docs/air_vacuum.pdf)
        INPUT:
           wave - vacuum wavelength in \AA
           sdssweb= (False) if True, use the expression from the SDSS website (http://classic.sdss.org/dr7/products/spectra/vacwavelength.html)
        OUTPUT:
           air wavelength in \AA
        HISTORY:
           2014-12-04 - Written - Bovy (IAS)
           2015-04-27 - Updated to CAP note expression - Bovy (IAS)
        """
        if sdssweb:
            return wave/(1.+2.735182*10.**-4.+131.4182/wave**2.+2.76249*10.**8./wave**4.)
        else:
            return wave/(1.+0.05792105/(238.0185-(10000./wave)**2.)+0.00167917/(57.362-(10000./wave)**2.))

    # From Smith et al. (2013)
    MgI_transitions = Table(np.array([
        [15740.716, 12.0,   5.931, -0.262],
        [15748.9,   12.0,   5.932, +0.276],
        [15765.8,   12.0,   5.933, +0.504],
        [15879.5,   12.0,   5.946, -1.248],
        [15886.2,   12.0,   5.946, -1.555],
        [15889.485, 12.0,   5.946, -2.013],
        [15954.477, 12.0,   6.588, -0.807]
        ]), names=["wavelength", "species", "excitation_potential", "loggf"])
    MgI_transitions["wavelength"] = np.array(map(air2vac, MgI_transitions["wavelength"]))

    AlI_transitions = Table(np.array([
        [16718.957, 13.0,   4.085, +0.290],
        [16763.359, 13.0,   4.087, -0.524],
        ]), names=["wavelength", "species", "excitation_potential", "loggf"])
    AlI_transitions["wavelength"] = np.array(map(air2vac, AlI_transitions["wavelength"]))

    SiI_transitions = Table(np.array([
        [15361.161, 14.0,   5.954, -1.925],      
        [15833.602, 14.0,   6.222, -0.168],      
        [15960.063, 14.0,   5.984, +0.107],
        [16060.009, 14.0,   5.954, -0.566],
        [16094.787, 14.0,   5.964, -0.168],
        [16215.670, 14.0,   5.964, -0.665],
        [16680.770, 14.0,   5.984, -0.140],
        [16828.159, 14.0,   5.984, -1.102],
        ]), names=["wavelength", "species", "excitation_potential", "loggf"])
    SiI_transitions["wavelength"] = np.array(map(air2vac, SiI_transitions["wavelength"]))

    from astropy.table import vstack
    atomic_data = vstack([MgI_transitions, AlI_transitions, SiI_transitions])


    DATA_PREFIX = "/Users/arc/research/luminosity-cannon/data/APOGEE-Hipparcos"
    stars = Table.read("{}.fits.gz".format(DATA_PREFIX))

    model = AtomicTransitionModel(stars, atomic_data)

    raise a




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