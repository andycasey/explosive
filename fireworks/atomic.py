#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Approximately model an atomic absorption line in a stellar spectrum. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import numpy as np
import os
import scipy.optimize as op
from astropy.table import Table

from oracle import photospheres, synthesis

logger = logging.getLogger("fireworks")

import model, utils

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




class MetaAtomicTransition(type):

    """
    Meta class to enforce read-only properties on the AtomicTransitionModel.
    """

    @property
    def element(cls):
        return cls.element

    @property
    def ion(cls):
        return cls.ion

    @property
    def wavelength(cls):
        return cls.wavelength

    @property
    def loggf(cls):
        return cls.loggf

    @property
    def excitation_potential(cls):
        return cls.excitation_potential

    @property
    def solar_abundance(cls):
        return cls.solar_abundance


class AtomicTransition(object):

    __metaclass__ = MetaAtomicTransition

    def __init__(self, data):
        """
        Initialise an AtomicTransition object.

        :param data:
            The relevant atomic data for this transition. This must include at
            least the wavelength, species, excitation potential, and the 
            oscillator strength (as loggf).

        :type data:
            dict or :class:`~astropy.table.Row`

        """

        # Require species or element/ion.
        try:
            species = data["species"]

        except (TypeError, KeyError):
            try:
                element, ion = data["element"], data["ion"]

            except (TypeError, KeyError):
                raise KeyError(
                    "must provide either species or (element and ion)")

        else:
            element, ion = utils.element(data["species"], full_output=True)

        _data = {
            "element": element,
            "ion": int(ion),
            "species": utils.species("{0} {1}".format(element, ion))
        }
        # Required keywords:
        for kwd in ("wavelength", "excitation_potential", "loggf"):
            try:
                _data[kwd] = float(data[kwd])
            except (TypeError, KeyError):
                raise KeyError("missing {0}".format(kwd))

        acceptable = {
            # Additional (optional) atomic data.
            "VDW_DAMP": 0,
            "D0": 0,
            "C4": 0,

            # Keywords related to the synthesis calculations.
            "metadata": None,
            "predictors": None,
            "predictor_labels": None,
            "predictor_ews": None,

            # Keywords related to the model fitting.
            "coefficients": None,
            "label_vector_description": None
        }
        for kwd, value in acceptable.items():
            try:
                _data[kwd] = data[kwd]
            except (TypeError, KeyError):
                _data[kwd] = value
        self._data = _data

        return None

    @property
    def element(self):
        return self._data["element"]

    @property
    def ion(self):
        return self._data["ion"]

    @property
    def wavelength(self):
        return self._data["wavelength"]

    @property
    def excitation_potential(self):
        return self._data["excitation_potential"]

    @property
    def loggf(self):
        return self._data["loggf"]

    @property
    def solar_abundance(self):
        return photospheres.solar_abundance(self.element)

    @property
    def _trained(self):
        """
        Private property to check whether this model has enough information to
        predict equivalent widths.
        """
        return None not in \
            (self._data["coefficients"], self._data["label_vector_description"])

    def __str__(self):
        return "<fireworks.atomic.{name} of {element} {ion} at {wavelength:.2f}"\
            " nm of {chi} eV at {location}>".format(
                name=self.__class__.__name__, location=hex(id(self)),
                wavelength=self.wavelength/10., chi=self.excitation_potential,
                element=self.element, ion=self.ion)

    def __repr__(self):
        return self.__str__()


    def save(self, filename, clobber=False):
        """
        Save the atomic transition to file.

        :param filename:
            The path to save the transition to.

        :type filename:
            str

        :param clobber: [optional]
            Overwrite the file if it already exists.

        :type clobber:
            bool
        """

        if os.path.exists(filename) and not clobber:
            raise IOError("filename exists and we were asked not to clobber: {}"\
                .format(filename))

        with open(filename, "wb") as fp:
            pickle.dump(self._data, fp, -1)
        return True


    @classmethod
    def load(cls, filename):
        """
        Load an atomic transition model from a filename.

        :param filename:
            The path to load the transition from.

        :type filename:
            str
        """

        with open(filename, "rb") as fp:
            data = pickle.load(fp)
        return cls(data)



    def line_abundances(self, stars, ews=None, mask_stars=None,
        stellar_parameter_labels=None, photosphere_kwargs=None,
        synthesis_kwargs=None, **kwargs):
        """
        Calculate atomic line abundances at different equivalent width values
        for some set of stellar parameters.

        :param stars:
            A table containing information for the stars to calculate abundances
            for (including stellar parameters).

        :type stars:
            :class:`~astropy.table.Table`

        :param ews: [optional]
            The equivalent width values to calculate line abundances for at all
            stellar parameters in the training set. If `None` is provided, line
            abundances will be calculated for 30 equivalent width values that
            are linearly spaced between 5 mA and 120 mA.

            The ews can be a single value, a list of values, or a callable
            function that accepts one argument, the star row in a table, and
            returns either a single value or a list of values.

        :type ews:
            float, iterable, or callable

        :param mask_stars: [optional]
            If provided this ignores stars where the `mask_stars` entry is True.
            It should have the same length as the stars provided.

        :type mask_stars:
            None or :class:`~np.array`

        :param stellar_parameter_labels: [optional]
            The label names of the (effective temperature, surface gravity,
            metallicity [, microturbulence]) in the `stars` table.

        :type stellar_parameter_labels:
            tuple of str

        :param photosphere_kwargs: [optional]
            Keywords that will be used when generating photospheres.

        :type photosphere_kwargs:
            dict

        :param synthesis_kwargs: [optional]
            Keywords that will by the synthesis code when calculating abundances.

        :type synthesis_kwargs:
            dict

        """
        return line_abundances([self], stars, ews=ews, mask_stars=mask_stars,
            stellar_parameter_labels=stellar_parameter_labels,
            photosphere_kwargs=photosphere_kwargs,
            synthesis_kwargs=synthesis_kwargs, **kwargs)



    def fit(self, label_vector_description=None, mask_predictors=None, **kwargs):

        # Create some additional labels:
        # REW


        # Do the fitting for each line.
        # We want some function that returns the approximate equivalent width for a
        # given set of logarithmic abundance and stellar parameters.
        def f(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
            return np.dot([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o], x)

        if mask_predictors is not None:
            raise NotImplementedError

        predictors = \
            Table(self._data["predictors"], names=self._data["predictor_labels"])

        # So I actually only want to use predictors where it's like
        # [X/Fe] ~ -0.5 to +0.5 (or maybe -1 to +1)
        x_h = predictors["LOG_EPSILON"] - self.solar_abundance
        x_fe = x_h \
            - predictors["PARAM_M_H"]

        rew = np.log(self._data["predictor_ews"]/self.wavelength)

        ok = x_fe > -100#(1 > x_fe) * (x_fe > -1) * (rew < -4.5)

        x_fe = x_fe[ok]
        x_h = x_h[ok]
        rew = rew[ok]
        predictors = predictors[ok]

        
        # Prepare the x and y data.
        
        y = x_h
        
        N = 15
        x = np.zeros((N, y.size))

        x[0] = 1.
        x[1] = rew
        x[2] = rew**2
        x[3] = rew**3
        x[4] = np.log(predictors["TEFF"])

        x[5] = np.log(predictors["TEFF"])**2
        x[6] = np.log(predictors["TEFF"])**3
        
        x[7] = predictors["TEFF"] * predictors["LOGG"]
        #x[8] = predictors["TEFF"] * predictors["PARAM_M_H"]
        #x[9] = predictors["LOGG"] * predictors["PARAM_M_H"]

        x[10] = predictors["PARAM_M_H"]
        x[11] = predictors["LOGG"]
        x[12] = predictors["XI"]
        x[13] = predictors["PARAM_M_H"]**2
        x[14] = (predictors["PARAM_M_H"] + 10)**3

        #x[11] = rew
        #x[12] = rew**2
        #x[13] = rew**3
        #x[14] = predictors["LOGG"]**2


        #x[10] = np.log(self._data["predictor_ews"][ok]/self.wavelength)
        #x[11] = np.log(self._data["predictor_ews"][ok]/self.wavelength)**2
        #x[12] = np.log(self._data["predictor_ews"][ok]/self.wavelength)**3
        #x[13] = 0
        

        #x[11] = np.log(self._data["predictor_ews"]/self.wavelength)
        #x[12] = np.log(self._data["predictor_ews"]/self.wavelength)**2
        #x[13] = np.log(self._data["predictor_ews"]/self.wavelength)

        """
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
        """


        p_opt, p_cov = op.curve_fit(f, x, y,
            p0=np.hstack([1, np.zeros(N - 1)]))

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2)

        ax = axes[0]

        y_synth, y_model = y, f(x, *p_opt)
        #y_synth = np.exp(y_synth) * self.wavelength
        #y_model = np.exp(y_model) * self.wavelength
        scat = ax.scatter(y_model, y_model - y_synth, c=predictors["XI"])
        
        """
        # show as abundance.
        expected_abund = predictors["LOG_EPSILON"]

        # actual_abund:
        actual_abund = []
        for i, ew_value in enumerate(y_model):

            def foobar(abund):

                x_ = x[:, i].copy()
                x_[10] = np.sqrt(abund[0])
                x_[11] = abund[0]
                x_[12] = abund[0]**2
                x_[13] = abund[0]**3

                return np.abs(np.exp(f(x_, *p_opt))*self.wavelength - ew_value)

            actual_abund.append(op.fmin(foobar, [5])[0])

            raise a


        actual_abund = np.array(actual_abund)

        ax = axes[1]
        ax.scatter(expected_abund, actual_abund)

        """
        #lims = ax.get_xlim()
        #ax.plot(lims, lims, zorder=-1)
        #ax.set_xlim(lims)
        #ax.set_ylim(lims)

        diff = y_model - y_synth
        print(np.mean(diff), np.std(diff))

        #diff2 = actual_abund - expected_abund
        #print(np.mean(diff2), np.std(diff2))
        plt.show()
        raise a




def line_abundances(atomic_transitions, stars, ews=None, mask_stars=None,
    stellar_parameter_labels=None, photosphere_kwargs=None,
    synthesis_kwargs=None, **kwargs):
    """
    Calculate atomic line abundances at different equivalent width values
    for some set of stellar parameters.

    :param atomic_transitions:
        The atomic transitions to calculate line abundances for.

    :type atomic_transitions:
        list of :class:`~fireworks.atomicAtomicTransitionModel`

    :param stars:
        A table containing information for the stars to calculate abundances for
        (including stellar parameters).

    :type stars:
        :class:`~astropy.table.Table`

    :param ews: [optional]
        The equivalent width values to calculate line abundances for at all
        stellar parameters in the training set. If `None` is provided, line
        abundances will be calculated for 30 equivalent width values that
        are linearly spaced between 5 mA and 120 mA.

        The ews can be a single value, a list of values, or a callable
        function that accepts one argument, the star row in a table, and
        returns either a single value or a list of values.

    :type ews:
        float, iterable, or callable

    :param mask_stars: [optional]
        If provided this ignores stars where the `mask_stars` entry is True.
        It should have the same length as the stars provided.

    :type mask_stars:
        None or :class:`~np.array`

    :param stellar_parameter_labels: [optional]
        The label names of the (effective temperature, surface gravity,
        metallicity [, microturbulence]) in the `stars` table.

    :type stellar_parameter_labels:
        tuple of str

    :param photosphere_kwargs: [optional]
        Keywords that will be used when generating photospheres.

    :type photosphere_kwargs:
        dict

    :param synthesis_kwargs: [optional]
        Keywords that will by the synthesis code when calculating abundances.

    :type synthesis_kwargs:
        dict

    """

    predictors = []
    equivalent_widths = []

    synthesis_kwargs = synthesis_kwargs or {}
    photosphere_kwargs = photosphere_kwargs or {}

    stellar_parameter_labels = stellar_parameter_labels or \
        ("TEFF", "LOGG", "PARAM_M_H")

    # Collate atomic data for all transitions.
    atomic_data = Table(rows=[_._data for _ in atomic_transitions])

    # Need to be calculated for each star.
    for i, star in enumerate(stars):
        print(i, len(stars))
        if mask_stars is not None and mask_stars[i]: continue

        # Build stellar parameters and generate the EW points.
        stellar_parameters = np.array([star[label] \
            for label in stellar_parameter_labels])
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
            raise ValueError("cannot calculate abundance for a negative EW")

        transitions_rtf = Table(np.repeat(atomic_data, len(ews_rtf)))
        transitions_rtf["equivalent_width"] = np.tile(ews_rtf, len(atomic_data))

        # Extend the predictors
        # f(teff, logg, feh, xi, log(abund)) --> ew
        # Note that these are predictors for all of the transitions, so we
        # will need to keep track of which predictors are associated with
        # each line.

        # (transition_index, teff, logg, feh, xi, log(X))
        predictor = np.repeat(np.hstack([0, stellar_parameters, xi, 0]),
            len(transitions_rtf)).reshape(-1, len(transitions_rtf)).T

        predictor[:,  0] = np.repeat(np.arange(len(atomic_data)),
            len(ews_rtf))
        predictor[:, -1] = synthesis.moog.atomic_abundances(
            transitions_rtf, stellar_parameters, xi,
            photosphere_kwargs=photosphere_kwargs, **synthesis_kwargs)

        predictors.extend(predictor)
        equivalent_widths.extend(np.tile(ews_rtf, len(atomic_data)))


    metadata = {
        "radiative_transfer_code": "MOOG",
        "synthesis_kwargs": synthesis_kwargs,
        "photosphere_kwargs": photosphere_kwargs
    }
    # Assign the predictors to each of the transitions
    predictors = np.hstack(predictors).reshape(-1, 6)
    equivalent_widths = np.hstack(equivalent_widths)
    label_names = list(stellar_parameter_labels) + ["XI", "LOG_EPSILON"]

    for i, atomic_transition in enumerate(atomic_transitions):
        match = (predictors[:, 0] == i)

        atomic_transition._data["metadata"] = metadata
        atomic_transition._data["predictors"] = predictors[match, 1:]
        atomic_transition._data["predictor_ews"] = equivalent_widths[match]
        atomic_transition._data["predictor_labels"] = label_names

    transition_indices = predictors[:, 0].astype(int)
    predictors = predictors[:, 1:]
    
    return (transition_indices, predictors, equivalent_widths, label_names)







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
    transitions = map(AtomicTransition, vstack([MgI_transitions, AlI_transitions, SiI_transitions]))


    DATA_PREFIX = "/Users/arc/research/luminosity-cannon/data/APOGEE-Hipparcos"
    stars = Table.read("{}.fits.gz".format(DATA_PREFIX))

    #model = AtomicTransitionModel(stars, atomic_data)
    #foo = line_abundances(transitions, stars, ews=np.linspace(5, 120, 10))

    t = AtomicTransition.load("t.pkl")
    t.fit()
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