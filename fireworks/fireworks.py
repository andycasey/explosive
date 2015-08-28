#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon with Chemistry """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np
from collections import OrderedDict

import scipy.optimize as op
from . import (atomic, cannon, model, plot, utils)

logger = logging.getLogger("fireworks")


class FireworksModel(cannon.CannonModel):

    _trained_attributes = ("_coefficients", "_scatter", "_offsets", "_ew_model",
        "_weak_line_mask", "_label_vector_description", "_atomic_lines")
    _data_attributes = ("_labels", "_wavelengths", "_fluxes",
        "_flux_uncertainties")

    def __init__(self, labels, wavelengths, fluxes, flux_uncertainties,
        verify=True):
        """
        Initialise a Cannon-with-Chemistry model.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            :class:`~astropy.table.Table`

        :param wavelengths:
            The wavelengths of the given pixels.

        :type wavelengths:
            :class:`np.array`

        :param fluxes:
            An array of fluxes for each star as shape (num_stars, num_pixels).
            The num_stars should match the rows in `labels`.

        :type fluxes:
            :class:`np.ndarray`

        :param flux_uncertainties:
            An array of 1-sigma flux uncertainties for each star as shape
            (num_stars, num_pixels). The shape of the `flux_uncertainties` array
            should match the `fluxes` array. 

        :type flux_uncertainties:
            :class:`np.ndarray`
        """

        super(self.__class__, self).__init__(labels, wavelengths, fluxes,
            flux_uncertainties, verify)


    @property
    @model.requires_training_wheels
    def labels(self):
        """
        Return a list of the labels involved in this model. This includes any
        labels in the description of the label vector, as well as any individual
        abundance labels.
        """

        _, labels = self._get_linear_indices(self._label_vector_description,
            full_output=True)
        if self._atomic_lines is not None:
            labels += self._atomic_lines.keys()
        return labels


    def train(self, label_vector_description, atomic_lines=None, X_H=False,
        N=None, limits=None, pivot=False, **kwargs):
        """
        Train a Cannon model based on the label vector description provided.

        :params label_vector_description:
            The human-readable form of the label vector description.

        :type label_vector_description:
            str or list of str

        :params atomic_lines: [optional]
            Atomic absorption lines that should be modelled as part of the first
            entry in the label vector. If given, this should be a dictionary
            containing the label names that correspond to $\log_\epsilon(X)$
            abundances in `CannonModel._labels` as keys, and the values should
            be an `~astropy.table.Table` containing the atomic transitions for
            that element.

        :type atomic_lines:
            dict

        :param X_H: [optional]
            Use abundances in X_H format. If set to `False`, then log(X)
            abundance formats are assumed.

        :type X_H:
            bool

        :param N: [optional]
            Limit the number of stars used in the training set. If left to None,
            all stars will be used.

        :type N:
            None or int

        :param limits: [optional]
            A dictionary containing labels (keys) and upper/lower limits (as a
            two-length tuple).

        :type limits:
            dict

        :param pivot: [optional]
            Pivot the data about the labels.

        :type pivot:
            bool

        :returns:
            A three-length tuple containing the model coefficients, the scatter
            in each pixel, and the label offsets.
        """

        # We'll need this later; store it. First sort by the species number.
        self._atomic_lines = OrderedDict(sorted(atomic_lines.items(), 
            key=lambda _: min(_[1]["species"])))
        self._label_vector_description = label_vector_description

        # Build the label vector array. Note that if `atomic_lines` are given,
        # the first element of the label vector (which would be 1 otherwise)
        # will need to be updated on a per-star basis.

        # Since building the atomic line models takes longer than building the
        # label vector array, we build the vector array first so that any errors
        # will appear first.
        lv = self._parse_label_vector_description(label_vector_description)
        lva, use, offsets = cannon._build_label_vector_array(self._labels, lv,
            N, limits, pivot)

        # Initialise the requisite arrays.
        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[0]))
        weak_line_fluxes = np.ones((N_stars, N_pixels))
        
        # Any atomic lines to model?
        if _check_atomic_lines(self._labels, self._atomic_lines):

            N_species = len(self._atomic_lines)
            N_transitions = sum(map(len, self._atomic_lines.values()))

            msg = []
            for k, v in self._atomic_lines.items():
                msg.append("{0} (species {1}; {2} lines)".format(
                    k, ", ".join(map(str, set(v["species"]))), len(v)))

            logger.info("Including {0} weak lines from {1} elements: {2}".format(
                N_transitions, N_species, ", ".join(msg)))

            # Build the log(X)->EW models (or vice-versa, sigh)

            # Estimate the FWHM kernel for each star, or estimate from all stars
            # (We need the FWHM to link the EW to an actual flux value.)
            # [TODO]

            p_sigma = 0.35

            # We should calculate the expected EWs (and therefore fluxes) for
            # each atomic line for each star in the training set, because this
            # will form the first element of our label vector array.

            teff_label = kwargs.pop("__label_teff", "TEFF")
            logg_label = kwargs.pop("__label_logg", "LOGG")
            fe_h_label = kwargs.pop("__label_fe_h", "FE_H")
            all_stellar_parameters = np.vstack([
                self._labels[teff_label],
                self._labels[logg_label],
                self._labels[fe_h_label]
            ]).T

            # [TODO] This part is unnecessarily slow. Speed it up.
            # [TODO] It's also probably catagorically wrong.
            ew_model = {}
            for label, transitions in self._atomic_lines.items():
                ew_coefficients = atomic.approximate_atomic_transitions(
                    all_stellar_parameters, transitions, X_H=X_H, **kwargs)
                ew_model[label] = (np.array(transitions["wavelength"]), ew_coefficients)

            # Generate the weak line fluxes for each star.
            for i, stellar_parameters in enumerate(all_stellar_parameters):
                for label, (wavelengths, ew_coefficients) in ew_model.items():

                    abundance = self._labels[label][i]
                    for j, mu in enumerate(wavelengths):
                        # The 10e-4 factor is to turn the EW from milliAngstroms
                        # into Angstroms.
                        expected_ew = atomic._solve_equivalent_width(abundance,
                            ew_coefficients[j], mu, stellar_parameters) * 10e-4

                        # Translate this into a weak profile.
                        # EW = sqrt(2*pi) * amplitude * sigma
                        # we know the central wavelength, we know the sigma
                        # (EW is in mA, and we want A)
                        amplitude = expected_ew/(np.sqrt(2*np.pi) * p_sigma)
                        weak_line_fluxes[i] *= 1. \
                            - amplitude * np.exp(-(self._wavelengths - mu)**2 \
                                / (2. * p_sigma**2))
                    

            import matplotlib.pyplot as plt
            plt.close("all")

            fig, ax = plt.subplots()
            ax.plot(self._wavelengths, weak_line_fluxes[0], c='b', zorder=100)
            ax.plot(self._wavelengths, self._fluxes[0], c='k')

            ax.plot(self._wavelengths, self._fluxes[0] / weak_line_fluxes[0], c='r', zorder=1000)
            #raise a

        else:
            ew_model = None


        assert N is None, "whoops?"
        pb_size = 100 if kwargs.pop("__progressbar", True) else 0
        pb_message = "Training {0} model from {1} stars with {2} pixels:\n"\
            .format(self.__class__.__name__[:-5], N_stars, N_pixels)
        for i in utils.progressbar(range(N_pixels), pb_message, pb_size):

            # Update the first element of the label vector array to include the
            # predicted EWs.
            fluxes = self._fluxes[use, i] / weak_line_fluxes[use, i]

            # Train the Cannon on the residuals of the data.
            # I *think* this is OK to do
            # (e.g., Hogg may be wrong??? -- famous last words?)
            coefficients[i, :], scatter[i] = cannon._fit_pixel(fluxes,
                self._flux_uncertainties[use, i], lva, **kwargs)

            if not np.any(np.isfinite(scatter[i] * coefficients[i, :])):
                logger.warn("No finite coefficients at pixel {}!".format(i))

        # Save all of these to the model.
        self._trained = True
        self._coefficients, self._scatter = coefficients, scatter
        self._offsets, self._ew_model = offsets, ew_model
        # Keep a record of pixels that are not affected by weak line fluxes at
        # all. These can be used for the initial guess of stellar parameters.
        self._weak_line_mask = np.all(weak_line_fluxes == 1, axis=0)

        return (coefficients, scatter, offsets, ew_model, weak_line_fluxes)


    @model.requires_training_wheels
    def predict(self, labels=None, **labels_as_kwargs):
        """
        Predict spectra from the trained model, given the labels.

        :param labels:
            The labels required for the trained model. This should be a N-length
            list matching the number of unique terms in the model, in the order
            given by the `self._get_linear_indices` function. Alternatively,
            labels can be explicitly given as keyword arguments.

        :type labels:
            list

        :returns:
            Model spectra for the given labels.

        :raises TypeError:
            If the model is not trained.
        """
        raise NotImplementedError
        
        try:
            labels[0]
        except (TypeError, IndexError):
            labels = [labels]

        indices, names = self._get_linear_indices(
            self._label_vector_description, full_output=True)
        if labels is None:
            # Must be given as keyword arguments.
            labels = [labels_as_kwargs[name] for name in names]

        else:
            if len(labels) != len(names):
                raise ValueError("expected number of labels is {0}, and {1} "
                    "were given: {2}".format(len(names), len(labels),
                        ", ".join(names)))

        label_vector_indices = self._parse_label_vector_description(
            self._label_vector_description, return_indices=True,
            __columns=names)

        return np.dot(self._coefficients, cannon._build_label_vector_rows(
            label_vector_indices, labels).T).flatten()


    @model.requires_training_wheels
    def solve_labels(self, flux, flux_uncertainties, **kwargs):
        """
        Solve the labels for given fluxes (and uncertainties) using the trained
        model.

        :param fluxes:
            The normalised fluxes. These should be on the same wavelength scale
            as the trained data.

        :type fluxes:
            :class:`~np.array`

        :param flux_uncertainties:
            The 1-sigma uncertainties in the fluxes. This should have the same
            shape as `fluxes`.

        :type flux_uncertainties:
            :class:`~np.array`

        :returns:
            The labels for the given fluxes as a dictionary.

        :raises TypeError:
            If the model is not trained.
        """
        
        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        finite = \
            np.isfinite(self._coefficients[:, 0] * flux * flux_uncertainties)
        p0mask = finite #self._weak_line_mask * finite
        Cinv = 1.0 / (self._scatter[p0mask]**2 + flux_uncertainties[p0mask]**2)
        A = np.dot(self._coefficients[p0mask, :].T,
            Cinv[:, None] * self._coefficients[p0mask, :])
        B = np.dot(self._coefficients[p0mask, :].T,
            Cinv * flux[p0mask])
        initial_vector_p0 = np.linalg.solve(A, B)

        # p0 contains all coefficients, but we only want the linear terms to
        # make an initial estimate.
        indices, names = self._get_linear_indices(self._label_vector_description,
            full_output=True)
        if len(indices) == 0:
            raise NotImplementedError("no linear terms in Cannon model -- TODO")

        # Get the initial guess of just the linear parameters.
        # (Here we make a + 1 adjustment for the first '1' term)
        parameters, p0 = list(names), initial_vector_p0[indices + 1]
        logger.debug("Initial guess: {0}".format(dict(zip(parameters, p0))))

        # Now we need to build up label vector rows by indexing relative to the
        # labels that we will actually be solving for (in this case it's the
        # variable 'names'), and not the labels as they are currently referenced
        # in self._labels
        label_vector_indices = self._parse_label_vector_description(
            self._label_vector_description, return_indices=True,
            __columns=names)

        # Do we have individual line abundances to consider?
        if self._atomic_lines is not None:
            # Need to extend p0 to include individual abundances.
            parameters.extend(self._atomic_lines.keys())

            # Just take the median of the abundances, that should be OK for an
            # initial guess (at any stellar parameters).

            # [TODO] In the future we may just guess this at M_H, but need to
            # be sure we are dealing with M_H and not log_X, or shift by solar.
            p0 = np.hstack([p0, [np.nanmedian(self._labels[p]) \
                for p in self._atomic_lines.keys()]])


        logger.warn("NO EXPLICIT REFERENCES TO STELLAR PARAMETER NAMES")

        # [TODO] Hacky as shhit
        self._tmp_mask = finite

        # Create the function.
        def f(coefficients, *labels):
            # Build the weak lines spectrum.

            stellar_parameters = labels[:3]
            # [TODO] No explicit reference to stellar parameter labels  

            weak_line_fluxes = np.ones(flux.size)
            N = len(self._atomic_lines) if self._atomic_lines is not None else 0
            for i, (label, abundance) \
            in enumerate(zip(self._ew_model.keys(), labels[-N:])):

                wavelengths, ew_coefficients = self._ew_model[label]

                for j, mu in enumerate(wavelengths):
                    # The 10e-4 factor is to turn the EW from milliAngstroms
                    # into Angstroms.
                    expected_ew = atomic._solve_equivalent_width(abundance,
                        ew_coefficients[j], mu, stellar_parameters) * 10e-4

                    p_sigma = 0.35

                    # Translate this into a weak profile.
                    # EW = sqrt(2*pi) * amplitude * sigma
                    # we know the central wavelength, we know the sigma
                    # (EW is in mA, and we want A)
                    amplitude = expected_ew/(np.sqrt(2*np.pi) * p_sigma)
                    weak_line_fluxes *= 1. \
                        - amplitude * np.exp(-(self._wavelengths - mu)**2 \
                            / (2. * p_sigma**2))

            return weak_line_fluxes[self._tmp_mask] *  np.dot(coefficients, cannon._build_label_vector_rows(
                label_vector_indices, labels[:-N]).T).flatten()


        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)

        p_opt, p_covariance = op.curve_fit(f, self._coefficients[finite],
            flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv), absolute_sigma=True,
            **kwds)

        del self._tmp_mask
        

        # We might have solved for any number of parameters, so we return a
        # dictionary.
        logger.debug("TODO: apply offsets as required")
        p_opt = dict(zip(parameters, p_opt))

        logger.debug("Final solution: {0}".format(p_opt))

        if full_output:
            return (p_opt, p_covariance)
        return p_opt


    @model.requires_training_wheels
    def plot_flux_residuals(self, parameter=None, percentile=False, **kwargs):
        """
        Plot the flux residuals as a function of an optional parameter or label.

        :param parameter: [optional]
            The name of a column provided in the labels table for this model. If
            none is provided, the spectra will be sorted by increasing $\chi^2$.

        :type parameter:
            str

        :param percentile: [optional]
            Display model residuals as a percentage of the flux difference, not
            in absolute terms.

        :type percentile:
            bool

        :returns:
            A figure showing the flux residuals.
        """
        return plot.flux_residuals(self, parameter, percentile, **kwargs)


def _check_atomic_lines(labels, atomic_lines):
    """
    Check the atomic line data provided.
    """

    if atomic_lines is None:
        return False

    if not isinstance(atomic_lines, dict):
        raise TypeError("atomic lines should be a dictionary with log(X) "\
            " abundance labels (as keys) and transition tables as values")

    num_transitions = 0
    # Check that the keys actually exist in the _labels table.
    for label, transitions in atomic_lines.items():

        num_transitions += len(transitions)
        if label not in labels.dtype.names:
            raise IndexError("cannot find atomic line abundance label {0} "\
                "in the labels table".format(label))
        
        required_columns = ("wavelength", "species", "excitation_potential",
            "loggf")
        for column in required_columns:
            if column not in transitions.dtype.names:
                raise TypeError("could not find '{0}' column in table of "
                    "transitions for corresponding label '{1}'".format(
                        column, label))

        # Check that the transitions in a given value set are all of the
        # same element.
        species = set(map(int, transitions["species"]))
        if len(species) > 1:
            raise ValueError("the '{0}' abundance label contains mixed "
                "species in the transitions table: {1}".format(label,
                    ", ".join(species)))

    return num_transitions > 0
            

