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

    _trained_attributes = ("_coefficients", "_scatter", "_offsets",
        "_label_vector_description",
        "_atomic_lines", "_stellar_parameter_labels")
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

        super(FireworksModel, self).__init__(labels, fluxes, flux_uncertainties,
            wavelengths=wavelengths, verify=verify)


    @property
    @model.requires_training_wheels
    def lv_labels(self):
        """
        Return a list of the labels involved in this model. This includes any
        labels in the description of the label vector, as well as any individual
        abundance labels.
        """

        labels = super(self.__class__, self).lv_labels
        # We check for _atomic_lines instead of _trained because we might train
        # without any atomic lines.
        if self._atomic_lines is not None:
            labels += self._atomic_lines.keys()
        return labels


    def train(self, label_vector_description, N=None, limits=None, pivot=False,
        atomic_lines=None, X_H=True, stellar_parameter_labels=None, **kwargs):
        """
        Train a Cannon model based on the label vector description provided.

        :params label_vector_description:
            The human-readable form of the label vector description.

        :type label_vector_description:
            str or list of str

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

        :returns:
            A three-length tuple containing the model coefficients, the scatter
            in each pixel, and the label offsets.
        """


        # Since building the atomic line models takes longer than building the
        # label vector array, we build the vector array first so that any errors
        # will appear first.
        self._label_vector_description = label_vector_description
        lv = self._parse_label_vector_description(label_vector_description)
        lva, use, offsets = cannon._build_label_vector_array(self._labels, lv,
            N, limits, pivot)

        # Initialise the requisite arrays.
        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[0]))
        weak_line_fluxes = np.ones((N_stars, N_pixels))

        # Any atomic lines to model?
        atomic_lines = _validate_atomic_lines(self._labels, atomic_lines)
        if atomic_lines:

            N_species = len(atomic_lines)
            N_transitions = sum(map(len, atomic_lines.values()))

            msg = []
            for k, v in atomic_lines.items():
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
            if stellar_parameter_labels is None:
                stellar_parameter_labels = ["TEFF", "LOGG", "PARAM_M_H"]
            all_stellar_parameters = np.vstack(
                [self._labels[label] for label in stellar_parameter_labels]).T

            # [TODO] This part is unnecessarily slow. Speed it up.
            # [TODO] It's also probably catagorically wrong.
            atomic_line_model = {}
            for label, transitions in atomic_lines.items():
                ew_coefficients = atomic.approximate_atomic_transitions(
                    all_stellar_parameters, transitions, X_H=X_H, **kwargs)
                atomic_line_model[label] = (transitions, ew_coefficients)

            # Generate the weak line fluxes for each star.
            for i, stellar_parameters in enumerate(all_stellar_parameters):
                for label, (transitions, ew_coefficients) in atomic_line_model.items():

                    abundance = self._labels[label][i]
                    for j, mu in enumerate(transitions["wavelength"]):
                        # The 1e-3 factor is to turn the EW from milliAngstroms
                        # into Angstroms.
                        expected_ew = atomic._solve_equivalent_width(abundance,
                            ew_coefficients[j], mu, stellar_parameters) * 1e-3

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
            atomic_line_model = None


        assert N is None, "whoops?"
        pb_size = 100 if kwargs.pop("__progressbar", True) else 0
        pb_message = "Training {0} model from {1} stars with {2} pixels:\n"\
            .format(self.__class__.__name__[:-5], N_stars, N_pixels)
        for i in utils.progressbar(range(N_pixels), pb_message, pb_size):

            # Train the Cannon on the residuals of the data.
            # I *think* this is OK to do
            # (e.g., Hogg may be wrong??? -- famous last words?)
            coefficients[i, :], scatter[i] = cannon._fit_pixel(
                self._fluxes[use, i] / weak_line_fluxes[use, i],
                self._flux_uncertainties[use, i], lva, **kwargs)

            if not np.any(np.isfinite(scatter[i] * coefficients[i, :])):
                logger.warn("No finite coefficients at pixel {}!".format(i))

        # Save all of these to the model.
        self._trained = True
        self._coefficients, self._scatter, self._offsets \
            = coefficients, scatter, offsets
        self._atomic_lines, self._stellar_parameter_labels \
            = atomic_line_model, stellar_parameter_labels

        return (coefficients, scatter, offsets, atomic_line_model,
            weak_line_fluxes)


    @model.requires_training_wheels
    def predict(self, labels=None, **labels_as_kwargs):
        """
        Predict spectra from the trained model, given the labels.

        :param labels:
            The labels required for the trained model. This should be a N-length
            list matching the number of unique terms in the model, including any
            atomic (weak) line abundances in the order given by `self.lv_labels`
            property. Alternatively, labels can be explicitly given as keyword
            arguments.

        :type labels:
            list

        :returns:
            Model spectra for the given labels.

        :raises TypeError:
            If the model is not trained.
        """
        
        try:
            labels[0]
        except (TypeError, IndexError):
            labels = [labels]

        names = self.lv_labels
        if labels is None:
            labels = [labels_as_kwargs[name] for name in names]
        elif len(labels) != len(names):
            raise ValueError("expected number of labels is {0}, and {1} were "\
                "given: {2}".format(len(names), len(labels),
                    ", ".join(names)))

        # Generate the Cannon-ical flux.
        label_vector_indices = self._parse_label_vector_description(
            self._label_vector_description, return_indices=True,
            __columns=names)

        offsets = np.array([self._offsets[names] for names in namess])
        fluxes = np.dot(self._coefficients, cannon._build_label_vector_rows(
            label_vector_indices, labels - offsets).T).flatten()

        # Include treatment of any atomic lines.
        if self._atomic_lines is not None:

            N = len(self._atomic_lines)
            stellar_parameters = [labels[labels.index(_)] \
                for _ in self._stellar_parameter_labels]
            
            weak_line_fluxes = np.ones(cannonical_flux.size)
            for i, (label, abundance) \
            in enumerate(zip(self._atomic_lines.keys(), labels[-N:])):

                transitions, ew_coefficients = self._atomic_lines[label]

                for j, mu in enumerate(transitions["wavelength"]):
                    # The 1e-3 factor is to turn the EW from milliAngstroms to A
                    expected_ew = atomic._solve_equivalent_width(abundance,
                        ew_coefficients[j], mu, stellar_parameters) * 1e-3

                    p_sigma = 0.35

                    # Translate this into a Gaussian profile.
                    # EW = sqrt(2*pi) * amplitude * sigma
                    # we know the central wavelength, we know the sigma
                    amplitude = expected_ew/(np.sqrt(2*np.pi) * p_sigma)
                    weak_line_fluxes *= 1. \
                        - amplitude * np.exp(-(self._wavelengths - mu)**2 \
                            / (2. * p_sigma**2))

            fluxes *= weak_line_fluxes
        return fluxes


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
        Cinv = 1.0 / (self._scatter[finite]**2 + flux_uncertainties[finite]**2)
        A = np.dot(self._coefficients[finite, :].T,
            Cinv[:, None] * self._coefficients[finite, :])
        B = np.dot(self._coefficients[finite, :].T,
            Cinv * flux[finite])
        initial_vector_p0 = np.linalg.solve(A, B)

        # p0 contains all coefficients, but we only want the linear terms to
        # make an initial estimate.
        indices, names = self._get_linear_indices(self._label_vector_description,
            full_output=True)
        if len(indices) == 0:
            raise NotImplementedError("no linear terms in Cannon model -- TODO")

        # Get the initial guess of just the linear parameters.
        # (Here we make a + 1 adjustment for the first '1' term)
        p0 = initial_vector_p0[indices + 1]
        logger.debug("Initial guess: {0}".format(dict(zip(names, p0))))

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
            names.extend(self._atomic_lines.keys())

            # Just take the median of the abundances, that should be OK for an
            # initial guess (at any stellar parameters).

            # [TODO] In the future we may just guess this at M_H, but need to
            # be sure we are dealing with M_H and not log_X, or shift by solar.
            p0 = np.hstack([p0, [np.nanmedian(self._labels[p]) \
                for p in self._atomic_lines.keys()]])

        self._finite_mask = finite

        # Create the function.
        def f(coefficients, *labels):
            # Build the weak lines spectrum.

            stellar_parameters = [labels[self.lv_labels.index(_)] \
                for _ in self._stellar_parameter_labels]

            weak_line_fluxes = np.ones(flux.size)
            N = len(self._atomic_lines)
            for i, (label, abundance) \
            in enumerate(zip(self._atomic_lines.keys(), labels[-N:])):

                transitions, ew_coefficients = self._atomic_lines[label]

                for j, mu in enumerate(transitions["wavelength"]):
                    # The 1e-3 factor is to turn the EW from milliAngstroms
                    # into Angstroms.
                    expected_ew = atomic._solve_equivalent_width(abundance,
                        ew_coefficients[j], mu, stellar_parameters) * 1e-3

                    p_sigma = 0.35

                    # Translate this into a weak profile.
                    # EW = sqrt(2*pi) * amplitude * sigma
                    # we know the central wavelength, we know the sigma
                    # (EW is in mA, and we want A)
                    amplitude = expected_ew/(np.sqrt(2*np.pi) * p_sigma)
                    weak_line_fluxes *= 1. \
                        - amplitude * np.exp(-(self._wavelengths - mu)**2 \
                            / (2. * p_sigma**2))

            return weak_line_fluxes[self._finite_mask] \
                * np.dot(coefficients, cannon._build_label_vector_rows(
                    label_vector_indices, labels[:-N]).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)

        p_opt, p_covariance = op.curve_fit(f, self._coefficients[finite],
            flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv), absolute_sigma=True,
            **kwds)

        # Remove the temporary finite mask.
        del self._finite_mask
        
        # We might have solved for any number of parameters, so we return a dict
        p_opt = { k: p_opt[i] + self._offsets[k] for i, k in enumerate(names) }
        logger.debug("Final solution: {}".format(p_opt))

        if full_output:
            return (p_opt, p_covariance)
        return p_opt


def _validate_atomic_lines(labels, atomic_lines):
    """
    Check the atomic line data provided.
    """

    if atomic_lines is None:
        return False

    if not isinstance(atomic_lines, dict):
        raise TypeError("atomic lines should be a dictionary with log(X) "\
            " abundance labels (as keys) and transition tables as values")

    atomic_lines = OrderedDict(sorted(atomic_lines.items(), 
        key=lambda _: min(_[1]["species"])))
        
    valid_atomic_lines = {}
    # Check that the keys actually exist in the _labels table.
    for label, transitions in atomic_lines.items():

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

        valid_atomic_lines[label] = transitions

    return valid_atomic_lines
            

