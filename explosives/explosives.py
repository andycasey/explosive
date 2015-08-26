#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon with Chemistry """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import logging
import numpy as np

import scipy.optimize as op
from . import (model, cannon, plot, utils)

logger = logging.getLogger("explosives")


class ExplosivesModel(cannon.CannonModel):

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


    def train(self, label_vector_description, atomic_lines=None, N=None,
        limits=None, pivot=False, **kwargs):
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

        # We'll need this later; store it.
        self._atomic_lines = atomic_lines
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

        # Any atomic lines to model?
        p_sigma = -1
        valid_atomic_lines = _check_atomic_lines(self._labels, atomic_lines)
        if valid_atomic_lines:

            # Build the log(X)->EW models (or vice-versa, sigh)

            # Estimate the FWHM kernel for each star, or estimate from all stars
            # (We need the FWHM to link the EW to an actual flux value.)
            p_sigma = 0.45

        
        # Initialise the requisite arrays.
        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[0]))
        
        pb_size = 100 if kwargs.pop("__progressbar", True) else 0
        pb_message = "Training {0} model from {1} stars with {2} pixels:\n"\
            .format(self.__class__.__name__[:-5], N_stars, N_pixels)
        for i in utils.progressbar(range(N_pixels), pb_message, pb_size):

            # Update the first element of the label vector array to include the
            # predicted EWs.
            if valid_atomic_lines:
                # We have a bunch of atomic transitions, the stellar parameters
                # and atomic abundances of all stars.
                raise NotImplementedError

                # Find out which atomic lines are 'nearby' to this pixel/lambda

                # For each of those lines, calculate the EW that should be seen
                # in each star, given the stellar parameters and abundance of
                # that line.
                
                # Translate the EW to a flux value at this exact pixel/lambda

                # The total depth is taken as 1 - product(d) of all lines.

                # Update the first element of the label vector array for all
                # stars.

                
            raise a

            coefficients[i, :], scatter[i] = cannon._fit_pixel(
                self._fluxes[use, i], self._flux_uncertainties[use, i], lva,
                **kwargs)

            if not np.any(np.isfinite(scatter[i] * coefficients[i, :])):
                logger.warn("No finite coefficients at pixel {}!".format(i))

        self._coefficients, self._scatter, self._offsets, self._trained \
            = coefficients, scatter, offsets, True

        return (coefficients, scatter, offsets)


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
        raise NotImplementedError

        # Get an initial estimate of those parameters from a simple inversion.
        # (This is very much incorrect for non-linear terms).
        finite = np.isfinite(self._coefficients[:, 0]*flux *flux_uncertainties)
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

        # Create the function.
        def f(coefficients, *labels):
            return np.dot(coefficients, cannon._build_label_vector_rows(
                label_vector_indices, labels).T).flatten()

        # Optimise the curve to solve for the parameters and covariance.
        full_output = kwargs.pop("full_output", False)
        kwds = kwargs.copy()
        kwds.setdefault("maxfev", 10000)
        labels, covariance = op.curve_fit(f, self._coefficients[finite],
            flux[finite], p0=p0, sigma=1.0/np.sqrt(Cinv), absolute_sigma=True,
            **kwds)

        # We might have solved for any number of parameters, so we return a
        # dictionary.
        logger.debug("TODO: apply offsets as required")
        labels = dict(zip(names, labels))

        logger.debug("Final solution: {0}".format(labels))

        if full_output:
            return (labels, covariance)
        return labels


    def _repr_label_vector_description(self, label_vector_indices, **kwargs):
        """
        Represent label vector indices as a readable label vector description.

        :param label_vector_indices:
            A list of label vector indices. Each item in the list is expected to
            be a tuple of cross-terms (each in a list).

        :returns:
            A human-readable string of the label vector description.
        """

        return super(self.__class__, self)._repr_label_vector_description(
            label_vector_indices, __first_vector_desc="g(theta)")


    @property
    def _trained_hash(self):
        """
        Return a hash of the trained state.
        """

        if not self._trained: return None
        args = (self._coefficients, self._scatter, self._offsets,
            self._label_vector_description, self._atomic_lines)
        return "".join([str(hash(str(each)))[:10] for each in args])


    @model.requires_training_wheels
    def save(self, filename, with_data=False, overwrite=False, verify=True):
        """
        Save the (trained) model to disk. This will save the label vector
        description, the optimised coefficients and scatter, and pivot offsets.

        :param filename:
            The file path where to save the model to.

        :type filename:
            str

        :param with_data: [optional]
            Save the wavelengths, fluxes and flux uncertainties used to train
            the model.

        :type with_data:
            bool

        :param overwrite: [optional]
            Overwrite the existing file path, if it already exists.

        :type overwrite:
            bool

        :returns:
            True

        :raise TypeError:
            If the model has not been trained, since there is nothing to save.
        """

        # Create a hash of the labels, fluxes and flux uncertainties.
        if verify:
            hashes = [hash(str(_)) for _ in \
                (self._labels, self._fluxes, self._flux_uncertainties)]
        else:
            hashes = None

        contents = \
            [self._label_vector_description, self._coefficients, self._scatter,
                self._offsets, self._atomic_lines, hashes]
        if with_data:
            contents.extend(
                [self._wavelengths, self._fluxes, self._flux_uncertainties])

        with open(filename, "w") as fp:
            pickle.dump(contents, fp, -1)

        return True


    def load(self, filename, verify=True):
        """
        Load a trained model from disk.

        :param filename:
            The file path where to load the model from.

        :type filename:
            str

        :param verify: [optional]
            Verify whether the hashes in the stored filename match what is
            expected from the label, flux and flux uncertainty arrays.

        :type verify:
            bool

        :returns:
            True

        :raises IOError:
            If the model could not be loaded.

        :raises ValueError:
            If the current hash of the labels, fluxes, or flux uncertainties is
            different than what was stored in the filename. Disable this option
            (at your own risk) by setting `verify` to False.
        """

        with open(filename, "r") as fp:
            contents = pickle.load(fp)

        hashes = contents[-1]
        if verify and hashes is not None:
            exp_hash = [hash(str(_)) for _ in \
                (self._labels, self._fluxes, self._flux_uncertainties)]
            descriptions = ("labels", "fluxes", "flux_uncertainties")
            for e_hash, r_hash, descr in zip(exp_hash, hashes, descriptions):
                if e_hash != r_hash:
                    raise ValueError("expected hash for {0} ({1}) is different "
                        "to that stored in {2} ({3})".format(descr, e_hash,
                            filename, r_hash)) 

        if len(contents) > 6:
            self._label_vector_description, self._coefficients, self._scatter, \
                self._offsets, self._atomic_lines, hashes, self._wavelengths, \
                self._fluxes, self._flux_uncertainties = contents
        else:
            self._label_vector_description, self._coefficients, self._scatter, \
                self._offsets, self._atomic_lines, hashes = contents
        self._trained = True

        return True


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
            

