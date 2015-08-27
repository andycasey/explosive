#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Cannon Chemistry """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import logging
import numpy as np

import scipy.optimize as op
from . import (model, plot, utils)

logger = logging.getLogger("explosives")


class CannonModel(model.BaseModel):

    _trained_attributes = ("_coefficients", "_scatter", "_offsets")
    _data_attributes = ("_labels", "_wavelengths", "_fluxes",
        "_flux_uncertainties")

    def __init__(self, labels, wavelengths, fluxes, flux_uncertainties,
        verify=True):
        """
        Initialise a Cannon model.

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

        super(CannonModel, self).__init__(labels, fluxes, flux_uncertainties,
            wavelengths, verify)


    def train(self, label_vector_description, N=None, limits=None, pivot=False,
        **kwargs):
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

        :returns:
            A three-length tuple containing the model coefficients, the scatter
            in each pixel, and the label offsets.
        """

        self._label_vector_description = label_vector_description
        
        # Build the label vector array.
        lv = self._parse_label_vector_description(label_vector_description)
        lva, use, offsets = _build_label_vector_array(self._labels, lv,
            N, limits, pivot)

        # Initialise the requisite arrays.
        N_stars, N_pixels = self._fluxes.shape[:2]
        scatter = np.nan * np.ones(N_pixels)
        coefficients = np.nan * np.ones((N_pixels, lva.shape[0]))
        
        pb_size = 100 if __kwargs.pop("__progressbar", True) else 0
        pb_message = "Training {0} model from {1} stars with {2} pixels:\n"\
            .format(self.__class__.__name__[:-5], N_stars, N_pixels)
        for i in utils.progressbar(range(N_pixels), pb_message, pb_size):

            coefficients[i, :], scatter[i] = _fit_pixel(
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

        return np.dot(self._coefficients, _build_label_vector_rows(
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
            return np.dot(coefficients, _build_label_vector_rows(
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


    @property
    @model.requires_training_wheels
    def label_residuals(self):
        """
        Calculate the residuals of the inferred labels using the trained model.

        :returns:
            The corresponding label names, the expected labels for all stars in
            the training set, and the inferred labels.

        :raises TypeError:
            If the model is not trained.
        """

        try:
            if self._residuals_hash == self._trained_hash \
            and self._trained_hash is not None:
                return self._residuals_cache

        except AttributeError:
            None

        label_indices, label_names = self._get_linear_indices(
            self._label_vector_description, full_output=True)
        expected_labels = np.zeros((self._fluxes.shape[0], len(label_names)))
        inferred_labels = np.zeros((self._fluxes.shape[0], len(label_names)))

        for i, (flux, uncertainty) \
        in enumerate(zip(self._fluxes, self._flux_uncertainties)):
            inferred = self.solve_labels(flux, uncertainty)
            for j, label_name in enumerate(label_names):
                expected_labels[i, j] = self._labels[label_name][i]
                inferred_labels[i, j] = inferred[label_name]

        # Cache for future, unless the training state changes.
        self._residuals_hash = self._trained_hash
        self._residuals_cache = (label_names, expected_labels, inferred_labels)
        return self._residuals_cache


    @model.requires_training_wheels
    def cross_validate(self, label_vector_description=None, **kwargs):
        """
        Perform leave-one-out cross-validation on the trained model.

        :params label_vector_description: [optional]
            The human-readable form of the label vector description. If None is
            given, the currently trained label vector description is used.

        :type label_vector_description:
            str

        :returns:
            A two-length tuple containing an array of the expected train labels
            for each star, and the inferred labels.
        """

        # Initialise arrays.
        if label_vector_description is None:
            label_vector_description = self._label_vector_description

        label_indices, label_names = self._get_linear_indices(
            label_vector_description, full_output=True)

        N_realisations, N_labels = self._fluxes.shape[0], len(label_names)
        inferred_test_labels = np.nan * np.ones((N_realisations, N_labels))
        expected_test_labels = np.ones((N_realisations, N_labels))

        # Go through each combination.
        # [TODO] Thread everything.
        for i in range(N_realisations):

            mask = np.ones(N_realisations, dtype=bool)
            mask[i] = False

            # Create a model to use so we don't overwrite self.
            model = self.__class__(self._labels[mask], self._fluxes[mask, :],
                self._flux_uncertainties[mask, :])
            model.train(label_vector_description, **kwargs)

            # Solve for the one left out.
            try:
                inferred_labels = model.solve_labels(
                    self._fluxes[~mask, :].flatten(),
                    self._flux_uncertainties[~mask, :].flatten())
            except:
                logger.exception("Exception in solving star with index {0} in "\
                    "cross-validation".format(i))

            else:
                # Save inferred test labels.
                for j, name in enumerate(label_names):
                    inferred_test_labels[i, j] = inferred_labels[name]

            finally:
                # Save expected test labels.
                for j, name in enumerate(label_names):
                    expected_test_labels[i, j] = self._labels[~mask][name]
                
        return (label_names, elxpected_test_labels, inferred_test_labels)


    def _get_linear_indices(self, label_vector_description, full_output=False):
        """
        Return indices of linear labels in the given label vector description.

        :param label_vector_description:
            The human-readable label vector description.

        :type label_vector_description:
            str

        :param full_output: [optional]
            Return the indices and the names of the corresponding linear labels.

        :type full_output:
            bool

        :returns:
            The indices of the linear terms in the label vector description. If
            `full_output` is True, then the corresponding label names are also
            provided. The indices are index-zeroed (i.e., there is no adjustment
            for the zeroth term of a label vector array typically being '1').
        """

        lvi = self._parse_label_vector_description(label_vector_description,
            verbose=False)

        names = []
        indices = []
        for i, term in enumerate(lvi):
            if len(term) == 1 and term[0][1] == 1:
                names.append(term[0][0])
                indices.append(i)
        names, indices = tuple(names), np.array(indices)

        if full_output:
            return (indices, names)
        return indices


    def _parse_label_vector_description(self, label_vector_description,
        return_indices=False, **kwargs):
        """
        Parse a human-readable label vector description into indices (that refer
        back to the labels table) and orders for all of the cross-terms.
        
        :param label_vector_description:
            The human-reable label vector description. These labels are expected
            to be columns in the labels table that was supplied when the class
            was initialised.

        :type label_vector_description:
            str

        :param return_indices: [optional]
            Return the indices corresponding to the label table instead of the
            actual parameter name itself.

        :type return_indices:
            bool

        :returns:
            A list where each item contains indices and order information to
            represent the label vector description.
        """

        if isinstance(label_vector_description, (str, unicode)):
            label_vector_description = label_vector_description.split()

        # This allows us to parse arbitrary columns instead of referencing
        # based on the self._labels. Use with caution! Note: if using this,
        # always turn off verbosity.
        columns = kwargs.pop("__columns", self._labels.colnames)

        # Functions to parse the parameter (or index) and order for each term.
        order = lambda t: int((t.split("^")[1].strip() + " ").split(" ")[0]) \
            if "^" in t else 1

        if return_indices:
            param = lambda d: columns.index((d + "^").split("^")[0].strip())
        else:
            param = lambda d: (d + "^").split("^")[0].strip()

        theta = []
        for description in label_vector_description:

            # Is it just a parameter?
            try:
                index = columns.index(description.strip())

            except ValueError:

                # Either the description is not a column in the labels table, or
                # this is a cross-term.

                if "*" in description:
                    cross_terms = []
                    for ct in description.split("*"):
                        try:
                            index = param(ct)
                        except ValueError:
                            raise ValueError("couldn't interpret '{0}' in the "\
                                "label '{1}' as a parameter coefficient".format(
                                    *map(str.strip, (ct, description))))
                        cross_terms.append((param(ct), order(ct)))
                    theta.append(cross_terms)

                elif "^" in description:
                    theta.append([(
                        param(description),
                        order(description)
                    )])

                else:
                    raise ValueError("could not interpret '{0}' as a parameter"\
                        " coefficient description".format(description))
            else:
                theta.append([(param(description), order(description))])

        if kwargs.pop("verbose", True) and not return_indices:
            logger.info("Training Cannon model using label vector description:"\
                " {}".format(self._repr_label_vector_description(theta)))

        return theta


    def _repr_label_vector_description(self, label_vector_indices, **kwargs):
        """
        Represent label vector indices as a readable label vector description.

        :param label_vector_indices:
            A list of label vector indices. Each item in the list is expected to
            be a tuple of cross-terms (each in a list).

        :returns:
            A human-readable string of the label vector description.
        """

        string = [kwargs.pop("__first_vector_desc", "1")]
        for cross_terms in label_vector_indices:
            sub_string = []
            for descr, order in cross_terms:
                if order > 1:
                    sub_string.append("{0}^{1}".format(descr, order))
                else:
                    sub_string.append(descr)

            if len(sub_string) > 1:
                string.append("({})".format(" * ".join(sub_string)))
            else:
                string.append(sub_string[0])

        return " + ".join(string)


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



def _fit_coefficients(fluxes, flux_uncertainties, scatter, lv_array,
    full_output=False):
    """
    Fit model coefficients and scatter to a given set of normalised fluxes for a
    single pixel.

    :param fluxes:
        The normalised fluxes for a single pixel (in many stars).

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma uncertainties in normalised fluxes. This should have the
        same shape as `fluxes`.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array for each pixel.

    :type lv_array:
        :class:`~np.ndarray`

    :param full_output: [optional]
        Return the coefficients and the covariance matrix.

    :type full_output:
        bool

    :returns:
        The label vector coefficients for the pixel, the inverse variance matrix
        and the total pixel variance.
    """

    variance = flux_uncertainties**2 + scatter**2
    CiA = lv_array.T * np.tile(1./variance, (lv_array.shape[0], 1)).T
    ATCiAinv = np.linalg.inv(np.dot(lv_array, CiA))

    ATY = np.dot(lv_array, fluxes/variance)
    coefficients = np.dot(ATCiAinv, ATY)

    return (coefficients, ATCiAinv, variance)
    

def _pixel_scatter_nll(scatter, fluxes, flux_uncertainties, lv_array,
    debug=False):
    """
    Return the negative log-likelihood for the scatter in a single pixel.

    :param scatter:
        The model scatter in the pixel.

    :type scatter:
        float

    :param fluxes:
        The fluxes for a given pixel (in many stars).

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma uncertainties in the fluxes for a given pixel. This should
        have the same shape as `fluxes`.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array for each star, for the given pixel.

    :type lv_array:
        :class:`~np.ndarray`

    :param debug: [optional]
        Re-raise captured exceptions.

    :type debug:
        bool

    :returns:
        The log-likelihood of the log scatter, given the fluxes and the label
        vector array.

    :raises np.linalg.linalg.LinAlgError:
        If there was an error in inverting a matrix, and `debug` is set to True.
    """

    if 0 > scatter:
        return np.inf

    try:
        # Calculate the coefficients for the given level of scatter.
        coefficients, ATCiAinv, variance = _fit_coefficients(fluxes,
            flux_uncertainties, scatter, lv_array)

    except np.linalg.linalg.LinAlgError:
        if debug: raise
        return np.inf

    model = np.dot(coefficients, lv_array)

    return 0.5 * np.sum((fluxes - model)**2 / variance) \
        + 0.5 * np.sum(np.log(variance))


def _fit_pixel(fluxes, flux_uncertainties, lv_array, debug=False):
    """
    Return the optimal label vector coefficients and scatter for a pixel, given
    the fluxes, uncertainties, and the label vector array.

    :param fluxes:
        The fluxes for the given pixel, from all stars.

    :type fluxes:
        :class:`~np.array`

    :param flux_uncertainties:
        The 1-sigma flux uncertainties for the given pixel, from all stars.

    :type flux_uncertainties:
        :class:`~np.array`

    :param lv_array:
        The label vector array. This should have shape `(N_stars, N_terms + 1)`.

    :type lv_array:
        :class:`~np.ndarray`

    :param debug: [optional]
        Re-raise exceptions that would otherwise be suppressed (and a large
        scatter provided).

    :type debug:
        bool

    :returns:
        The optimised label vector coefficients and scatter for this pixel.
    """

    # Get an initial guess of the scatter.
    scatter = np.var(fluxes) - np.median(flux_uncertainties)**2
    scatter = np.sqrt(scatter) if scatter >= 0 else np.std(fluxes)
    
    # Optimise the scatter, and at each scatter value we will calculate the
    # optimal vector coefficients.
    op_scatter, fopt, direc, n_iter, n_funcs, warnflag = op.fmin_powell(
        _pixel_scatter_nll, scatter, args=(fluxes, flux_uncertainties, lv_array),
        disp=False, full_output=True)

    if warnflag > 0:
        logger.warn("Warning: {}".format([
            "Maximum number of function evaluations made during optimisation.",
            "Maximum number of iterations made during optimisation."
            ][warnflag - 1]))

    # Calculate the coefficients at the optimal scatter value.
    # Note that if we can't solve for the coefficients, we should just set them
    # as zero and send back a giant variance.
    try:
        coefficients, _, __ = _fit_coefficients(fluxes, flux_uncertainties,
            op_scatter, lv_array)

    except np.linalg.linalg.LinAlgError:
        logger.exception("Failed to calculate coefficients")
        if debug: raise

        return (np.zeros(lv_array.shape[0]), 10e8)

    else:
        return (coefficients, op_scatter)


def _build_label_vector_rows(label_vector, labels):
    """
    Build a label vector row from a description of the label vector (as indices
    and orders to the power of) and the label values themselves.

    For example: if the first item of `labels` is `A`, and the label vector
    description is `A^3` then the first item of `label_vector` would be:

    `[[(0, 3)], ...`

    This indicates the first label item (index `0`) to the power `3`.

    :param label_vector:
        An `(index, order)` description of the label vector. 

    :type label_vector:
        list

    :param labels:
        The values of the corresponding labels.

    :type labels:
        list

    :returns:
        The corresponding label vector row.
    """
    
    columns = [np.ones(len(labels))]
    for cross_terms in label_vector:
        column = 1
        for descr, order in cross_terms:
            column *= np.array(labels[descr]).flatten()**order
        columns.append(column)

    try:
        return np.vstack(columns).T

    except ValueError:
        columns[0] = np.ones(1)
        return np.vstack(columns).T


def _build_label_vector_array(labels, label_vector, N=None, limits=None,
    pivot=True, ignore_non_finites=True):
    """
    Build the label vector array.

    :param labels:
        The labels for each star as a table. This should have `N_star` rows.

    :type labels:
        :class:`~astropy.table.Table`

    :param label_vector:
        The label vector description in `(index, order)` form.

    :type label_vector`
        list

    :param N: [optional]
        The number of stars to use for the label vector array. If None is given,
        then all stars (rows) will be used.

    :type N:
        int

    :param limits: [optional]
        Place limits on the labels to train with. This should be provided as a
        dictionary where the labels are keys and two-length tuples of
        `(lower, upper)` values are given as values.

    :type limits:
        dict

    :param pivot: [optional]
        Pivot about some mean label values.

    :type pivot:
        bool

    :param ignore_non_finites: [optional]
        Ignore rows with non-finite labels.

    :type ignore_non_finites:
        bool

    :returns:
        A three-length tuple containing the label vector array, a boolean array
        of which stars (rows) from `labels` that were used (in case `N` and/or
        `limits` were provided, or there were non-finite labels and
        `ignore_non_finites` is set to True) and the coefficient offsets
        (for when `pivot` is True).
    """

    logger.debug("Building Cannon label vector array")

    indices = np.ones(len(labels), dtype=bool)
    if limits is not None:
        for parameter, (lower_limit, upper_limit) in limits.items():
            indices *= (upper_limit >= labels[parameter]) * \
                (labels[parameter] >= lower_limit)

    indices = np.where(indices)[0]
    if N is not None:
        indices = indices[np.linspace(0, indices.sum() - 1, N, dtype=int)]
    
    labels = labels[indices]
    if pivot:
        raise NotImplementedError
        offsets = labels.mean(axis=0)
        labels -= offsets
    else:
        offsets = np.zeros(len(labels.colnames))

    lva = _build_label_vector_rows(label_vector, labels).T
    if ignore_non_finites:
        finite = np.all(np.isfinite(lva), axis=0)
        lva, indices = lva[:, finite], indices[finite]

    elif not np.all(np.isfinite(lva)):
        logger.warn("Non-finite labels identified in the label vector array!")

    use = np.zeros(len(labels), dtype=bool)
    use[indices] = True

    return (lva, use, offsets)

