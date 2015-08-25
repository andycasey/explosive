#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An abstract base model class. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np

def requires_training_wheels(f):
    """
    A decorator for model functions where the model needs training first.
    """
    def wrapper(model, *args, **kwargs):
        if not model._trained:
            raise TypeError("the model needs training first")
        return f(model, *args, **kwargs)
    return wrapper


class BaseModel(object):

    def __init__(self, labels, fluxes, flux_uncertainties, wavelengths=None,
        verify=True):
        """
        Initialise a base model.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            :class:`~astropy.table.Table`

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

        :param wavelengths: [optional]
            The wavelengths corresponding to the given pixels.

        :type wavelengths:
            :class:`np.array`
        """

        self._check_data(labels, fluxes, flux_uncertainties)
        self._wavelengths = wavelengths
        self._trained = False
        self._labels = labels
        self._label_vector_description = None
        self._fluxes, self._flux_uncertainties = fluxes, flux_uncertainties

        if verify:
            self._check_data(labels, fluxes, flux_uncertainties, wavelengths)
            self._check_forbidden_label_characters("^*")
        return None


    def _check_data(self, labels, fluxes, flux_uncertainties, wavelengths=None):
        """
        Check that the labels, flux and flux uncertainty data is OK.

        :param labels:
            A table with columns as labels, and stars as rows.

        :type labels:
            :class:`~astropy.table.Table`

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

        :param wavelengths: [optional]
            The wavelengths corresponding to the given pixels.

        :type wavelengths:
            :class:`np.array`
        """

        fluxes = np.atleast_2d(fluxes)
        flux_uncertainties = np.atleast_2d(flux_uncertainties)

        if len(labels) != fluxes.shape[0]:
            raise ValueError("the fluxes should have shape (n_stars, n_pixels) "
                "where n_stars is the number of rows in the labels array")

        if fluxes.shape != flux_uncertainties.shape:
            raise ValueError("the flux and flux uncertainties array should have"
                " the same shape")

        if len(labels) == 0:
            raise ValueError("no stars (labels) given")

        if wavelengths is not None:
            wavelengths = np.atleast_1d(wavelengths)
            if wavelengths.size != fluxes.shape[1]:
                raise ValueError("mis-match between number of wavelength values"
                    " ({0}) and flux values ({1})".format(
                        wavelengths.size, fluxes.shape[1]))

        return None


    def _check_forbidden_label_characters(self, characters):
        """
        Check the label table for potentially forbidden characters.

        :param characters:
            A string of forbidden characters.

        :type characters:
            str

        :returns:
            True

        :raises ValueError:
            If a forbidden character is in a potential label name.
        """

        for column in self._labels.dtype.names:
            for character in characters:
                if character in column:
                    raise ValueError("forbidden character '{0}' is in potential"
                        " label '{1}' - to ignore this use verify=False".format(
                            character, column))
        return True
