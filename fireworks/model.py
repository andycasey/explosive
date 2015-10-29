#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" An abstract base model class. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import cPickle as pickle
import numpy as np
import os

def requires_training_wheels(f):
    """
    A decorator for model functions where the model needs training first.
    """
    def wrapper(model, *args, **kwargs):
        if not model._trained:
            raise TypeError("the model needs training/optimising first")
        return f(model, *args, **kwargs)
    return wrapper


def _short_hash(contents):
    """ Return a short hash of some iterable contents. """
    return "".join([str(hash(str(item)))[:10] for item in contents])


class BaseModel(object):

    _trained_attributes = None
    _data_attributes = ("_labels", "_fluxes", "_flux_uncertainties",
        "_wavelengths")
    
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


    @property
    def lv_labels(self):
        """ This should be overwritten by the parent model. """
        raise WTFError("is this your first day?")


    @property
    def _trained_hash(self):
        """ Return a short joint hash of the trained attributes. """

        if not self._trained: return None
        return _short_hash([getattr(self, _) for _ in self._trained_attributes])


    @property
    def _data_hash(self):
        """ Return a short joint hash of the data attributes. """

        return _short_hash([getattr(self, _) for _ in self._data_attributes])


    @requires_training_wheels
    def save(self, filename, with_data=False, overwrite=False):
        """
        Save the (trained) model to disk. This will save all of the relevant
        training attributes, and optionally, the data attributes.

        :param filename:
            The file path where to save the model to.

        :type filename:
            str

        :param with_data: [optional]
            Also the the data used to train the model.

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

        contents = [getattr(self, _) for _ in self._trained_attributes]
        contents += [self._data_hash]
        if with_data:
            contents.extend([getattr(self, _) for _ in self._data_attributes])

        if os.path.exists(filename) and not overwrite:
            raise IOError("filename '{}' exists, asked not to overwrite".format(
                filename))

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

        # Contents is: trained attributes, data hash, [data trained on]
        trained_contents = dict(zip(self._trained_attributes, contents))
        N = len(trained_contents)
        expected_data_hash = contents[N]

        if len(contents) > N + 1:
            # There was data as well.
            data_contents = dict(zip(self._data_attributes, contents[N + 1:]))
            if verify and expected_data_hash is not None:
                actual_data_hash = _short_hash(data_contents)
                if actual_data_hash != expected_data_hash:
                    raise ValueError("expected data hash ({0}) is different "\
                        "({1})".format(expected_data_hash, actual_data_hash))

            # Set the data attributes.
            for k, v in data_contents.items():
                setattr(self, k, v)

        # Set the training attributes.
        for k, v in trained_contents.items():
            setattr(self, k, v)

        self._trained = True
        return True


    @classmethod
    def from_filename(cls, filename, verify=True):
        """
        Initialise a trained model from disk. The saved model must include data.

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

        # Contents is: trained attributes, data hash, data trained on
        trained_contents = dict(zip(cls._trained_attributes, contents))
        N = len(trained_contents)
        expected_data_hash = contents[N]

        if N + 1 >= len(contents):
            raise TypeError("saved model in {} does not include data".format(
                filename))

        # There was data as well.
        if verify and expected_data_hash is not None:
            actual_data_hash = _short_hash(contents[N:])
            if actual_data_hash != expected_data_hash:
                raise ValueError("expected data hash ({0}) is different ({1})"\
                    .format(expected_data_hash, actual_data_hash))

        # Create the model by initialising it with the data attributes.
        model = cls(**dict(zip([_[1:] for _ in cls._data_attributes],
            contents[N+1:])))

        # Set the training attributes.
        for k, v in trained_contents.items():
            setattr(model, k, v)

        model._trained = True
        return model


