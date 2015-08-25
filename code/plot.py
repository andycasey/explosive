#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plots """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def flux_residuals(model, parameter=None, percentile=False, linearise=True,
    mask=None, **kwargs):

    fig, ax = plt.subplots()

    # Generate model fluxes at each trained point.
    indices, label_names = model._get_linear_indices(
        model._label_vector_description, full_output=True)

    model_fluxes = np.nan * np.ones(model._fluxes.shape)
    for i, star in enumerate(model._labels):
        model_fluxes[i] = model.predict([star[label] for label in label_names])

    residuals = model_fluxes - model._fluxes
    
    # Order y-axis by the requested parameter.
    if parameter is None:
        variance = model._flux_uncertainties**2 + model._scatter**2
        chi_sqs = np.sum(residuals**2/variance, axis=1)
        chi_sqs /= model._fluxes.shape[1] - model._coefficients.shape[1] - 2
        y, y_label = chi_sqs, r"$\chi^2$"
        
    else:
        y, y_label = model._labels[parameter], kwargs.pop("y_label", parameter)

    if mask is not None:
        residuals = residuals[mask, :]
        y = y[mask]

    sort_indices = np.argsort(y)
    y, residuals = y[sort_indices], residuals[sort_indices]
    if percentile: residuals /= model._fluxes[sort_indices]

    x = model._wavelengths or np.arange(model._fluxes.shape[1])

    vmin = kwargs.pop("vmin", residuals.min())
    vmax = kwargs.pop("vmax", residuals.max())
    cmap = kwargs.pop("cmap", "Grey")
    
    if linearise:
        image = matplotlib.image.NonUniformImage(ax, interpolation="nearest",
            extent=[x[0], x[-1], y[0], y[-1]], cmap=cmap)
        image.set_data(x, y, np.clip(residuals, vmin, vmax))
        ax.images.append(image)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    else: 
        image = ax.imshow(residuals, aspect="auto", vmin=vmin, vmax=vmax,
            interpolation="nearest", extent=[x[0], x[-1], y[0], y[-1]],
            cmap=cmap)

    colorbar = plt.colorbar(image)
    label = kwargs.pop("colorbar_label", r"$\Delta{}F(\lambda)$")
    units = r" $[\%]$" if percentile else ""
    colorbar.set_label(label + units)
    ax.set_xlabel("Pixel")
    ax.set_ylabel(y_label)

    fig.tight_layout()

    return fig
    

def label_residuals(model, parameters, percent=False):
    """
    Plot the residuals between the inferred and expected labels with respect to
    some set of parameters.
    """

    if isinstance(parameters, (str, unicode)):
        parameters = [parameters]

    # x-axis: some parameter
    # y-axes: differences in inferred labels

    #indices, label_names = model._get_linear_indices(
    #    model._label_vector_description, full_output=True)

    labels, expected, inferred = model.label_residuals

    N_labels, N_parameters = len(label_names), len(parameters)



    fig, axes = plt.subplots(N_labels, N_parameters)
    for i, label in enumerate(labels):
        for j, parameter in enumerate(parameters):
            
            ax = axes[i, j]

            difference = inferred[:, i] - expected[:, i]
            if percent: difference /= expected[:, i]

            ax.scatter(model._labels[parameter], difference, facecolor="k")

            ax.set_xlabel(parameter)
            ax.set_ylabel("Delta {}".format(label))


