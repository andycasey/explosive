#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Plots """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

cmaps = {
    "GnBu": [(0.90354479621438422, 0.96276816830915568, 0.88175317890503824),
             (0.84367551873711977, 0.93903883485233086, 0.82059209066278793),
             (0.77674741815118231, 0.91252595677095305, 0.76221454704509062),
             (0.67044984663234042, 0.87118801229140341, 0.71497118192560527),
             (0.546020769254834,   0.82405229806900027, 0.74740485934650192),
             (0.41868512595401092, 0.76462900287964763, 0.78985007019603959),
             (0.29457901909070855, 0.68936564711963433, 0.82066898416070377),
             (0.19123414667213665, 0.57420994464088881, 0.75866206744137932),
             (0.092195312678813937, 0.47040370585871677, 0.70579009827445538),
             (0.031372550874948502, 0.36416763593168822, 0.62755865349489104) ],
    "RdBu_r": [(0.11864668090699937, 0.37923876006229257, 0.64567475868206403),
             (0.23660131996753175, 0.54186853357389864, 0.74702039185692293),
             (0.48143024129026069, 0.7148789097281063, 0.83944637635174923),
             (0.73241062725291528, 0.8537485669640934, 0.91626298076966228),
             (0.90142253567190733, 0.93679354471318865, 0.95624759968589335),
             (0.97923875556272622, 0.91910804019254799, 0.88373703115126667),
             (0.97970011655022116, 0.78408305785235233, 0.68489044554093303),
             (0.92226067360709696, 0.56747406545807333, 0.44867361117811766),
             (0.81153403660830326, 0.32110727271612954, 0.27581700390460445),
             (0.66920416904430768, 0.084890428419206659, 0.16401384522517523)],
    "GnBu_d": [(0.21084711176897186, 0.28135332939671537, 0.30823529626808915),
 (0.22169422353794371, 0.36270665879343067, 0.41647059253617835),
 (0.23254133530691556, 0.44405998819014603, 0.5247058888042675),
 (0.24386006063106011, 0.5289504188649794, 0.6376470675187953),
 (0.25470717240003193, 0.61030374826169487, 0.74588236378688433),
 (0.30033577836416903, 0.67208511502135038, 0.79375113365696925),
 (0.3807458785234713, 0.71429451914394604, 0.78125337712904985),
 (0.46465206999404751, 0.75833911475013283, 0.76821223988252529),
 (0.54506217015334979, 0.80054851887272849, 0.75571448335460589),
 (0.62547227031265207, 0.84275792299532426, 0.74321672682668649)],
    }


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
    cmap = kwargs.pop("cmap", matplotlib.colors.ListedColormap(cmaps["GnBu_d"]))

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
    

def model_scatter(scatter, wavelengths=None, **kwargs):

    fig, ax = plt.subplots()
    x = wavelengths if wavelengths is not None else np.arange(scatter.size) + 1

    kwds = {"c": "k"}
    kwds.update(**kwargs)
    ax.plot(x, scatter, **kwds)
    ax.set_xlabel("Wavelength" if wavelengths is not None else "Pixel")
    ax.set_ylabel(r"$s_\lambda$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, kwargs.pop("ylim", 0.1))

    fig.tight_layout()

    return fig


def spectra(wavelengths, observed_fluxes, model_fluxes, **kwargs):


    title = kwargs.pop("title", None)
    fig, ax = plt.subplots(figsize=(wavelengths.size / 500., 4))

    ax.plot(wavelengths, observed_fluxes, c="k")
    ax.plot(wavelengths, model_fluxes, c="r")

    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    if title is not None:
        ax.set_title(title)

    fig.tight_layout()

    return fig



def label_residuals(labels, expected, inferred, aux=None, aux_label=None,
    **kwargs):
    """
    Plot the residuals between the inferred and expected labels with respect to
    some set of parameters.

    :param aux: [optional]
        The auxiliary label to colour the scatter points by.

    :type aux:
        str or None
    """

    N = len(labels)
    cols = np.ceil(N**0.5)
    rows = np.ceil(N / cols)
    cols, rows = map(int, (cols, rows))

    fig, axes = plt.subplots(cols, rows)
    axes = np.array([axes]) if N == 1 else axes.flatten()

    kwds = {}
    if aux is not None:
        kwds["c"] = aux
        kwds["vmin"] = np.nanmin(aux)
        kwds["vmax"] = np.nanmax(aux)
        
    else:
        kwds["facecolor"] = "k"

    kwds.update(**kwargs)

    for i, (ax, label) in enumerate(zip(axes, labels)):

        #residual = inferred[:, i] - expected[:, i]
        #if percentile: residual *= 100./expected[:, i]

        residuals = inferred[:, i] - expected[:, i]
        scat = ax.scatter(expected[:, i], inferred[:, i], **kwds)

        # Limits and lines.
        limits = [
            min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ]
        # Show a 1:1 line.
        ax.plot(limits, limits, c="#cccccc", zorder=-100)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        # Labels.
        ax.set_xlabel(label)
        ax.set_ylabel(label)

        # Title.
        #ax.set_title(r"$\mu\,\,/\,\,\sigma\,\,=\,\,{0:.2f}\,\,/\,\,{1:.2f}$"\
        #    .format(np.nanmean(residuals), np.nanstd(residuals)), fontsize=8)
        ax.set_title("mu / sigma = {0:.2f} / {1:.2f}".format(
            np.nanmean(residuals), np.nanstd(residuals)), fontsize=8)

    if aux is not None:
        cbar = plt.colorbar(scat)
        cbar.set_label(aux_label or "")

    for ax in axes[N:]:
        ax.set_visible(False)

    return fig
