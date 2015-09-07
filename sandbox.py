#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" fireworks sandbox. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np
from astropy.table import Table
from fireworks import fireworks

DATA_PREFIX = "/Users/arc/research/luminosity-cannon/data/APOGEE-Hipparcos"
stars = Table.read("{}.fits.gz".format(DATA_PREFIX))


import scipy.optimize as op
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

transitions = Table(np.array([
    [15194.492, 26.0, 2.223, -4.779],
    [15207.526, 26.0, 5.385, +0.080],
    [15395.718, 26.0, 5.620, -0.341],
    [15490.339, 26.0, 2.198, -4.807],
    # Missing offset is entirely attributable to loggf differences
    [15648.510, 26.0, 5.426, -0.701],
    [15964.867, 26.0, 5.921, -0.128],
    [16040.657, 26.0, 5.874, +0.066],
    [16153.247, 26.0, 5.351, -0.743],
    [16165.032, 26.0, 6.319, +0.723]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])

transitions["wavelength"] = map(air2vac, transitions["wavelength"])

stellar_parameters = np.vstack([stars["TEFF"], stars["LOGG"],
    stars["PARAM_M_H"]]).T

#atomic.approximate_atomic_transitions(stellar_parameters, transitions)


# Create FireworksModel
"""
fluxes = np.memmap("{}-flux.memmap".format(DATA_PREFIX), mode="r", dtype=float)
flux_uncertainties = np.memmap("{}-flux-uncertainties.memmap".format(DATA_PREFIX),
    mode="r", dtype=float)
wavelengths = np.memmap("{}-wavelength.memmap".format(DATA_PREFIX), mode="r",
    dtype=float)

fluxes = fluxes.reshape((len(stars), -1))
flux_uncertainties = flux_uncertainties.reshape(fluxes.shape)

atomic_lines = {
    "FE_H": transitions
}
label_vector_description = "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M" 

if False:
    m = fireworks.FireworksModel(stars, wavelengths, fluxes, flux_uncertainties)
    stuff = m.train(label_vector_description, atomic_lines, X_H=True)
    m.save("temp.pkl", with_data=True, overwrite=True)

else:
    m = fireworks.FireworksModel.from_filename("temp.pkl")

label_names, expected_labels, inferred_labels = m.label_residuals
"""


# Create a FireworksModel with many elements.
fluxes = np.memmap("{}-flux.memmap".format(DATA_PREFIX), mode="r", dtype=float)
flux_uncertainties = np.memmap("{}-flux-uncertainties.memmap".format(DATA_PREFIX),
    mode="r", dtype=float)
wavelengths = np.memmap("{}-wavelength.memmap".format(DATA_PREFIX), mode="r",
    dtype=float)

fluxes = fluxes.reshape((len(stars), -1))
flux_uncertainties = flux_uncertainties.reshape(fluxes.shape)

# Let's try with 5 elements from different nucleosynthetic pathways, all with
# different number of lines and different line qualities.

# Mg 1, Al 1, Si I, Cr I, Co I

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


# It seems this is not in APOGEE.
CrI_transitions = Table(np.array([
    [15680.063, 24.0,   4.697, +0.179],
    [15860.214, 24.0,   4.697, -0.012]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])
CrI_transitions["wavelength"] = np.array(map(air2vac, CrI_transitions["wavelength"]))


CoI_transitions = Table(np.array([
    [16757.7, 27.0,     3.409, -1.230]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])
CoI_transitions["wavelength"] = np.array(map(air2vac, CoI_transitions["wavelength"]))

VI_transitions = Table(np.array([
    [15924., 23.0, 2.138, -1.175]
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])
VI_transitions["wavelength"] = np.array(map(air2vac, VI_transitions["wavelength"]))



Ni1_transitions = Table(np.array([
    [15605.680, 28.0, 5.299, -0.376],
    [15632.654, 28.0, 5.305, -0.106],
    [16584.439, 28.0, 5.299, -0.528],
    [16589.295, 28.0, 5.469, -0.600],
    [16673.711, 28.0, 6.029, +0.317],
    [16815.471, 28.0, 5.305, -0.606],
    [16818.760, 28.0, 6.039, +0.311],
    ]), names=["wavelength", "species", "excitation_potential", "loggf"])

atomic_lines = {
    "MG_H": MgI_transitions,
    "AL_H": AlI_transitions,
    "SI_H": SiI_transitions,
#    "CR_H": CrI_transitions, # Not in APOGEE
    "V_H": VI_transitions,
#    "CO_H": CoI_transitions, # Not in APOGEE
    "NI_H": Ni1_transitions,

}
label_vector_description = "TEFF^4 TEFF^3 TEFF^2 TEFF LOGG LOGG^2 TEFF*LOGG TEFF^2*LOGG TEFF*LOGG^2 PARAM_M_H PARAM_M_H*TEFF PARAM_M_H*TEFF^2 PARAM_ALPHA_M PARAM_M_H*PARAM_ALPHA_M" 

"""
temp_filename = "fireworks-Mg+Al+Si+V+Ni.pkl"

import os
if not os.path.exists(temp_filename):
    model = fireworks.FireworksModel(stars, wavelengths, fluxes, flux_uncertainties)
    result = model.train(label_vector_description, atomic_lines, X_H=True)
    model.save(temp_filename, with_data=True)

else:
    model = fireworks.FireworksModel.from_filename(temp_filename)

label_names, expected_labels, inferred_labels = model.label_residuals
"""

"""
model = fireworks.FireworksModel(stars, wavelengths, fluxes, flux_uncertainties)
atomic_lines = {
    "AL_H": AlI_transitions
}
result = model.train(label_vector_description, atomic_lines=atomic_lines)
model.save("t2.pkl", with_data=True, overwrite=True)
"""

model = fireworks.FireworksModel.from_filename("t2.pkl", verify=False)


# 
"""
From https://github.com/jobovy/apogee/blob/master/apogee/spec/plot.py
(thanks Jo)

_MGI_lines= [air2vac(l) for l in [15740.716,15748.9,15765.8,15879.5,
                                  15886.2,15889.485,15954.477]]
_ALI_lines= [air2vac(l) for l in [16718.957,16763.359]]
_SII_lines= [air2vac(l) for l in [15361.161,15376.831,15833.602,15960.063,
                                  16060.009,16094.787,16215.670,16680.770,
                                  16828.159]]
_KI_lines= [air2vac(l) for l in [15163.067,15168.376]]
_CAI_lines= [air2vac(l) for l in [16136.823,16150.763,16155.236,16157.364]]
_TII_lines= [air2vac(l) for l in [15543.756,15602.842,15698.979,15715.573,
                                  16635.161]]
_VI_lines= [air2vac(15924.)]
_CRI_lines= [air2vac(l) for l in [15680.063,15860.214]]
_MNI_lines= [air2vac(l) for l in [15159.,15217.,15262.]]
_COI_lines= [air2vac(16757.7)]
_NII_lines= [air2vac(l) for l in [15605.680,15632.654,16584.439,16589.295,
                                  16673.711,16815.471,16818.760]]
"""


raise a
