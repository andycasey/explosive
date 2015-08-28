#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Explosives sandbox. """

__author__ = "Andy Casey <arc@ast.cam.ac.uk>"

import numpy as np
from astropy.table import Table
from explosives import explosives

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


# Create ExplosivesModel
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
    m = explosives.ExplosivesModel(stars, wavelengths, fluxes, flux_uncertainties)
    stuff = m.train(label_vector_description, atomic_lines, X_H=True)
    m.save("temp.pkl", with_data=True, overwrite=True)

else:
    m = explosives.ExplosivesModel.from_filename("temp.pkl")

label_names, expected_labels, inferred_labels = m.label_residuals
raise a
