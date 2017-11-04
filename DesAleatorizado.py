from sklearn.random_projection import johnson_lindenstrauss_min_dim
import numpy as np
import sys
import pyfits

wavelength = pyfits.open("wavelengths.fits")[0].data #1000
espectros = pyfits.open("sdss_spectra.fits")[0].data #4000x1000

n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)

epsilon = 0.6
