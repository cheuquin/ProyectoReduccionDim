from sklearn.random_projection import johnson_lindenstrauss_min_dim
from numpy import linalg as la
import numpy as np
import sys
#import pyfits
from astropy.io import fits


#wavelength = pyfits.open("wavelengths.fits")[0].data #1000
#espectros = pyfits.open("sdss_spectra.fits")[0].data #4000x1000

wavelength = fits.open("wavelengths.fits")[0].data #1000
espectros = fits.open("sdss_spectra.fits")[0].data #4000x1000

n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)


epsilon = 0.1

s = int(np.log(n)/epsilon)
m = int(np.log(n)/epsilon**2)
t = 3/(2*np.log(2)*m)

Pessimist = []

for i in range(n):
    Pessimist.append(np.exp(-s**2*t*np.log(2) + s*np.log(1+epsilon)*(1-la.norm(espectros[i],4) )/2 ))

Loss = sum(list(Pessimist))

Delta = np.array([[0 for j in range(d)] for i in range(m) ])
delta = int(m/s)

B = [[] for x in range(s)]

for j in range(d):
    for q in range(s):
        B[q] = (list(i for i in range(delta*q, delta + delta*q) ))
        index = np.random.choice(B[q])
        Delta[index][j] = 1


#for q in range(s):
#    sigma = [0 for i in range(n)]
#    for j in range(d):
#       for i in range(n):
#            if (espectros[i][j]!=0):
#               for k in range(j):
#                   if (espectros[i][k]!=0):
#                       for rk in B[q]:
#                           if (p[rk][k] == 1):
#                               if (br == 0):
#                                   br = 1




sys.exit()
