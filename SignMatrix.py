from sklearn.random_projection import johnson_lindenstrauss_min_dim
from scipy import sparse as sp
from numpy import linalg as la
import numpy as np
import sys
import pyfits

wavelength = pyfits.open("wavelengths.fits")[0].data #1000
espectros = pyfits.open("sdss_spectra.fits")[0].data #4000x1000

n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)

epsilon = 0.1
resto = int((1/epsilon) - int(np.log(n)/epsilon**2)%(1/epsilon))

m = int(np.log(n)/epsilon**2) + resto
s = int(m*epsilon)

def signo (Omegaplus, Omegaminus, sigma, sigmaplus, sigmaminus, Sparse):
{
    sum1, sum2 = 0, 0
    radpos, radneg, rademacher = 0, 0, 1

    rad = 1
    for i in range(n):
        for r in range(m):
            for j in np.where(Sparse[:,r]== 1)[0]:
                sum1 += s**2(sigma[i][r] + rad*espectros[i][j])**2*(1-Omegaplus[i][r])/(Omegaplus[i][r]) +
                        (sigma[i][r] + rad*espectros[i][j]**2)
                sum2 += s**2(sigma[i][r] + rad*espectros[i][j])**2*(1-Omegaminus[i][r])/(Omegaminus[i][r]) +
                        (sigma[i][r] + rad*espectros[i][j]**2)
        radpos += sigmaplus[i]*sum1 + sigmaminus[i]*sum2

    rad = -1
    for i in range(n):
        for r in range(m):
            for j in np.where(Sparse[:,r]== 1)[0]:
                sum1 += s**2(sigma[i][r] + rad*espectros[i][j])**2*(1-Omegaplus[i][r])/(Omegaplus[i][r]) +
                        (sigma[i][r] + rad*espectros[i][j]**2)
                sum2 += s**2(sigma[i][r] + rad*espectros[i][j])**2*(1-Omegaminus[i][r])/(Omegaminus[i][r]) +
                        (sigma[i][r] + rad*espectros[i][j]**2)
        radneg += sigmaplus[i]*sum1 + sigmaminus[i]*sum2

    if (radpos > radneg):
        rademacher = -1
    elif (radneg > radpos):
        rademacher = 1
    return rademacher
}


Sparse = sp.lil_matrix([m,d])
t = 3/(2*np.log(2)*m)

b = [0 for i in range(n)]
sigmaplus, sigmaminus = [1 for i in range(n)], [1 for i in range(n)]
Rademacher = [[1 for j in range(d)] for r in range(m)]
Omegaplus = [[0 for i in range(n)] for r in range(m) ]
Omegaminus = [[0 for i in range(n)] for r in range(m) ]

for r in range(m):
    for j in range(len(Sparse[r])):
        for i in range(len(espectros[:,j])):
            if b[i] = 0:
                b[i] = 1
                Omegaplus[i][r] = 1
                Omegaminus[i][r] = 1
                sigma[i][r] = 0
            if b[i] = 1:
                Omegaplus[i][r] = Omegaplus[i][r] - espectros[i][j]**2/(1+espectros[i][j]**2/2)
                Omegaminus[i][r] = Omegaminus[i][r] + espectros[i][j]**2/(1-espectros[i][j]**2/2)
                sigmaplus[i] = sigmaplus[i]*(1 + espectros[i][j]**2/2)**(-1/2)
                sigmaminus[i] = sigmaminus[i]*(1 - espectros[i][j]**2/2)**(-1/2
        for i in np.where(b[:] = 1)[0]:
            sigmaplus[i] = sigmaplus[i]*(Omegaplus[i][r])**(-1/2)
            sigmaminus[i] = sigmaminus[i]*(Omegaminus[i][r])**(-1/2)
        b = [0 for i in range(n)]

for r in range(m):
    for j in range(len(Sparse[r])):
        for i in range(len(espectros[:,j])):
            sigmaplus = sigmaplus*Omegaplus**(1/2)*(1+espectros[i][j]**2/2)**(1/2)*
                    math.exp(-1/4*(s**2*sigma[i][r]**2*(1-Omegaplus[i][r])/(Omegaplus[i][r]) + espectros[i][j]**2 + sigma[i][r]**2 ) )
            sigmaminus = sigmaminus*Omegaminus**(1/2)*(1+espectros[i][j]**2/2)**(1/2)*
                    math.exp(1/4*(s**2*sigma[i][r]**2*(1-Omegaminus[i][r])/(Omegaminus[i][r]) + espectros[i][j]**2 + sigma[i][r]**2 ) )
            Omegaplus = Omegaplus + espectros[i][j]**2/(1 + espectros[i][j]**2/2)
            Omegaminus = Omegaminus - espectros[i][j]**2/(1 - espectros[i][j]**2/2)

        Rademacher[r][j] = signo(Omegaplus, Omegaminus, sigma, sigmaplus, sigmaminus, Sparse)
        sigma[i][r] += Rademacher[r][j]*espectros[i][j]
        sigmaplus[i] = sigmaplus[i]*(Omegaplus[i][r])**(-1/2)*math.exp(1/4*(s**2*sigma[i][r]**2*(1-Omegaplus[i][r])/Omegaplus[i][r] + sigma[i][r]**2 ) )
        sigmaminus[i] = sigmaminus[i]*(Omegaminus[i][r])**(-1/2)*math.exp(1/4*(s**2*sigma[i][r]**2*(1-Omegaminus[i][r])/Omegaminus[i][r] + sigma[i][r]**2))





































sys.exit()
