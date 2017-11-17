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

epsilon = 0.4 #Fijar el epsilon de modo que se compruebe m<<d
resto = int((1/epsilon) - int(np.log(n)/epsilon**2)%(1/epsilon))

m = int(np.log(n)/epsilon**2) + resto
s = int(m*epsilon)
t = 3/(2*np.log(2)*m);

p = np.ones((m,d))*epsilon;
Bq = np.reshape(np.arange(m), (s,int(1/epsilon)) ).T;

p[Bq[:,0], 0] = np.zeros( (len(Bq[:,0]))); 
p[Bq[0,0], 0] = 1

start = 0
start2 = 0

for q in range(s):
	for l in range(1,d):
		if q==1: start = 1
		else: start = 0

	for l in range(start,d):
		alfa = np.zeros( len( Bq[:,q] ) )
		for rr in range(0, int(1/epsilon)):
			a = 0
			for j in range(l-1):
				D = 1
				if q==1:
					start2 = 0
				else:
					start2 = 1

				for q_bar in range(q-start2):
					rho = sum( p[Bq[:,q_bar], j]*p[Bq[:,q_bar],l] )
					D = D*(1+rho)

				a = a + (D*p[Bq[rr,q],j])

			alfa[rr] = a
		
		i = np.argmin(alfa)
		p[Bq[:,q],l] = 0
		p[Bq[i,q],l] = 1


print p
sys.exit()


       
