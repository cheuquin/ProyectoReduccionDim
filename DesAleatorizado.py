from sklearn.random_projection import johnson_lindenstrauss_min_dim
from numpy import linalg as la
import numpy as np
import sys
import pyfits
import scipy.sparse as sp

wavelength = pyfits.open("wavelengths.fits")[0].data #1000
espectros = pyfits.open("sdss_spectra.fits")[0].data #4000x1000

n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)
print np.shape(espectros)

epsilon = 0.1

s = int(np.log(n)/epsilon)
m = int(np.log(n)/epsilon**2)



#Para la matriz signo.

#Por mientras para tener una matriz delta.
Delta=sp.csr_matrix(espectros[0:m,],shape=(m,d))

#Voy a escribir los datos (V) en formato sparse,
V=sp.csc_matrix(espectros, shape=(n,d))

#Inicio

b=np.zeros((n)) #Linea 1 del algoritmo 2

#L\'inea 2 del algoritmo 2: Los estimadores se inicializan por default
#iguales a uno para todos sus valores.
PesimistaMas=np.ones((n))
PesimistaMenos=np.ones((n))

#L\'inea 3 del algoritmo 2: La lista vacia de los valores sigma.
sigma=[]

ThetaMas=sp.lil_matrix((n,m))
ThetaMenos=sp.lil_matrix((n,m))

nu=sp.lil_matrix((n,m))


#Comienzan los c\'alculos de PesimistaMas
for r in range(0,m):  #L\'inea 5 del algoritmo 2
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    for j in NZIndicesDeltar:
        NZIndicesVj=V.getcol(j).nonzero()[0]
        print NZIndicesVj
        for i in NZIndicesVj:
            #V.getcol(j).getrow(i).toarray()[0][0]
            if b[i]==0.:
                b[i]=1.
                ThetaMas[i,r]=1.
                ThetaMenos[i,r]=1.
                nu[i,r]=0.
            if b[i]==1.:
                temp=(V.getcol(j).getrow(i).toarray()[0][0])**2
                ThetaMas[i,r]=ThetaMas[i,r]-temp/(1+temp)
                ThetaMenos[i,r]=ThetaMenos[i,r]+temp/(1-temp)
                PesimistaMas[i]=PesimistaMas[i]*np.sqrt(1+temp/2)**(-1)
                PesimistaMenos[i]=PesimistaMenos[i]*np.sqrt(1-temp/2)**(-1)
        for i in range(0,n):
            if b[i]==1.:
                PesimistaMas[i]=PesimistaMas[i]*np.sqrt(ThetaMas[i,r])**(-1)
                PesimistaMenos[i]=PesimistaMenos[i]*np.sqrt(ThetaMenos[i,r])**(-1)
        b=np.zeros((n))
            
        





#Lo que sigue despu\'es del inicio






#sys.exit()
