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
t = 3/(2*np.log(2)*m)
'''
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


#Voy a escribir Delta en formato sparse, tal vez deberia venir de antes as\'i,
Delta=sp.csr_matrix(Delta, shape=(m,d)) #Ayuda memoria para el futuro: getcol(i)
'''
#Voy a escribir los datos (V) en formato sparse,
V=sp.csc_matrix(espectros, shape=(n,d))

#Inicio

b=np.zeros((n)) #Linea 1 del algoritmo 2

#L\'inea 2 del algoritmo 2: Los estimadores se inicializan por default
#iguales a uno para todos sus valores.
PesimistaMas=np.ones((n))
PesimistraMenos=np.ones((n))

#L\'inea 3 del algoritmo 2: La lista vacia de los valores sigma.
sigma=[]

ThetaMas=sp.lil_matrix((n,m))
ThetaMenos=sp.lil_matrix((n,m))

nu=sp.lil_matrix((n,m))

print type(V[1])
print type(V.getrow(1))

#print V[1].getcol(0)
#  (0, 0)        2.18929

#Comienzan los c\'alculos de PesimistaMas
for r in range(0,m):  #L\'inea 5 del algoritmo 2
    for j in range(0,d):
        for i in range(0,n):
            if b[i]==0.:
                b[i]=1.
                ThetaMas[i,r]=1.
                ThetaMenos[i,r]=1.
                nu[i,r]=0.
            if b[i]==1.:
                ThetaMas[i,r]=ThetaMas[i,r]#-
                ThetaMenos[i,r]=ThetaMenos[i,r]#+
            break
        break
    break
            
        





#Lo que sigue despu\'es del inicio






#sys.exit()
