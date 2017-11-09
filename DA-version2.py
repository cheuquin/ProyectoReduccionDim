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

Sparse = sp.lil_matrix([m,d])
t = 3/(2*np.log(2)*m)

#voy a comentar muy a la rapida donde me quedan dudas en la
#construccion de este algoritmo

Pessimist = [[0 for j in range(d)] for i in range(n)]

for i in range(n):
    for j in range(d):
        Pessimist[i][j] = np.exp(-s**2*t*np.log(2) + s*np.log(1+epsilon)*(1- espectros[i][j]**4 )/2)

Loss = sum(list(Pessimist))
delta = int(m/s)

B = [[] for x in range(s)]
for q in range(s):
    B[q] = (list(i for i in range(delta*q, delta + delta*q) ))

#Creo que hay que definir bien este arreglo!
PessimistNew = [0 for j in range(m)]
Total = 0



for q in range(s):
    b = [0 for i in range(m)]
    sigma = [0 for i in range(n)]
    for j in range(d):
       for i in np.where(espectros[:,j]!=0)[0]:
           for k in np.where(espectros[i][:j]!=0 )[0]:
               #No se si el r se obtiene en base a la matriz Sparse
               #Que debería tener los p[r][k] iguales a 0 o 1
               #Me parece que es equivalente, pero dejo la duda
               r = B[q][ np.where(Sparse[B[q], k] == 1)[0][0]]
               if (b[r] == 0): b[r] = 1
               #Me parece que el algoritmo sugiere una actualizacion del estimador
               #pesimista, por ello lo use de esta forma, definiendolo para
               #cada coordenada i,r pero no tengo claro si puedo dividir
               #El estimador pesimista general que proponen en el paper
               #por cada uno de los i datos, en r coordenadas
               Pessimist[i][r] = Pessimist[i][r]*2**(espectros[i][k]**2 *espectros[i][j]**2)
       for r in np.where(b[:] == 1)[0]:
           for i in np.where(espectros[:,j])[0]:
              Total += Pessimist[i][r]*(1+epsilon)**(sigma[i]*espectros[i][j])
           #Introduje un nuevo estimador Pesimista, no se si eso es lo que
           #propone el algoritmo, pero no encontré otra manera de usar
           #el epsi[i][j] y combinarlo con un epsi[r], que es su suma pero no
           #precisamente
           PessimistNew[r] = Total
       #Hay que revisar este R, no considera la condicion tal que b[R] = 1
       R = B[q][np.argmax(Pessimist[ B[q] ])]
       for r in B[q]:
           if r == R:
               Sparse[r][j] = 1
           else: Sparse[r][j] = 0
       for i in np.where(espectros[:,j]!=0)[0]:
           sigma[i] = sigma[i] + espectros[i][j]**2
       for r in np.where(b[:] == 1)[0]:
           #Aca dice que hay que actualizar el epsi[i][r], pero no tiene sentido
           #no estamos iterando sobre ningun ciclo que contenga al i
           #Lo usé con el epsi[r], me parecía más lógico
           Pessimist[r] = 1
           b[r] = 0










sys.exit()
