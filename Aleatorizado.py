from sklearn.random_projection import johnson_lindenstrauss_min_dim
import numpy as np
import sys
import pyfits

#Buscar basededatos
#Extraer n y d de ellos
#Estas base de datos en particular cuentan con espectros de 4000 objetos
#Cada uno tiene una dimensionalidad (longitudes de onda) de 1000

wavelength = pyfits.open("wavelengths.fits")[0].data #1000
espectros = pyfits.open("sdss_spectra.fits")[0].data #4000x1000

n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)

epsilon = 0.6

#Necesitamos definir un m, fraccion de d, la dimension a reducir.
m = johnson_lindenstrauss_min_dim(n_samples=n, eps=epsilon)
#El metodo anterior calcula de manera automatica la dimension minima a
#la que se puede reducir, manteniendo las distancias entre los datos
#dentro de un epsilon fijo (eps)

#La mascara sparse y la matriz con distribucion de rademacher son de igual
#cardinalidad mxd, pero la matriz sparse es tal que tiene s valores no nulos
#con s = epsilon*m

s = int(epsilon*m)

#Creando la matriz sparse
#Se puede incluir de manera inmediata la distribucion rademacher
#haciendo que los valores validos de la matriz sparse sean -1 o 1
#de manera equiprobable

Sparse = np.array([[0 for i in range(d)]for j in range(m)])

for columns in range(d):
    validindexes = []
    n_index = 0
    while n_index < s:
        index = np.random.randint(0,m)
        if index not in (validindexes):
            validindexes.append(index)
            n_index += 1
    for k in range(s):
        Sparse.T[columns][validindexes[k]] = np.random.choice([-1,1], p = [.5,.5])/s**(0.5)

DimReducedData = np.dot(Sparse,espectros.T).T
NewWavelength = np.dot(Sparse,wavelength)

#Ahora se podrian graficar los nuevos espectros de dimension reducida
#dada la nueva dimension (rango de wavelength)

#Haciendo esto me surgio una gran duda, ¿Que cosas vamos a comprobar?
#Los nuevos espectros de dimensionalidad reducida deberian corroborar el
#lema de Johnson-Lindenstrauss, cosa que podemos medir y entregar un resultado,
#no obstante, no me queda claro si es valida la idea de la proyeccion dada
#la distribucion de rademacher y la sparsity puesto que se elimina de manera
#aleatoria informacion que puede ser importante como lineas de emision en tal
#o tal longitud de onda, con este metodo incluso aparecerían flujos negativos!

#Comprobando Lema de Johnson-Lindenstrauss







sys.exit()
