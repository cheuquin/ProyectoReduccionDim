from numpy import linalg as la
import numpy as np
import sys
import pyfits
import scipy.sparse as sp


#Importar datos:
#wavelength = pyfits.open("wavelengths.fits")[0].data #1000
espectros = (pyfits.open("sdss_spectra.fits")[0].data)[0:500,0:200] #4000x1000

#Par\'ametros:
n = len(espectros) #numero de datos (espectros)
d = len(espectros[0]) #dimension de los datos (longitudes de onda)

#Normalizaci\'on de los vectores
espectros=espectros/np.transpose(np.tile(np.linalg.norm(espectros, axis=1), (d, 1)))

epsilon = 0.2
resto = int((1/epsilon) - int(np.log(n)/epsilon**2)%(1/epsilon))

m = int(np.log(n)/epsilon**2) + resto
s = int(m*epsilon)

lambdaa= epsilon*s/(2.*(epsilon+2.*s/m))#s**2*np.log(2)

#Carga de la matriz delta.
nombre='maskn500d200e02.csv'
txt = open(nombre, "r")
Delta= np.genfromtxt(nombre,delimiter=',',dtype='float')
txt.close()
Delta=sp.coo_matrix(Delta,shape=(m,d))

#Par\'ametro auxiliar:
lambdaaDivididoEns=lambdaa/s

#Como los datos son densos NO voy a escribir los datos en formato sparse.
V=espectros
del espectros


#Inicio, inicializaci\'on

#Lista del estimador pesimista para cada vector.
PesimistaMas=np.ones((n))
PesimistaMenos=np.ones((n))

#Valores que coinciden con los Theta_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.

#Se construyen apartir de la matriz Delta, para ahorrar cambios en la estructura en sí
#sin embargo hay que eliminar el 1 cuando se pueda y se deba.
ThetaMas=sp.lil_matrix((n,m),dtype='float')
ThetaMenos=sp.lil_matrix((n,m),dtype='float')

for v in range(0,n):
    NZIndicesVj=range(0,d)
    r=0
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
    for j in Indices[1:]:
        temp=lambdaaDivididoEns*V[v,j]**2
        ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp)
        ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
    #Este es un paso extra, para ahorrar en la construcción las matrices Theta parten con un 1.
    #El que se elimina ahora
    ThetaMas[v,r]=ThetaMas[v,r]-1.
    ThetaMenos[v,r]=ThetaMenos[v,r]-1.
    
    PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
    PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])
    
    for r in range(1,m):
        NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
        Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
        for j in Indices:
            temp=lambdaaDivididoEns*V[v,j]**2
            ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp)
            ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
            
            PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
            PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
        ThetaMas[v,r]=ThetaMas[v,r]-1.
        ThetaMenos[v,r]=ThetaMenos[v,r]-1.
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])

PesimistaMas=np.sqrt(PesimistaMas)**(-1)
PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)

#Matriz sigma, en formato sparse que se puede llenar coordenada a coordenada
sigma=sp.lil_matrix(Delta,dtype='float')

#Valores que coinciden con los nu_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.
#Se construyen apartir de la matriz Delta, para ahorrar cambios en la estructura en sí
#sin embargo hay que eliminar el 1 cuando se pueda y se deba.
nu=sp.lil_matrix((n,m),dtype='float')

#Fase 2
suma=np.zeros(n)
for r in range(0,m-1):
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
        #Buscando sigma
        PesimistaMasPos=np.zeros((n))
        PesimistaMenosPos=np.zeros((n))
        PesimistaMasNeg=np.zeros((n))
        PesimistaMenosNeg=np.zeros((n))
        NZIndicesVj=range(0,d)
        for v in NZIndicesVj:
            temp=V[v,j]
            suma[v]=suma[v]+temp**2
            PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
            PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
            PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
            PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
        if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
            sigma[r,j]=-1.
        #Actualizando los Theta y los Pesimista
        PesimistaMas=PesimistaMas**(-2)
        PesimistaMenos=PesimistaMenos**(-2)
        for v in NZIndicesVj:
            nu[v,r]=nu[v,r]+lambdaaDivididoEns*V[v,j]*sigma[r,j]
            PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
            PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
            temp=lambdaaDivididoEns*V[v,j]**2
            ThetaMas[v,r]=ThetaMas[v,r]-temp/(1.+2.*temp)
            ThetaMenos[v,r]=ThetaMenos[v,r]-temp/(1.-2.*temp)
            
            PesimistaMas[v]=PesimistaMas[v]/(1.+2.*temp)
            PesimistaMenos[v]=PesimistaMenos[v]/(1.-2.*temp)
            PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
            PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])
        PesimistaMas=np.sqrt(PesimistaMas)**(-1)
        PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
    j=NZIndicesDeltar[len(NZIndicesDeltar)-1]
    PesimistaMasPos=np.zeros((n))
    PesimistaMenosPos=np.zeros((n))
    PesimistaMasNeg=np.zeros((n))
    PesimistaMenosNeg=np.zeros((n))
    NZIndicesVj=range(0,d)
    for v in NZIndicesVj:
        temp=V[v,j]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+lambdaaDivididoEns*V[v,j]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r+1])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r+1])
        temp=lambdaaDivididoEns*V[v,j]**2
        ThetaMas[v,r+1]=ThetaMas[v,r+1]-temp/(1.+2.*temp)
        ThetaMenos[v,r+1]=ThetaMenos[v,r+1]-temp/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]/(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]/(1.-2.*temp)
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r+1])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r+1])
    PesimistaMas=np.sqrt(PesimistaMas)**(-1)
    PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
r=m-1
NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
    #Buscando sigma
    PesimistaMasPos=np.zeros((n))
    PesimistaMenosPos=np.zeros((n))
    PesimistaMasNeg=np.zeros((n))
    PesimistaMenosNeg=np.zeros((n))
    NZIndicesVj=range(0,d)
    for v in NZIndicesVj:
        temp=V[v,j]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+lambdaaDivididoEns*V[v,j]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
        temp=lambdaaDivididoEns*V[v,j]**2
        ThetaMas[v,r]=ThetaMas[v,r]-temp/(1.+2.*temp)
        ThetaMenos[v,r]=ThetaMenos[v,r]-temp/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]/(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]/(1.-2.*temp)
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])
    PesimistaMas=np.sqrt(PesimistaMas)**(-1)
    PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
j=NZIndicesDeltar[len(NZIndicesDeltar)-1]
PesimistaMasPos=np.zeros((n))
PesimistaMenosPos=np.zeros((n))
PesimistaMasNeg=np.zeros((n))
PesimistaMenosNeg=np.zeros((n))
NZIndicesVj=range(0,d)
for v in NZIndicesVj:
    temp=V[v,j]
    suma[v]=suma[v]+temp**2
    PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
    PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
    PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
    PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
    sigma[r,j]=-1.

Delta.tolil()
Pi=Delta.multiply(sigma)/np.sqrt(s).tocsr()

np.save('Pi.npy',Pi.toarray())

#    return sigma.tocsr()


#sigma()

