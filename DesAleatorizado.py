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
resto = int((1/epsilon) - int(np.log(n)/epsilon**2)%(1/epsilon))

m = int(np.log(n)/epsilon**2) + resto
s = int(m*epsilon)

lambdaa=s/4.
lambdaaDivididoEns=lambdaa/s


#Para la matriz signo.

#Por mientras para tener una matriz delta.
Delta=sp.csr_matrix(espectros[0:m,],shape=(m,d)) #BORRAR DESPUES


#Para cumplir las condiciones del teorema tengo que ponderar todos los vectores.
ponderador=lambdaaDivididoEns*max(np.linalg.norm(espectros, axis=1))**2*2

espectros=espectros/ponderador

#Voy a escribir los datos (V) en formato sparse,
V=sp.csc_matrix(espectros, shape=(n,d))

#Inicio
#def sigma():
#Lista del estimador pesimista para cada vector.
PesimistaMas=np.ones((n))
PesimistaMenos=np.ones((n))

#Matriz sigma, en formato sparse que se puede llenar coordenada a coordenada
sigma=sp.lil_matrix((m,d),dtype='float')

#Valores que coinciden con los Theta_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.
ThetaMas=sp.lil_matrix((n,m),dtype='float')
ThetaMenos=sp.lil_matrix((n,m),dtype='float')

#Valores que coinciden con los nu_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.
nu=sp.lil_matrix((n,m),dtype='float')

for v in range(0,n):
    NZIndicesVj=V.getcol(v).nonzero()[0]
    r=0
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
    for j in Indices[1:]:
        temp=lambdaaDivididoEns*(V.getcol(v).getrow(j).toarray()[0][0])**2
        ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp)
        ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
        
    PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
    PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])
    
    for r in range(1,m):
        NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
        Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
        for j in Indices:
            temp=lambdaaDivididoEns*(V.getcol(v).getrow(j).toarray()[0][0])**2
            ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp)
            ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
            
            PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
            PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
            
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])

PesimistaMas=np.sqrt(PesimistaMas)**(-1)
PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)

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
        NZIndicesVj=V.getcol(v).nonzero()[0]
        for v in NZIndicesVj:
            temp=V.getcol(v).getrow(j).toarray()[0][0]
            suma[v]=suma[v]+temp**2
            PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
            PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
            PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
            PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
        if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
            sigma[r,j]=-1.
        else:
            sigma[r,j]=1.
        #Actualizando los Theta y los Pesimista
        PesimistaMas=PesimistaMas**(-2)
        PesimistaMenos=PesimistaMenos**(-2)
        for v in NZIndicesVj:
            nu[v,r]=nu[v,r]+lambdaaDivididoEns*V.getcol(v).getrow(j).toarray()[0][0]*sigma[r,j]
            PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
            PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
            temp=lambdaaDivididoEns*(V.getcol(v).getrow(j+1).toarray()[0][0])**2
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
    NZIndicesVj=V.getcol(v).nonzero()[0]
    for v in NZIndicesVj:
        temp=V.getcol(v).getrow(j).toarray()[0][0]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    else:
        sigma[r,j]=1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+lambdaaDivididoEns*V.getcol(v).getrow(j).toarray()[0][0]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r+1])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r+1])
        temp=lambdaaDivididoEns*(V.getcol(v).getrow(0).toarray()[0][0])**2
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
    NZIndicesVj=V.getcol(v).nonzero()[0]
    for v in NZIndicesVj:
        temp=V.getcol(v).getrow(j).toarray()[0][0]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    else:
        sigma[r,j]=1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+lambdaaDivididoEns*V.getcol(v).getrow(j).toarray()[0][0]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
        temp=lambdaaDivididoEns*(V.getcol(v).getrow(j+1).toarray()[0][0])**2
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
NZIndicesVj=V.getcol(v).nonzero()[0]
for v in NZIndicesVj:
    temp=V.getcol(v).getrow(j).toarray()[0][0]
    suma[v]=suma[v]+temp**2
    PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]+temp)**2-suma[v])
    PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]+temp)**2+suma[v])
    PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+(nu[v,r]-temp)**2-suma[v])
    PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-(nu[v,r]-temp)**2+suma[v])
if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
    sigma[r,j]=-1.
else:
    sigma[r,j]=1.

sigma=sigma.tocsr()


#    return sigma.tocsr()


#sigma()




#sys.exit()
