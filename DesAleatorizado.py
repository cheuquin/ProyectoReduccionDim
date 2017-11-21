import numpy as np
import pyfits
import scipy.sparse as sp
import time as tm

Inicio=tm.time()

#####Importar datos.

#Para la matriz chica
V = (pyfits.open("sdss_spectra.fits")[0].data)[0:500,0:200]
'''
#Para la matriz grande
V = (pyfits.open("sdss_spectra.fits")[0].data)
'''
#####Par\'ametros de los datos:
n = len(V) #numero de datos
d = len(V[0]) #dimension de los datos

#Normalizaci\'on de los vectores.
V=V/np.transpose(np.tile(np.linalg.norm(V, axis=1), (d, 1)))

#Para guardar tiempos
EtiquetasTiempo=['Obtencion de datos y parametros de los datos. ']
Tiempos=[tm.time()-Inicio]


#####Par\'ametros a elecci\'on:

#Matriz chica
epsilon = 0.2
'''
#Matriz grande
epsilon=0.1
'''
#M\'as parametros
resto = int((1/epsilon) - int(np.log(n)/epsilon**2)%(1/epsilon))

m = int(np.log(n)/epsilon**2) + resto
s = float(int(m*epsilon))

lambdaa= epsilon*s/(2.*(epsilon+2.*s/float(m)))


####Carga de la matriz Delta.


#Matriz chica
nombre='maskn500d200e02.csv'
'''

#Matriz grande
nombre='maskn4000d1000e01.csv'
'''

txt = open(nombre, "r")
Delta= np.genfromtxt(nombre,delimiter=',',dtype='float')
txt.close()
Delta=sp.coo_matrix(Delta,shape=(m,d)) #Se guarda en formato sparse

#Lista del estimador pesimista para cada vector.
PesimistaMas=np.ones((n))
PesimistaMenos=np.ones((n))

#Valores que coinciden con los Theta_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.

#Se construyen apartir de la matriz Delta, para ahorrar cambios en la estructura en sí
#sin embargo hay que eliminar el 1 cuando se pueda y se deba.
ThetaMas=sp.lil_matrix((n,m),dtype=np.float64)
ThetaMenos=sp.lil_matrix((n,m),dtype=np.float64)

#Par\'ametro auxiliar.
lambdaaDivididoEns=lambdaa/s

#####Paso 1 de la matriz sigma

#Indices distintos de cero de los vectores de V
#(esto asume que los datos son densos)
NZIndicesVj=range(0,d)

#Se ven los aportes al estimador pesimista por fila (r) de la matriz Delta.
r=0 #La primera fila es un caso especial, en el Paso 2, vamos 
NZIndicesDeltar=Delta.getrow(r).nonzero()[1] #Indices distintos de cero de 
Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)

for v in range(0,n):
    for j in Indices[1:]:
        temp=lambdaaDivididoEns*V[v,j]**2 #u_j^2 del teorema
        ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp) #
        ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
    #Este es un paso extra, para ahorrar en la construcción las matrices Theta parten con un 1.
    #El que se elimina ahora
    
    PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
    PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])
    
for r in range(1,m):
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
    for v in range(0,n):
        for j in Indices:
            temp=lambdaaDivididoEns*V[v,j]**2
            ThetaMas[v,r]=ThetaMas[v,r]+temp/(1.+2.*temp)
            ThetaMenos[v,r]=ThetaMenos[v,r]+temp/(1.-2.*temp)
            
            PesimistaMas[v]=PesimistaMas[v]*(1.+2.*temp)
            PesimistaMenos[v]=PesimistaMenos[v]*(1.-2.*temp)
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r])

PesimistaMas=np.sqrt(PesimistaMas)**(-1)
PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)

#Matriz sigma, en formato sparse que se puede llenar coordenada a coordenada
sigma=Delta.copy().tolil()

#Valores que coinciden con los nu_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.
#Se construyen apartir de la matriz Delta, para ahorrar cambios en la estructura en sí
#sin embargo hay que eliminar el 1 cuando se pueda y se deba.
nu=sp.lil_matrix((n,m),dtype=np.float64)

NZIndicesVj=range(0,n)
RaizCuadradalambdaaDivididoEns=np.sqrt(lambdaaDivididoEns)
#Fase 2
for r in range(0,m-1):
    NZIndicesDeltar=np.sort(Delta.getrow(r).nonzero()[1])
    contador=0
    for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
        #Buscando sigma
        PesimistaMasPos=np.zeros((n))
        PesimistaMenosPos=np.zeros((n))
        
        PesimistaMasNeg=np.zeros((n))
        PesimistaMenosNeg=np.zeros((n))
        
        suma=np.zeros(n)
        for v in NZIndicesVj:
            temp=RaizCuadradalambdaaDivididoEns*V[v,j]
            suma[v]=suma[v]+temp**2
            PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
            PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
            PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
            PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
        if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
            sigma[r,j]=-1.
        #Actualizando los Theta y los Pesimista
        PesimistaMas=PesimistaMas**(-2)
        PesimistaMenos=PesimistaMenos**(-2)
        contador=contador+1
        for v in NZIndicesVj:
            nu[v,r]=nu[v,r]+RaizCuadradalambdaaDivididoEns*V[v,j]*sigma[r,j]
            PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
            PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
            temp=lambdaaDivididoEns*V[v,NZIndicesDeltar[contador]]**2
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
    suma=np.zeros(n)
    for v in NZIndicesVj:
        ThetaMas[v,r]=0.
        ThetaMenos[v,r]=0.
        temp=RaizCuadradalambdaaDivididoEns*V[v,j]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    tempr=Delta.getrow(r+1).nonzero()[1][0]
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+RaizCuadradalambdaaDivididoEns*V[v,j]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r+1])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r+1])
        temp=lambdaaDivididoEns*V[v,tempr]**2
        ThetaMas[v,r+1]=ThetaMas[v,r+1]-temp/(1.+2.*temp)
        ThetaMenos[v,r+1]=ThetaMenos[v,r+1]-temp/(1.-2.*temp)
        
        
        PesimistaMas[v]=PesimistaMas[v]/(1.+2.*temp)
        PesimistaMenos[v]=PesimistaMenos[v]/(1.-2.*temp)
        
        PesimistaMas[v]=PesimistaMas[v]*(np.exp(nu[v,r]**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v]))**(-2)
        PesimistaMenos[v]=PesimistaMenos[v]*(np.exp(nu[v,r]**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v]))**(-2)
        
        '''
        PesimistaMas[v]=PesimistaMas[v]*(1.-2.*ThetaMas[v,r+1])
        PesimistaMenos[v]=PesimistaMenos[v]*(1.+2.*ThetaMenos[v,r+1])'''
    PesimistaMas=np.sqrt(PesimistaMas)**(-1)
    PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
r=m-1
NZIndicesDeltar=np.sort(Delta.getrow(r).nonzero()[1])
contador=0
for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
    #Buscando sigma
    PesimistaMasPos=np.zeros((n))
    PesimistaMenosPos=np.zeros((n))
    PesimistaMasNeg=np.zeros((n))
    PesimistaMenosNeg=np.zeros((n))
    suma=np.zeros(n)
    for v in NZIndicesVj:
        temp=RaizCuadradalambdaaDivididoEns*V[v,j]
        suma[v]=suma[v]+temp**2
        PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
        PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
        PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
        PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    contador=contador+1
    for v in NZIndicesVj:
        nu[v,r]=nu[v,r]+RaizCuadradalambdaaDivididoEns*V[v,j]*sigma[r,j]
        PesimistaMas[v]=PesimistaMas[v]/(1.-2.*ThetaMas[v,r])
        PesimistaMenos[v]=PesimistaMenos[v]/(1.+2.*ThetaMenos[v,r])
        temp=lambdaaDivididoEns*V[v,NZIndicesDeltar[contador]]**2
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
suma=np.zeros(n)
for v in NZIndicesVj:
    temp=RaizCuadradalambdaaDivididoEns*V[v,j]
    suma[v]=suma[v]+temp**2
    PesimistaMasPos[v]=PesimistaMas[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
    PesimistaMenosPos[v]=PesimistaMenos[v]*np.exp((nu[v,r]+temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
    PesimistaMasNeg[v]=PesimistaMas[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMas[v,r]/(1.-2.*ThetaMas[v,r])+1)-suma[v])
    PesimistaMenosNeg[v]=PesimistaMenos[v]*np.exp((nu[v,r]-temp)**2*(2*ThetaMenos[v,r]/(1.+2.*ThetaMenos[v,r])-1)+suma[v])
if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
    sigma[r,j]=-1.

Delta.tolil()
Pi=(Delta.multiply(sigma)/np.sqrt(s)).tocsr()

#np.save('PiV1.npy',Pi.toarray())

#Comprobacion de resultados.

ContadorBien=0
ContadorMal=0

#Primero para los V normalizados

normasCuadrado=np.linalg.norm(V, axis=1)**2

UnoMenosEpsilon=1.-epsilon
UnoMasEpsilon=1.+epsilon

for v in range(0,n):
    vector=np.transpose(np.asmatrix(V[v,]))
    temp=np.linalg.norm(Pi*vector)**2
    print temp
    if UnoMenosEpsilon*normasCuadrado[v]<temp:
        if temp<UnoMasEpsilon*normasCuadrado[v]:
            ContadorBien=ContadorBien+1
        else:
            ContadorMal=ContadorMal+1
    else:
        ContadorMal=ContadorMal+1

print ContadorBien
print ContadorMal

'''
#Para la matriz chica
V = (pyfits.open("sdss_spectra.fits")[0].data)[0:500,0:200]
'''

'''
#Para la matriz grande
V = (pyfits.open("sdss_spectra.fits")[0].data)
'''

'''
normasCuadrado=np.linalg.norm(V, axis=1)**2
for v in range(0,n):
    vector=np.transpose(np.asmatrix(V[v,]))
    temp=np.linalg.norm(Pi*vector)**2
    if UnoMenosEpsilon*normasCuadrado[v]<temp:
        if temp<UnoMasEpsilon*normasCuadrado[v]:
            ContadorBien=ContadorBien+1
        else:
            ContadorMal=ContadorMal+1
    else:
        ContadorMal=ContadorMal+1

print ContadorBien
print ContadorMal
'''
