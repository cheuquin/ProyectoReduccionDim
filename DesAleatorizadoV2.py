import numpy as np
import pyfits as pyfits
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

#Para guardar tiempos
EtiquetasTiempo=['Obtencion de datos y parametros de los datos. ']
Tiempos=[tm.time()-Inicio]

Inicio=tm.time()

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

#Para guardar tiempos
EtiquetasTiempo.append('Carga matriz delta. ')
Tiempos.append(tm.time()-Inicio)

Inicio=tm.time()



#Lista del estimador pesimista para cada vector.
PesimistaMas=np.ones((n))
PesimistaMenos=np.ones((n))

#Valores que coinciden con los Theta_r que est\'an en el paper, es uno para cada
#vector de V, por esta raz\'on se guardan como una matriz sparse.

#Se construyen apartir de la matriz Delta, para ahorrar cambios en la estructura en sí
#sin embargo hay que eliminar el 1 cuando se pueda y se deba.
ThetaMas=np.zeros((n,m),dtype=np.float64)
ThetaMenos=np.zeros((n,m),dtype=np.float64)

#Par\'ametro auxiliar.
lambdaaDivididoEns=lambdaa/s

#####Paso 1 de la matriz sigma

#Indices distintos de cero de los vectores de V
#(esto asume que los datos son densos)
NZIndicesVj=range(0,d)

#Todas las componentes de los datos se elevan al cuadrado y se multiplican por
#lambdaaDivididoEns
u_j2=lambdaaDivididoEns*V**2
ComponentesDeThetaMas=u_j2/(1.+2.*u_j2)
ComponentesDeThetaMenos=u_j2/(1.-2.*u_j2)

#Se ven los aportes al estimador pesimista por fila (r) de la matriz Delta.
r=0 #La primera fila es un caso especial, en el Paso 2, vamos a eliminar parte
#de lo que se calcularia, asi que se decidio no calcularlo

NZIndicesDeltar=Delta.getrow(r).nonzero()[1] #Indices distintos de cero de 
Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)[1:]

ThetaMas[:,r]=np.sum(ComponentesDeThetaMas[:,Indices],axis=1)
ThetaMenos[:,r]=np.sum(ComponentesDeThetaMenos[:,Indices],axis=1)

PesimistaMas=np.prod(1.+2.*u_j2[:,Indices],axis=1)
PesimistaMenos=np.prod(1.-2.*u_j2[:,Indices],axis=1)

PesimistaMas=PesimistaMas*(1.-2.*ThetaMas[:,r])
PesimistaMenos=PesimistaMenos*(1.+2.*ThetaMenos[:,r])

for r in range(1,m):
    NZIndicesDeltar=Delta.getrow(r).nonzero()[1]
    Indices=np.intersect1d(NZIndicesVj, NZIndicesDeltar, assume_unique=True)
    
    ThetaMas[:,r]=np.sum(ComponentesDeThetaMas[:,Indices],axis=1)
    ThetaMenos[:,r]=np.sum(ComponentesDeThetaMenos[:,Indices],axis=1)
    
    PesimistaMas=PesimistaMas*np.prod(1.+2.*u_j2[:,Indices],axis=1)
    PesimistaMenos=PesimistaMenos*np.prod(1.-2.*u_j2[:,Indices],axis=1)
    
PesimistaMas=PesimistaMas*np.prod(1.-2.*ThetaMas,axis=1)
PesimistaMenos=PesimistaMenos*np.prod(1.+2.*ThetaMenos,axis=1)

PesimistaMas=np.sqrt(PesimistaMas)**(-1)
PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)

del Indices

EtiquetasTiempo.append('Paso 1. ')
Tiempos.append(tm.time()-Inicio)

Inicio=tm.time()

######Paso 2 de la matriz sigma

#Matriz sigma, en formato sparse que se puede llenar coordenada a coordenada
sigma=Delta.copy().tolil()

nu=np.zeros((n),dtype=np.float64)
NZIndicesVj=range(0,n)

RaizCuadradalambdaaDivididoEns=np.sqrt(lambdaaDivididoEns)

u_j=RaizCuadradalambdaaDivididoEns*V

for r in range(0,m-1):
    NZIndicesDeltar=np.sort(Delta.getrow(r).nonzero()[1])
    contador=0
    nu=np.zeros((n),dtype=np.float64)
    suma=np.zeros(n)
    for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
        #suma=np.zeros(n)
        suma=suma+u_j2[:,j]
        
        PesimistaMasPos=PesimistaMas*np.exp((nu+u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
        PesimistaMenosPos=PesimistaMenos*np.exp((nu+u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
        PesimistaMasNeg=PesimistaMas*np.exp((nu-u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
        PesimistaMenosNeg=PesimistaMenos*np.exp((nu-u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
        
        if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
            sigma[r,j]=-1.
        #Actualizando los Theta y los Pesimista
        PesimistaMas=PesimistaMas**(-2)
        PesimistaMenos=PesimistaMenos**(-2)
        contador=contador+1
        
        nu=nu+u_j[:,j]*sigma[r,j]
        PesimistaMas=PesimistaMas/(1.-2.*ThetaMas[:,r])
        PesimistaMenos=PesimistaMenos/(1.+2.*ThetaMenos[:,r])
        ThetaMas[:,r]=ThetaMas[:,r]-ComponentesDeThetaMas[:,NZIndicesDeltar[contador]]
        ThetaMenos[:,r]=ThetaMenos[:,r]-ComponentesDeThetaMenos[:,NZIndicesDeltar[contador]]
        
        PesimistaMas=PesimistaMas*(1.-2.*ThetaMas[:,r])/(1.+2.*u_j2[:,NZIndicesDeltar[contador]])
        PesimistaMenos=PesimistaMenos*(1.+2.*ThetaMenos[:,r])/(1.-2.*u_j2[:,NZIndicesDeltar[contador]])
        
        PesimistaMas=np.sqrt(PesimistaMas)**(-1)
        PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
    j=NZIndicesDeltar[len(NZIndicesDeltar)-1]
    #suma=np.zeros(n)
    suma=suma+u_j2[:,j]
    ThetaMas[:,r]=0.
    ThetaMenos[:,r]=0.
    PesimistaMasPos=PesimistaMas*np.exp((nu+u_j[:,j])**2-suma)
    PesimistaMenosPos=PesimistaMenos*np.exp(-(nu+u_j[:,j])**2+suma)
    PesimistaMasNeg=PesimistaMas*np.exp((nu-u_j[:,j])**2-suma)
    PesimistaMenosNeg=PesimistaMenos*np.exp(-(nu-u_j[:,j])**2+suma)
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    tempr=np.sort(Delta.getrow(r+1).nonzero()[1])[0]
    
    nu=nu+u_j[:,j]*sigma[r,j]
    PesimistaMas=PesimistaMas/(1.-2.*ThetaMas[:,r+1])
    PesimistaMenos=PesimistaMenos/(1.+2.*ThetaMenos[:,r+1])
    ThetaMas[:,r+1]=ThetaMas[:,r+1]-ComponentesDeThetaMas[:,tempr]
    ThetaMenos[:,r+1]=ThetaMenos[:,r+1]-ComponentesDeThetaMenos[:,tempr]
    
    PesimistaMas=PesimistaMas*(1.-2.*ThetaMas[:,r+1])*(np.exp(nu**2*-suma))**(-2)/(1.+2.*u_j2[:,tempr])
    PesimistaMenos=PesimistaMenos*(1.+2.*ThetaMenos[:,r+1])*(np.exp(-nu**2*+suma))**(-2)/(1.-2.*u_j2[:,tempr])
        
    PesimistaMas=np.sqrt(PesimistaMas)**(-1)
    PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
r=m-1
NZIndicesDeltar=np.sort(Delta.getrow(r).nonzero()[1])
contador=0
nu=np.zeros((n),dtype=np.float64)
suma=np.zeros(n)
for j in NZIndicesDeltar[0:(len(NZIndicesDeltar)-1)]:
    #Buscando sigma
    #suma=np.zeros(n)
    suma=suma+u_j2[:,j]
    PesimistaMasPos=PesimistaMas*np.exp((nu+u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
    PesimistaMenosPos=PesimistaMenos*np.exp((nu+u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
    PesimistaMasNeg=PesimistaMas*np.exp((nu-u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
    PesimistaMenosNeg=PesimistaMenos*np.exp((nu-u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
    if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
        sigma[r,j]=-1.
    #Actualizando los Theta y los Pesimista
    PesimistaMas=PesimistaMas**(-2)
    PesimistaMenos=PesimistaMenos**(-2)
    contador=contador+1
    nu=nu+u_j[:,j]*sigma[r,j]
    PesimistaMas=PesimistaMas/(1.-2.*ThetaMas[:,r])
    PesimistaMenos=PesimistaMenos/(1.+2.*ThetaMenos[:,r])
    ThetaMas[:,r]=ThetaMas[:,r]-ComponentesDeThetaMas[:,NZIndicesDeltar[contador]]
    ThetaMenos[:,r]=ThetaMenos[:,r]-ComponentesDeThetaMenos[:,NZIndicesDeltar[contador]]
    
    PesimistaMas=PesimistaMas*(1.-2.*ThetaMas[:,r])/(1.+2.*u_j2[:,NZIndicesDeltar[contador]])
    PesimistaMenos=PesimistaMenos*(1.+2.*ThetaMenos[:,r])/(1.-2.*u_j2[:,NZIndicesDeltar[contador]])
    
    PesimistaMas=np.sqrt(PesimistaMas)**(-1)
    PesimistaMenos=np.sqrt(PesimistaMenos)**(-1)
j=NZIndicesDeltar[len(NZIndicesDeltar)-1]
#suma=np.zeros(n)
suma=suma+u_j2[:,j]
PesimistaMasPos=PesimistaMas*np.exp((nu+u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
PesimistaMenosPos=PesimistaMenos*np.exp((nu+u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
PesimistaMasNeg=PesimistaMas*np.exp((nu-u_j[:,j])**2*(2*ThetaMas[:,r]/(1.-2.*ThetaMas[:,r])+1)-suma)
PesimistaMenosNeg=PesimistaMenos*np.exp((nu-u_j[:,j])**2*(2*ThetaMenos[:,r]/(1.+2.*ThetaMenos[:,r])-1)+suma)
if np.sum(PesimistaMasPos+PesimistaMenosPos)>np.sum(PesimistaMasNeg+PesimistaMenosNeg):
    sigma[r,j]=-1.

EtiquetasTiempo.append('Paso 2. ')
Tiempos.append(tm.time()-Inicio)

Inicio=tm.time()

Delta.tolil()
Pi=(Delta.multiply(sigma)/np.sqrt(s)).tocsr()

EtiquetasTiempo.append('Paso Final: armar la matriz. ')
Tiempos.append(tm.time()-Inicio)

Inicio=tm.time()


#np.save('PiV1.npy',Pi.toarray())

#Comprobacion de resultados.

ContadorBien=0
ContadorMal=0

normasCuadrado=np.linalg.norm(V, axis=1)**2

UnoMenosEpsilon=1.-epsilon
UnoMasEpsilon=1.+epsilon

Temp=np.linalg.norm(Pi*np.transpose(V),axis=0)

SuperBool=((UnoMenosEpsilon*normasCuadrado)<Temp)
SuperBool2=(Temp<(UnoMasEpsilon*normasCuadrado))
ContadorBien=sum(SuperBool*SuperBool2)
ContadorMal=n-ContadorBien

print ContadorBien
print ContadorMal

ContadorBien=0
ContadorMal=0

for v in range(0,n):
    vector=np.transpose(np.asmatrix(V[v,:]))
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

