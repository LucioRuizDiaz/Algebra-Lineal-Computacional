import numpy as np

def fAv(A, v, k):
    for i in range(k):
        w = multiplicacion(A, v)
        normal = norma(w, 2)
        if normal!= 0:
            w = w/normal
        v = w
    return w
    
    

def multiplicacion(A, v):
    w = np.zeros(len(A))
    for i in range(len(A)):
        for j in range(len(A)):
            w[i] += A[i, j] * v[j]
    return w 

def norma(x, p):
    if(p=='inf'):
        norma = 0
        for i in x:
            if(np.abs(i) > norma):
                norma = np.abs(i)
    else:
        suma = 0
        for i in x:
            suma += (np.abs(i))**p
        norma = suma ** (1/p)
    return norma

def metpot2k(A, tol, K):
    if not (esCuadrada(A)):
        return None
    v = np.random.randn(A.shape[0])
    vi = fAv(A, v, 2)
    e = productoPunto(v, vi)
    k = 0
    while abs(e-1) > tol and k < K:
        v = np.copy(vi)
        vi = fAv(A, vi, 2)
        e = productoPunto(v, vi)
        k += 1
    autovalor = productoPunto(vi, multiplicacion(A, vi))
    e = e-1
    return vi, autovalor, k


def productoPunto(v, w):
    u= 0
    for i in range(len(v)):
        u += v[i]*w[i]
    return u


def esCuadrada(A):
    filas = len(A)
    columnas = len(A[filas - 1])
    return filas == columnas


#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95
print(exitos)


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95
print(exitos)
