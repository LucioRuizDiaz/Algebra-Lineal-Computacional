import numpy as np

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    filas = len(A)
    Q = np.zeros((filas, filas))
    R = np.zeros((filas, filas))
    if(esCuadarda(A)):
        Q[:,0] = A[:,0] / norma(A[:,0], 2)
        R[0][0] = norma(A[:,0], 2)
        for j in range(2, filas - 1):
            qj = A[:,j]
            for k in range(j - 1):
                for i in range(filas):
                    R[k][j] +=  traspuesta(Q[:,k])[i] * qj[i]
                qj = qj
    else:
        return None
    


def esCuadarda(A):
    filas = len(A)
    columnas = len(A[0])
    return filas == columnas

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

def traspuesta(A):
    filas = len(A)
    columnas = len(A[0]) 
    U = np.zeros((columnas, filas))
    for f in range(filas):
        for c in range(columnas):
            U[c][f] = A[f][c]
    return U

def multiplicacion(A, B):
    filas_o = len(A)
    columnas_i = len(A[0])
    filas_i = len(B)
    columnas_o = len(B[0])
    if(columnas_i == filas_i):
        U = np.zeros((filas_o, columnas_o))
        U