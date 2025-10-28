import numpy as np

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    filas = len(A)
    Q = np.zeros((filas, filas))
    R = np.zeros((filas, filas))
    if(esCuadarda(A)):
        Q[:,0] = A[:,0]/ norma(A[:,0], 2)
        R[0][0] = norma(A[:,0], 2)
        for j in range(1, filas):
            qj = A[:,j]
            for k in range(j - 1):
                R[k][j] = multiplicacion(traspuesta(Q[:,k]), qj)
                qj = resta(qj, multiplicacion(R[k][j], Q[:,k]))
            R[j][j] = norma(qj, 2)
            Q[:, j] = qj/ R[j][j]
        return Q, R
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
    m = 0
    for i in range(len(A)):
        m+= A[i] * B[i]
    return m

def resta(A, B):
    filas = len(A)
    columnas = len(A[0])
    R = np.zeros((filas, columnas))
    for f in filas:
        for c in columnas:
            R[f][c] = A[f][c] - B[f][c]
    return R

def division(A, x):
    D = np.zeros((1, len(A)))
    for i in range(len(A)):
        D[i] = A[i] / x
    return D



# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

