import numpy as np

#--- QR CON GRAM-SCHMIDT ---

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    filas, columnas = A.shape
    Q = np.zeros((filas, filas))
    R = np.zeros((filas, filas))
    if(esCuadarda(A)):
        r = 0 
        for j in range(filas):
            qj = A[:, j].copy()
            for k in range(r):
                R[k, j] = producto_punto(Q[:, k], qj)
                qj = qj - R[k, j] * Q[:, k]
            norma_qj = norma(qj, 2)            
            if norma_qj > tol:
                R[r, j] = norma_qj
                Q[:, r] = qj / norma_qj
                r = r + 1
        Q_sombrero = Q[:, :r] 
        R_sombrero = R[:r, :]
        return Q_sombrero, R_sombrero  
    else:
        return None

# --- QR CON HOUSEHOLDER ---

def QR_con_HH(A, tol=1e-12):
    filas, columnas = A.shape
    Q = np.eye(filas)
    R = A.copy()
    for k in range (columnas):
        x = vector_de_elementos(R, k, filas, k)
        norma_x = norma(x, 2)
        if(x[0] >= 0):
            alpha = -norma_x
        else:
            alpha = norma_x
        h = filas - k
        u = x - (alpha * vector_canonico(h, 0))
        norma_u = norma(u, 2)
        if(norma_u > tol):
            u = u / norma_u
            H = np.eye(h) - 2 * np.outer(u, u) #preguntar si se puede usar
            H_sombrero = H_sombrero(filas, k, H)
            R = multiplicacion_matricial(H_sombrero, R)
            Q = multiplicacion_matricial(Q, H_sombrero) 
    return Q, R
    

#   --- funciones auxiliares ---

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

def producto_punto(A, B):
    producto = 0
    for i in range(len(A)):
        producto += A[i] * B[i]
    return producto

def traspuesta(A):
    filas, columnas = A.shape
    U = np.zeros((columnas, filas))
    for f in range(filas):
        for c in range(columnas):
            U[c][f] = A[f][c]
    return U

def multiplicacion_matricial(A, B):
    filas_A, cols_A = A.shape
    filas_B, cols_B = B.shape
    C = np.zeros((filas_A, cols_B))
    for i in range(filas_A):
        for j in range(cols_B):
            suma = 0.0
            for k in range(cols_A):
                suma += A[i, k] * B[k, j]
            C[i, j] = suma
    return C

def vector_de_elementos(A, k, m, c):
    vector = A[k:m, c]
    return vector


def vector_canonico(l, i):    
  e = np.zeros(l)
  e[i] = 1.0
  return e

def H_sombrero(m, k, H):
    H_sombrero = np.eye(m, dtype=H.dtype) #Identidad de filasxfilas
    index = k
    H_sombrero[index: , index:] = H
    return H_sombrero



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

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)
