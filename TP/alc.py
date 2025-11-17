import numpy as np

# ------------------------
# FUNCIONES AUXILIARES
# ------------------------
def tamañoMatriz(A):
    filas, columnas = A.shape
    return (filas, columnas)

def esCuadrada(A):
    filas, columnas = A.shape
    return filas == columnas

def multiplicacionMatricial(A, B):
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

def multiplicacionMatrizVector(A, x):
        resultado = np.zeros(x.shape)
        for i in range(A.shape[0]):
            suma = 0.0
            for j in range(A.shape[1]):
                suma += A[i, j] * x[j]
            resultado[i] = suma
        return resultado

def traspuesta(A):
    filas, columnas = A.shape
    U = np.zeros((columnas, filas))
    for f in range(filas):
        for c in range(columnas):
            U[c][f] = A[f][c]
    return U

def es_simetrica(A):
    matriz = np.array(A)
    if not esCuadrada(matriz):
        return False
    return np.allclose(matriz, traspuesta(matriz))

def determinante(A):
    matriz = np.array(A)
    res = 0
    if not esCuadrada(matriz):
        return
    else:
        if len(matriz) == 1:
            return matriz[0][0]
        elif len(matriz) == 2:
            return matriz[0][0] * matriz[1][1] - matriz[0][1] * matriz[1][0]
        else:
            for i in range(len(matriz)):
                submat = np.delete(np.delete(matriz, 0, axis=0), i, axis=1)         #se puede esto???
                res += ((-1) ** i) * matriz[0][i] * determinante(submat)
            return res
        
def es_triang_sup(A):
    matriz = np.array(A)
    if not esCuadrada(matriz):
        return False
    if matriz.shape[0] < 2:
        return True
    else:
        for i in range(1,matriz.shape[0],1):
            for j in range(0,i,1):
                if matriz[i][j] != 0:
                    return False
        return True

def es_defpos(A):
    matriz = np.array(A)
    m, n = matriz.shape
    if not esCuadrada(matriz):
        return False
    if not es_simetrica(matriz): 
        return False
    for k in range(1, n + 1):
        menor_principal = matriz[:k, :k]
        if determinante(menor_principal) <= 0:
            return False
    return True


def producto_punto(A, B):
    producto = 0
    for i in range(len(A)):
        producto += A[i] * B[i]
    return producto

    return C

def vector_de_elementos(A, k, m, c):
    vector = A[k:m, c]
    return vector


def vector_canonico(l, i):    
  e = np.zeros(l)
  e[i] = 1.0
  return e

def construir_H_sombrero(m, k, H):
    H_sombrero = np.eye(m, dtype=H.dtype) #Identidad de filasxfilas
    index = k
    H_sombrero[index: , index:] = H
    return H_sombrero

def simetrica(A, n, tol):
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i, j] - A[j, i]) > tol:
                return False
    return True


def diagonalHastaN(A,n):
    matriz = np.array(A)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = matriz[i, i]
    return D

def suma_elem_fuera_diagonal(A, n): #Suma el valor absoluto de los elementos fuera de la diagonal
    suma = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                suma += abs(A[i, j])
    return suma



# ------------------------
# LABO 01
# ------------------------

def error(x,y):
     x = np.float64(x)
     y = np.float64(y)
     return abs(x-y)
#
def error_relativo(x,y):
     x = np.float64(x)
     y = np.float64(y)
     if x != 0:
          return abs(x-y)/abs(x)
     else:
          return "infinito"

def matricesIguales(A,B, tol):
    i = 0
    j = 0
    if tamañoMatriz(A) != tamañoMatriz(B):
        return False
    else:
        if tamañoMatriz(A) == (0,0): 
            return True
        else:
            while i < len(A):
                while j < len(A[0]):
                    if abs(A[i,j] - B[i,j]) > tol: return False
                    else:
                        j = j + 1
                j = 0   # reinicia columnas para la siguiente fila
                i = i + 1
        return True



# ------------------------
# LABO 02
# ------------------------

def rota(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def escala(s):
    matriz = np.zeros((len(s),len(s)))
    i = 0
    while i < len(s):
        matriz[i,i] = s[i]
        i = i + 1
    return matriz

def rota_y_escala(theta,s):
    return multiplicacionMatricial(escala(s), rota(theta))

def afin(theta,s,b):
    R = rota(theta)
    E = escala(s)
    M = multiplicacionMatricial(E, R)
    matriz = np.zeros((3,3))
    matriz[0,2] = b[0]
    matriz[1,2] = b[1]
    matriz[2,2] = 1
    for i in range(0,2,1):
        for j in range(0,2,1):
            matriz[i,j] = M[i,j]
    return matriz

def trans_afin(v,theta,s,b):
    m = afin(theta,s,b)
    vector = np.array([ v[0], v[1], 1 ])
    matriz = multiplicacionMatrizVector(m, vector)
    res = np.array([(matriz[0]),(matriz[1])])
    return res

# ------------------------
# LABO 03
# ------------------------

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


# ------------------------
# LABO 04
# ------------------------

def calculaLU(A):
    cant_op = 0
    m, n = A.shape
    Ac = A.copy()
    matriz = np.array(Ac)
    identidad = np.eye(m,n)    
    L = identidad
    if not esCuadrada(A):
        return None, None, 0
    
    for k in range(1, n + 1):
        menor_principal = matriz[:k, :k]
        if determinante(menor_principal) == 0:
            return None, None, 0
        
    if es_triang_sup(matriz):
        U = matriz
    else:
        for i in range(m):
            for j in range(i+1, m):
                mult = matriz[j, i] / matriz[i, i]
                matriz[j, i:] = matriz[j, i:] - ( matriz[i, i:] * mult )
                L[j, i] = mult
                cant_op += (2 * (m - i - 1)) + 1
        U = matriz
    return L, U, cant_op


def res_tri(L, b, inferior):
    matrizL = np.array(L)
    matrizb = np.array(b)
    n = matrizb.shape[0]
    if not esCuadrada(L):
        return 'L no es cuadrada o no coincide el tamaño con b'
    X = np.zeros(n)
    if inferior == False:
        for i in range(n - 1, -1, -1):
            suma = 0.0
            for j in range(i + 1, n):
                suma += matrizL[i, j] * X[j]
            X[i] = (matrizb[i] - suma) / matrizL[i, i]
    else: 
        for i in range(n):
            suma = 0.0
            for j in range(i):
                suma += matrizL[i, j] * X[j]
            X[i] = (matrizb[i] - suma) / matrizL[i, i]  
    return X



def inversa(A):
    matriz = np.array(A)
    n = matriz.shape[0]
    if not esCuadrada(A):
        return None
    if determinante(A) == 0:
        return None
    L, U, nops = calculaLU(matriz) 
    identidad = np.eye(n) 
    A_inversa = np.zeros((n, n))
    for j in range(n):
        e_j = identidad[:, j] 
        y_j = res_tri(L, e_j, inferior=True) 
        x_j = res_tri(U, y_j, inferior=False) 
        A_inversa[:, j] = x_j
    return A_inversa



def calculaLDV(A):
    matriz = np.array(A, dtype=float)
    n = matriz.shape[0]
    if not esCuadrada(A):
        return None
    L, U, nops = calculaLU(matriz)
    if L is None or U is None:
        return None
    D_vector = np.zeros(n)
    for i in range(n):
        D_vector[i] = U[i, i]
    for d in D_vector:
        if d == 0:
            return None
    D_matriz = np.zeros((n, n))
    for i in range(n):
        D_matriz[i, i] = D_vector[i]
    V = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            V[i, j] = U[i, j] / D_vector[i]
            nops += 1
    return L, D_matriz, V, nops



def esSDP(A, atol=1e-8):
    matriz = np.array(A,dtype=float)
    n = matriz.shape[0]
    if not esCuadrada(A):
        return False
    es_simetrica = True
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            dif = matriz[i, j] - matriz[j, i]
            if abs(dif)>atol:
                es_simetrica = False
            j += 1
        i += 1
    if not es_simetrica:
        return False
    chequeo = calculaLDV(matriz)
    if chequeo == None: 
        return False
    else:
        L, D_matriz, V, nops = chequeo
    D_vector = np.zeros(n)
    i = 0
    while i < n:
        D_vector[i] = D_matriz[i, i]
        i += 1
    es_definida_positiva = True
    i = 0
    while i < n:
        if D_vector[i] <= atol:
            es_definida_positiva = False
        i += 1
    if not es_definida_positiva:
        return False
    return True

def calculaCholesky(A, atol=1e-10):
    if not esSDP(A, atol):
        return None
    res = calculaLDV(A)
    if res is None:
        return None 
    L, D_matriz, V, nops = res
    n = A.shape[0]
    R = np.zeros((n, n))
    # Al multiplicar una matriz L por una diagonal D^(1/2) a derecha,
    # el efecto es multiplicar cada COLUMNA j de L por el elemento raiz(Djj).
    for j in range(n):
        valorDiagonal = D_matriz[j, j]
        if valorDiagonal < 0: 
            return None 
        raizD = np.sqrt(valorDiagonal)
        # Como L es triangular inferior solo iteramos desde i=j hasta n
        for i in range(j, n):
            R[i, j] = L[i, j] * raizD
    return R

# ------------------------
# LABO 05
# ------------------------


def QR_con_GS(A,tol,retorna_nops=False):
    filas, columnas = A.shape
    Q = np.zeros((filas, columnas))
    R = np.zeros((columnas, columnas))
    r = 0 
    for j in range(columnas):

        # --- BARRA DE PROGRESO ---
        # Imprime el progreso y vuelve al inicio de la línea con '\r'
        print(f"  Progreso GS: {j+1}/{columnas}", end="\r")
        # -------------------------
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
    R_sombrero = R[:r, :columnas]
    return Q_sombrero, R_sombrero  

def QR_con_HH(A, tol):
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
            H_sombrero = construir_H_sombrero(filas, k, H)
            R = multiplicacionMatricial(H_sombrero, R)
            Q = multiplicacionMatricial(Q, H_sombrero) 
    Q_sombrero = Q[:, :columnas] # Toma las primeras 'n' columnas -> (m, n)
    R_sombrero = R[:columnas, :] # Toma las primeras 'n' filas -> (n, n)
    return Q_sombrero, R_sombrero
    
    
def calculaQR(A, metodo, tol):
    if(metodo == "RH"):
        Q, R = QR_con_HH(A, tol)
    elif(metodo == "GS"):
        Q, R = QR_con_GS(A, tol)
    else:
        Q, R = None, None
    return Q, R



# ------------------------
# LABO 06
# ------------------------


def metpot2k(A, tol=1e-15, K=1000):
    matriz = np.array(A)
    n = matriz.shape[0]
    v = np.ones(n)
    if norma(v, 2) == 0: 
        return v, 0, 0
    v = v / norma(v, 2)
    for k in range(1, int(K)+1,1):
        v_inicial = v.copy()
        w = multiplicacionMatrizVector(matriz, v)
        autoval = producto_punto(v, w)
        if norma(w, 2) == 0:
             return v, 0, k
        v = w / norma(w, 2)
        if norma(v- v_inicial, 2) < tol:
            return v, autoval, k
    return v, autoval, int(K)



def diagRH(A, tol=1e-15, K=1000):
    matriz = np.array(A)
    n = matriz.shape[0]
    ident = np.eye(n)
    if not esCuadrada(A):
         return None, None
    if simetrica(A, n, tol) == False:
         return None, None
    for i in range(int(K)):
        Q, R = QR_con_HH(A, tol)    # CAMBIAR POR FUNCION QR DE LUCHO !!!
        matriz = multiplicacionMatricial(R, Q)
        ident = multiplicacionMatricial(ident, Q)
        aux = suma_elem_fuera_diagonal(matriz, n)
        if aux < tol:
            D = diagonalHastaN(matriz, n)
            return ident, D

    return ident, diagonalHastaN(matriz, n)   

# ------------------------
# LABO 07
# ------------------------




# ------------------------
# LABO 08
# ------------------------

def svd(A, tol=1e-15, K=1000):

    n, p = A.shape
    
    At = traspuesta(A)
    AtA = multiplicacionMatricial(At, A) 

    # V (p, p) contiene los autovectores (vectores singulares derechos)
    # D (p, p) es una matriz diagonal de autovalores
    V, D , _ = diagRH(AtA, tol, K) 
    
    # Obtener Valores Singulares y ordenar
    autovalores = np.diag(D)
    
    # Ordenar autovalores y autovectores de mayor a menor
    idx_ordenados = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx_ordenados]
    V = V[:, idx_ordenados]
    
    # Los valores singulares son la raíz cuadrada de los autovalores
    S_vector = np.sqrt(np.abs(autovalores)) 
    
    # Calcular U: A = U @ S @ V.T 

    S_inv = np.zeros_like(S_vector)

    umbral = 1e-15 # Puedes usar 'tol'
    
    idx_no_cero = S_vector > umbral
    
    # Solo en esos índices, calcular la inversa
    S_inv[idx_no_cero] = 1.0 / S_vector[idx_no_cero]
    
    # Convertir el vector S_inv a una matriz diagonal
    S_inv_diag = np.diag(S_inv)
    
    # U = (A @ V) @ S_inv_diag
    AV = multiplicacionMatricial(A, V) 
    U = multiplicacionMatricial(AV, S_inv_diag) 

    return U, S_vector, V

# ------------------------
# LABO 07
# ------------------------




# ------------------------
# LABO 08
# ------------------------

def svd(A, tol=1e-15, K=1000):

    n, p = A.shape
    
    At = traspuesta(A)
    AtA = multiplicacionMatricial(At, A) 

    # V (p, p) contiene los autovectores (vectores singulares derechos)
    # D (p, p) es una matriz diagonal de autovalores
    V, D , _ = diagRH(AtA, tol, K) 
    
    # Obtener Valores Singulares y ordenar
    autovalores = np.diag(D)
    
    # Ordenar autovalores y autovectores de mayor a menor
    idx_ordenados = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx_ordenados]
    V = V[:, idx_ordenados]
    
    # Los valores singulares son la raíz cuadrada de los autovalores
    S_vector = np.sqrt(np.abs(autovalores)) 
    
    # Calcular U: A = U @ S @ V.T 

    S_inv = np.zeros_like(S_vector)

    umbral = 1e-15 # Puedes usar 'tol'
    
    idx_no_cero = S_vector > umbral
    
    # Solo en esos índices, calcular la inversa
    S_inv[idx_no_cero] = 1.0 / S_vector[idx_no_cero]
    
    # Convertir el vector S_inv a una matriz diagonal
    S_inv_diag = np.diag(S_inv)
    
    # U = (A @ V) @ S_inv_diag
    AV = multiplicacionMatricial(A, V) 
    U = multiplicacionMatricial(AV, S_inv_diag) 

    return U, S_vector, V