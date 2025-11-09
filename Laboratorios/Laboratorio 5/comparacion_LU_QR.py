import numpy as np
import matplotlib as plt

import numpy as np
import matplotlib.pyplot as plt
import math # Para np.inf en la función norma

# --- CONFIGURACIÓN DEL TESTEO ---
N_sizes = [2, 5, 10, 20, 50, 100]
N_matrices = 100
TOLERANCIA_PIVOTE_LU = 1e-12
TOLERANCIA_GS = 1e-12

# ===================================================================
# SECCIÓN 1: Funciones L03 (Normas)
# ===================================================================

def norma(x, p):
    """
    Calcula la norma p de un vector x.
    """
    if(p == 'inf' or p == np.inf):
        norma_val = 0
        for i in x:
            if(np.abs(i) > norma_val):
                norma_val = np.abs(i)
    else:
        suma = 0
        for i in x:
            suma += (np.abs(i))**p
        norma_val = suma ** (1/p)
    return norma_val

def norma_inf_matriz(A):
    """
    Calcula la norma infinito de una MATRIZ A (máxima suma absoluta por filas).
    Esta es la "normaExacta" que probablemente te pedían en L03.
    """
    filas, _ = A.shape
    max_suma_fila = 0
    for i in range(filas):
        suma_fila_actual = 0
        for j in range(filas):
            suma_fila_actual += np.abs(A[i, j])
        
        if suma_fila_actual > max_suma_fila:
            max_suma_fila = suma_fila_actual
    return max_suma_fila

# ===================================================================
# SECCIÓN 2: Funciones L04 (Factorización LU y Solvers)
# ===================================================================

def factorizacion_LU(A):
    """
    Factorización LU (Doolittle) sin pivoteo.
    Devuelve (None, None) si un pivote es muy pequeño.
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for k in range(n):
        # --- Cálculo de U ---
        for j in range(k, n):
            suma = 0.0
            for s in range(k):
                suma += L[k, s] * U[s, j]
            U[k, j] = A[k, j] - suma

        # --- Chequeo de pivote ---
        # Si el pivote es cero (o muy pequeño), no se puede seguir.
        if np.abs(U[k, k]) < TOLERANCIA_PIVOTE_LU:
            return None, None # Falla la factorización

        # --- Cálculo de L ---
        L[k, k] = 1.0 # Diagonal de L es 1
        for i in range(k + 1, n):
            suma = 0.0
            for s in range(k):
                suma += L[i, s] * U[s, k]
            L[i, k] = (A[i, k] - suma) / U[k, k]
            
    return L, U

def sustitucion_adelante(L, b):
    """
    Resuelve Lx = b (L triangular inferior)
    """
    n = len(L)
    y = np.zeros(n)
    for i in range(n):
        suma = 0.0
        for j in range(i):
            suma += L[i, j] * y[j]
        y[i] = (b[i] - suma) / L[i, i]
    return y

def sustitucion_atras(U, y):
    """
    Resuelve Ux = y (U triangular superior)
    """
    n = len(U)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        suma = 0.0
        for j in range(i + 1, n):
            suma += U[i, j] * x[j]
        x[i] = (y[i] - suma) / U[i, i]
    return x

# ===================================================================
# SECCIÓN 3: Funciones L05 (Factorización QR por Gram-Schmidt)
# ===================================================================
def QR_con_GS(A,tol=1e-12,retorna_nops=False):
        filas, columnas = A.shape
        Q = np.zeros((filas, filas))
        R = np.zeros((filas, filas))
        if(esCuadarda(A)):
            R[0][0] = norma(A[:,0], 2)
            if (R[0, 0] > tol):
                Q[:, 0] = A[:, 0] / R[0, 0]
            else:
                Q[:, 0] = 0.0 # Vector nulo si la norma es cero

            for j in range(1, filas):
                qj = A[:,j]
                for k in range(j):
                    R[k][j] = producto_punto(Q[:,k], qj)
                    qj = qj - R[k][j] * Q[:,k]
                R[j][j] = norma(qj, 2)
                if R[j, j] > tol:
                    Q[:, j] = qj / R[j, j]
                else:
                    Q[:, j] = 0.0 
                
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

def producto_punto(A, B):
    producto = 0
    for i in range(len(A)):
        producto += A[i] * B[i]
    return producto


# ===================================================================
# SECCIÓN 4: Funciones del Esquema de Testeo
# ===================================================================

def calcular_inversa_LU(L, U):
    """
    Calcula A_inv resolviendo L U A_inv = I
    """
    n = len(L)
    I = np.eye(n)
    A_inv_LU = np.zeros((n, n))
    
    for i in range(n):
        e_i = I[:, i]
        y = sustitucion_adelante(L, e_i)
        x = sustitucion_atras(U, y)
        A_inv_LU[:, i] = x
        
    return A_inv_LU

def calcular_inversa_QR(Q, R):
    """
    Calcula A_inv resolviendo Q R A_inv = I
    Esto es R A_inv = Q^T I = Q^T
    """
    n = len(Q)
    Qt = Q.T # Usamos NumPy para la traspuesta
    A_inv_QR = np.zeros((n, n))

    for i in range(n):
        y_i = Qt[:, i] # La columna i de Q^T
        x = sustitucion_atras(R, y_i)
        A_inv_QR[:, i] = x
        
    return A_inv_QR

# ===================================================================
# SECCIÓN 5: Bucle Principal de Simulación
# ===================================================================

print("Iniciando simulación...")

# Para guardar todos los errores para los histogramas
resultados_completos = {} 
# Para guardar solo la media para el gráfico de error vs n
errores_medios = {'n': [], 'lu': [], 'qr': []}

for n in N_sizes:
    
    errores_lu_n = []
    errores_qr_n = []
    matrices_generadas = 0
    
    print(f"\n--- Probando para n = {n} ---")
    
    while len(errores_lu_n) < N_matrices:
        
        # 1. Generar matriz A
        # Valores entre -1 y 1
        A = np.random.uniform(-1.0, 1.0, (n, n))
        matrices_generadas += 1
        
        # 2. Calcular LU
        L, U = factorizacion_LU(A)
        
        # Si falla, descartar y volver a 1
        if L is None:
            continue 
            
        # 3. Calcular QR
        Q, R = QR_con_GS(A)
        
        # 4. Calcular inversas
        A_inv_LU = calcular_inversa_LU(L, U)
        A_inv_QR = calcular_inversa_QR(Q, R)
        
        # 5. Calcular errores
        I = np.eye(n)
        
        # Usamos A @ A_inv (matmul de NumPy) para calcular E
        E_LU = A @ A_inv_LU
        E_QR = A @ A_inv_QR
        
        # epsilon = || E - I ||_inf
        eps_LU = norma_inf_matriz(E_LU - I)
        eps_QR = norma_inf_matriz(E_QR - I)
        
        errores_lu_n.append(eps_LU)
        errores_qr_n.append(eps_QR)

    # Fin del while
    
    # Calcular y guardar medias
    media_lu = np.mean(errores_lu_n)
    media_qr = np.mean(errores_qr_n)
    
    errores_medios['n'].append(n)
    errores_medios['lu'].append(media_lu)
    errores_medios['qr'].append(media_qr)
    
    # Guardar resultados para histogramas
    resultados_completos[n] = (errores_lu_n, errores_qr_n)
    
    print(f"  Matrices válidas (LU) encontradas: {N_matrices} (de {matrices_generadas} generadas)")
    print(f"  Error medio LU (eps_LU): {media_lu: .4e}")
    print(f"  Error medio QR (eps_QR): {media_qr: .4e}")


# ===================================================================
# SECCIÓN 6: Graficación (Histogramas)
# ===================================================================

print("\nGenerando histogramas...")

for n in N_sizes:
    err_lu, err_qr = resultados_completos[n]
    
    # Convertir a log10 para mejor visualización
    # Se suma una pequeña cantidad para evitar log10(0)
    log_err_lu = np.log10(np.array(err_lu) + 1e-30)
    log_err_qr = np.log10(np.array(err_qr) + 1e-30)
    
    plt.figure(figsize=(12, 6))
    plt.suptitle(f"Distribución de Errores (n = {n}) - {N_matrices} matrices")
    
    # Histograma para LU
    plt.subplot(1, 2, 1)
    plt.hist(log_err_lu, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Error LU (Media: {np.mean(err_lu):.2e})")
    plt.xlabel("Error $log_{10}(|| A A^{-1}_{LU} - I ||_{\infty})$")
    plt.ylabel("Frecuencia")
    
    # Histograma para QR
    plt.subplot(1, 2, 2)
    plt.hist(log_err_qr, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.title(f"Error QR (Media: {np.mean(err_qr):.2e})")
    plt.xlabel("Error $log_{10}(|| A A^{-1}_{QR} - I ||_{\infty})$")
    plt.ylabel("Frecuencia")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ===================================================================
# SECCIÓN 7: Graficación (Errores Medios vs n)
# ===================================================================

print("Generando gráfico de error medio vs n...")

plt.figure(figsize=(10, 6))
plt.plot(errores_medios['n'], errores_medios['lu'], 'o-', label='Error Medio LU', color='blue')
plt.plot(errores_medios['n'], errores_medios['qr'], 's-', label='Error Medio QR (GS)', color='green')

plt.yscale('log') # Escala logarítmica es crucial para ver la diferencia
plt.xscale('log') # Escala logarítmica para n también ayuda
plt.xlabel('Tamaño de la Matriz (n)')
plt.ylabel('Error Medio $|| A A^{-1} - I ||_{\infty}$')
plt.title('Error Medio de Inversión vs Tamaño de Matriz')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()

print("\nSimulación completada.")