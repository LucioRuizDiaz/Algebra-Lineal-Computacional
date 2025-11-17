import numpy as np
import os
import pandas as pd
import alc as alc

# --------------------------------------
# ITEM 1 - Lectura de Datos
# --------------------------------------

def cargarDataset(ruta_base_dataset):
    """
    Carga los embeddings pre-calculados para los conjuntos
    de entrenamiento y validacion, y genera sus respectivas etiquetas.
    
    Parámetros:

    ruta_base_dataset : str
       La ruta raíz (principal) donde se encuentran las carpetas 'train' y 'val'.

    Retorna:
    tuple
       Una tupla de 4 arrays de NumPy en el siguiente orden: 
       (X_entrenamiento, Y_entrenamiento, X_validacion, Y_validacion)
    """
    
    NOMBRE_ARCHIVO_EMBEDDINGS = 'efficientnet_b3_embeddings.npy'

    # Construir las rutas 
    ruta_entrenamiento = os.path.join(ruta_base_dataset, 'train')
    ruta_validacion = os.path.join(ruta_base_dataset, 'val')
    
    ruta_train_gatos = os.path.join(ruta_entrenamiento, 'cats', NOMBRE_ARCHIVO_EMBEDDINGS)
    ruta_train_perros = os.path.join(ruta_entrenamiento, 'dogs', NOMBRE_ARCHIVO_EMBEDDINGS)
    ruta_val_gatos = os.path.join(ruta_validacion, 'cats', NOMBRE_ARCHIVO_EMBEDDINGS)
    ruta_val_perros = os.path.join(ruta_validacion, 'dogs', NOMBRE_ARCHIVO_EMBEDDINGS)

    # Cargar los features del conjunto de entrenamiento
    X_train_gatos = np.load(ruta_train_gatos) 
    X_train_perros = np.load(ruta_train_perros)
    
    # Unir los datos de gatos y perros en una sola matriz.
    X_entrenamiento = np.hstack((X_train_gatos, X_train_perros))

    # Cargar los features del conjunto de validacion
    X_val_gatos = np.load(ruta_val_gatos)
    X_val_perros = np.load(ruta_val_perros)
    
    # Unir los datos de gatos y perros en una sola matriz.
    X_validacion = np.hstack((X_val_gatos, X_val_perros))

    # Definir etiquetas 
    etiqueta_gato = np.array([[1], [0]]) # GATO
    etiqueta_perro = np.array([[0], [1]]) # PERRO

    # ENTRENAMIENTO
    
    # Contar cuantas columnas hay para cada clase en entrenamiento
    num_gatos_entrenamiento = X_train_gatos.shape[1]
    num_perros_entrenamiento = X_train_perros.shape[1]
    
    # Usar np.tile para repetir el vector de etiqueta para cada muestra de esa clase
    Y_train_gatos = np.tile(etiqueta_gato, num_gatos_entrenamiento)
    Y_train_perros = np.tile(etiqueta_perro, num_perros_entrenamiento)

    # Apilar horizontalmente las etiquetas en el mismo orden 
    Y_entrenamiento = np.hstack((Y_train_gatos, Y_train_perros))

    # VALIDACION

    # Contar cuantas columnas hay para cada clase en validacion
    num_gatos_validacion = X_val_gatos.shape[1]
    num_perros_validacion = X_val_perros.shape[1]
    
    # Repetir las etiquetas para el conjunto de validacion
    Y_val_gatos = np.tile(etiqueta_gato, num_gatos_validacion)
    Y_val_perros = np.tile(etiqueta_perro, num_perros_validacion)

    # Apilar horizontalmente las etiquetas de validación
    Y_validacion = np.hstack((Y_val_gatos, Y_val_perros))

    # Imprimir las dimensiones finales de las matrices
    print(f"Dimensiones X Entrenamiento (Xt): {X_entrenamiento.shape}")
    print(f"Dimensiones Y Entrenamiento (Yt): {Y_entrenamiento.shape}")
    print(f"Dimensiones X Validación (Xv): {X_validacion.shape}")
    print(f"Dimensiones Y Validación (Yv): {Y_validacion.shape}")
  
    return X_entrenamiento, Y_entrenamiento, X_validacion, Y_validacion 

# --------------------------------------
# ITEM 2 - Ecuaciones Normales
# --------------------------------------

def cholesky(matriz_A):
    """
    Calcula la descomposición de Cholesky (A = L * L.T) de una matriz 
    simétrica y definida positiva.

    Parámetros:
    matriz_A : np.ndarray
        Una matriz cuadrada, simétrica y definida positiva.

    Retorna:
    np.ndarray
        La matriz triangular inferior L tal que L @ L.T = matriz_A.
    """
    
    # Obtener la dimensión de la matriz 
    n = matriz_A.shape[0]
    
    # Crear una matriz de ceros para almacenar L
    matriz_L = np.zeros((n, n), dtype=float)

    # Iterar sobre las filas 
    for i in range(n):
        # Iterar sobre las columnas 
        for j in range(i + 1):
            
            # Inicializar la suma de productos 
            suma_parcial = 0
            for k in range(j):
                suma_parcial += matriz_L[i, k] * matriz_L[j, k]

            # Elementos de la diagonal
            if i == j:
                valor_diagonal = matriz_A[i, i] - suma_parcial
                
                # Manejar posibles matrices no def-positivas
                if valor_diagonal < 0:
                    raise ValueError("La matriz no es definida positiva. No se puede calcular Cholesky.")
                
                matriz_L[i, i] = np.sqrt(valor_diagonal)

            # Elementos fuera de la diagonal
            else:
                if matriz_L[j, j] == 0:
                    raise ValueError("División por cero, L[j, j] es cero.")
                    
                matriz_L[i, j] = (matriz_A[i, j] - suma_parcial) / matriz_L[j, j]
                
    return matriz_L

def pinvEcuacionesNormales(X, Y):
    """
    Resuelve el sistema de ecuaciones lineales Y = W @ X para la matriz W,
    utilizando la pseudoinversa calculada por Ecuaciones Normales.
    
    Elige automáticamente el algoritmo basado en las dimensiones de la matriz de datos X (n, p).

    Parámetros:
    X : np.ndarray
        Matriz de datos (features), con dimensiones (n, p).
    
    Y : np.ndarray
        Matriz de etiquetas (objetivos), con dimensiones (m, p).

    Retorna:

    W : np.ndarray
        La matriz de pesos W (m, n) (ej: (2, 1536)) que mejor 
        resuelve el sistema.
    """
    
    # Verificación de Dimensiones 
    
    # m = número de clases 
    # p_Y = número de muestras en Y
    m, p_Y = Y.shape
    
    # n = número de características
    # p_X = número de muestras en X
    n, p_X = X.shape

    # Verificación de consistencia: ambas matrices deben tener el mismo numero de muestras
    if p_Y != p_X:
        raise ValueError("Error en la consistencia")

    p = p_X


    # Selección del Algoritmo 

    ## CASO A
    # Resuelve A @ U = B donde A = X.T @ X (p, p)
    # La solución final es W = Y @ U
    if n > p :
        Xt = X.T 

        A = alc.multiplicacionMatricial(Xt, X)
        # B es el lado derecho del sistema 
        B = Xt

        # Calcular la descomposición L @ L.T = A
        L = cholesky(A)

        # Resolver L @ L.T @ U = B en dos pasos:
        
        # Resolver L @ Z = B para Z (Sustitución hacia adelante)
        Z = np.zeros_like(B)
        # Iteramos sobre cada columna de B que tiene 'n' columnas
        for i in range(B.shape[1]): 
            # res_tri resuelve un sistema para un solo vector (columna)
            Z[:, i] = alc.res_tri(L, B[:, i], inferior=True)

        # Resolver L.T @ U = Z para U (Sustitución hacia atrás)
        U = np.zeros_like(Z)
        L_transpuesta = L.T
        # Iteramos sobre cada columna de Z que tiene 'n' columnas
        for i in range(Z.shape[1]): 
            U[:, i] = alc.res_tri(L_transpuesta, Z[:, i], inferior=False)
        
        # Calcular W = Yt @ U
        W = alc.multiplicacionMatricial(Y, U)
        
        return W

    ## CASO B: 
    # Resuelve W = (Y @ X.T) @ (X @ X.T)^-1
    # Se calcula V = X.T @ (X @ X.T)^-1 (que es la pseudoinversa)
    # Y luego W = Y @ V
    elif n < p:

        Xt = X.T # (p, n)

        # A es la matriz pequeña y definida positiva (n, n)
        A = alc.multiplicacionMatricial(X, Xt) 
        # B es X transpuesta (p, n)
        B = Xt 
        
        # Calcular la descomposición L @ L.T = A
        L = cholesky(A) 
            
        # El algoritmo resuelve (L @ L.T) @ V.T = X
        Bt = B.T 
        
        # Resolver L @ L.T @ V.T = X en dos pasos:
        
        # Resolver L @ Z = X (Sustitución hacia adelante)
        Z = np.zeros_like(Bt) 
        for i in range(Bt.shape[1]): 
            Z[:, i] = alc.res_tri(L, Bt[:, i], inferior=True)
        
        # Resolver L.T @ V.T = Z (Sustitución hacia atrás)
        Vt = np.zeros_like(Z) 
        for i in range(Z.shape[1]): 
            Vt[:, i] = alc.res_tri(L.T, Z[:, i], inferior=False)
        
        # Obtener V (la pseudoinversa de X)
        V = Vt.T 
        
        # Calcular W = Y @ V
        # W (m, n) = Y (m, p) @ V (p, n)
        W = alc.multiplicacionMatricial(Y, V) 
        return W
    
    ## CASO C: 
    # Solución directa si X es invertible
    elif n == p:
        # Calcula la inversa de X (n, n) o (p, p)
        X_inv = alc.inversa(X) 
        
        # W (m, n) = Y (m, p) @ X_inv (p, n)
        W = alc.multiplicacionMatricial(Y, X_inv)
        
        return W

# --------------------------------------
# ITEM 3 - Descomposición en Valores Singulares
# --------------------------------------

def svd(A, tol=1e-15, K=1000):
    """
    Calcula la Descomposición en Valores Singulares (SVD) de una matriz A.
    
    Implementa el cálculo A = U @ S @ V.T (versión reducida)
    utilizando el método de los autovalores.
    
    Parámetros:
    ----------
    A : np.ndarray
        La matriz de datos (n, p) a descomponer.
    
    tol : float, opcional
        Tolerancia para la convergencia en la diagonalización (diagRH).
        
    K : int, opcional
        Número máximo de iteraciones para la diagonalización (diagRH).

    Retorna:
    -------
    tuple
        Una tupla (U, S, V) donde:
        - U (np.ndarray): Matriz (n, p) de vectores singulares izquierdos.
        - S (np.ndarray): Vector (p,) de valores singulares.
        - V (np.ndarray): Matriz (p, p) de vectores singulares derechos.
    """
    n, p = A.shape
    
    # Calcular A.T @ A 
    At = A.T
    AtA = alc.multiplicacionMatricial(At, A) 
    
    # Diagonalizar A.T @ A para obtener V y Autovalores
    # Usamos diagRH para resolver AtA = V @ D @ V.T

    # V (p, p) contiene los autovectores (vectores singulares derechos)
    # D (p, p) es una matriz diagonal de autovalores
    V, D , _ = alc.diagRH(AtA, tol, K) 
    
    # Obtener Valores Singulares (S) y ordenar
    autovalores = np.diag(D)
    
    # Ordenar autovalores y autovectores de mayor a menor
    idx_ordenados = np.argsort(autovalores)[::-1]
    autovalores = autovalores[idx_ordenados]
    V = V[:, idx_ordenados] # Reordenar las columnas de V
    
    # Los valores singulares son la raíz cuadrada de los autovalores
    S_vector = np.sqrt(np.abs(autovalores)) 
    
    # Calcular U usando la relación A = U @ S @ V.T 

    # Crear un vector de ceros para S_inv
    S_inv = np.zeros_like(S_vector)
    
    # Definir un umbral para considerar un valor como "no cero"
    umbral = 1e-15 # Puedes usar 'tol'
    
    # Encontrar los índices donde S_vector es mayor que el umbral
    idx_no_cero = S_vector > umbral
    
    # Solo en esos índices, calcular la inversa
    S_inv[idx_no_cero] = 1.0 / S_vector[idx_no_cero]
    
    # Convertir el vector S_inv a una matriz diagonal
    S_inv_diag = np.diag(S_inv)
    
    # U = (A @ V) @ S_inv_diag
    AV = alc.multiplicacionMatricial(A, V) 
    U = alc.multiplicacionMatricial(AV, S_inv_diag) 

    return U, S_vector, V

def pinvSVD(U, S_vector, V, Yt):
    """
    Calcula la matriz de pesos W usando la pseudoinversa de SVD.
    
    Implementa el Algoritmo 2: W = Y @ V @ S_plus @ U.T
    
    Parámetros:
    U : np.ndarray
        Matriz U (n, p) 
    
    S_vector : np.ndarray
        Vector (p,) de valores singulares 
        
    V : np.ndarray
        Matriz V (p, p) .
    
    Yt : np.ndarray
        Matriz de etiquetas (m, p).

    Retorna:
    W : np.ndarray
        La matriz de pesos W (m, n).
    """

    m, p_Y = Yt.shape
    n, p_U = U.shape
    
    # Construir S_plus 

    # Crear un vector de ceros para S_plus
    S_plus_vec = np.zeros_like(S_vector)
    
    # Definir un umbral
    umbral = 1e-15 
    
    # Encontrar los índices no cero
    idx_no_cero = S_vector > umbral
    
    # Calcular 1/s_i solo para esos índices
    S_plus_vec[idx_no_cero] = 1.0 / S_vector[idx_no_cero]
    
    # Convertir a matriz diagona
    S_plus = np.diag(S_plus_vec)
    
    # Calcular W = Yt @ V @ S_plus @ U.T 
    
    # Tmp1 = S_plus @ U.T
    Ut = U.T 
    Tmp1 = alc.multiplicacionMatricial(S_plus, Ut) 
    
    # Tmp2 = V @ Tmp1 
    # Tmp2 = V @ S_plus @ U.T  
    Tmp2 = alc.multiplicacionMatricial(V, Tmp1) 
    
    # W = Yt @ Tmp2
    # W = Yt @ (V @ S_plus @ U.T)
    W = alc.multiplicacionMatricial(Yt, Tmp2) 

    return W

# --------------------------------------
# ITEM 4 - Descomposicion QR
# --------------------------------------
def xTraspuestaQR(X, metodo):
    Q, R = alc.calculaQR(alc.traspuesta(X), metodo, tol=1e-12)
    return  Q, R

def pinvHouseHolder(Q, R, Y):
    #
    #   Para evitar usar inversas, usamos la funcion res_tri que calcula Lx=b 
    #   con una L triangular superior, que en este caso es R
    #

    Q = alc.traspuesta(Q)
    filas_q, columnas_q = Q.shape
    Vt = np.zeros((filas_q, columnas_q)) 
    for c in range(columnas_q):
        Vt[:, c] = alc.res_tri(R, Q[:, c], inferior=False)
    V = alc.traspuesta(Vt) 
    W = alc.multiplicacion_matricial(Y, V)
    return W

def pinvGramSchmidt(Q, R, Y):
    Q = alc.traspuesta(Q)
    filas_q, columnas_q = Q.shape
    Vt = np.zeros((filas_q, columnas_q)) 
    for c in range(columnas_q):
        Vt[:, c] = alc.res_tri(R, Q[:, c], inferior=False)
    V = alc.traspuesta(Vt) 
    W = alc.multiplicacion_matricial(Y, V)
    return W


# --------------------------------------------
# ITEM 5 - Pseudo-Inversa de Moore-Penrose 
# --------------------------------------------
def esPseudoInversa(X, pX, tol):
    return (condicion1(X, pX, tol) and
            condicion2(X, pX, tol) and
            condicion3(X, pX, tol) and
            condicion4(X, pX, tol))

def condicion1(X, pX, tol):
    XpX = alc.multiplicacionMatricial(X, pX)
    XpXX = alc.multiplicacionMatricial(XpX, X)
    return alc.matricesIguales(XpXX, X, tol)

def condicion2(X, pX, tol):
    pXX = alc.multiplicacionMatricial(pX, X)
    pXXpX = alc.multiplicacionMatricial(pXX, pX)
    return alc.matricesIguales(pXXpX, pX, tol)

def condicion3(X, pX, tol):
    XpX = alc.multiplicacionMatricial(X, pX)
    T = alc.traspuesta(XpX)
    return alc.matricesIguales(XpX, T, tol)

def condicion4(X, pX, tol):
    pXX = alc.multiplicacionMatricial(pX, X)
    T = alc.traspuesta(pXX)
    return alc.matricesIguales(pXX, T ,tol)

# --------------------------------------------
# ITEM 6 - Evaluación y Benchmarking
# --------------------------------------------
"""
Si pred[0] > pred[1] -> gato
Si pred[1] > pred[0] -> perro
"""
def predecir_clases(W, Xv):
    Y_pred = alc.multiplicacionMatricial(W, Xv)
    clases = np.zeros(Y_pred.shape[1], dtype=int)

    for i in range(Y_pred.shape[1]):
        if Y_pred[0, i] > Y_pred[1, i]:
            clases[i] = 0   # gato
        else:
            clases[i] = 1   # perro
    return clases

def clases_reales(Y):
    clases = np.zeros(Y.shape[1], dtype=int)

    for i in range(Y.shape[1]):
        if Y[0, i] == 1:
            clases[i] = 0   # gato
        else:
            clases[i] = 1   # perro
    return clases

def matriz_confusion(clases_reales, clases_pred):
    M = np.zeros((2,2), dtype=int)
    for i in range(clases_reales.shape[0]):
        M[clases_reales[i], clases_pred[i]] += 1
    return M

def evaluar_metodo(W, Xv, Yv):
    true = clases_reales(Yv)
    pred = predecir_clases(W, Xv)
    return matriz_confusion(true, pred)

# EVALÚO EN CADA MÉTODO

resultados = {}

W_en = pinvEcuacionesNormales(Xt, L, Yt)
M_en = evaluar_metodo(W_en, Xv, Yv)
resultados["Ecuaciones Normales"] = M_en

W_svd = pinvSVD(U, S, V, Yt)
M_svd = evaluar_metodo(W_svd, Xv, Yv)
resultados["SVD"] = M_svd

Q_hh, R_hh = xTraspuestaQR(Xt, "RH")
W_hh = pinvHouseHolder(Q_hh, R_hh, Yt)
M_hh = evaluar_metodo(W_hh, Xv, Yv)
resultados["QR Householder"] = M_hh

Q_gs, R_gs = xTraspuestaQR(Xt, "GS")
W_gs = pinvGramSchmidt(Q_gs, R_gs, Yt)
M_gs = evaluar_metodo(W_gs, Xv, Yv)
resultados["QR Gram-Schmidt"] = M_gs

# MÉTRICAS

def calcular_metricas(M):
    # M:
    # [[TP_gato, FN_gato],
    #  [FP_gato, TP_perro]]

    tp_g = M[0,0]
    fn_g = M[0,1]
    fp_g = M[1,0]
    tp_p = M[1,1]

    tn_g = tp_p
    tn_p = tp_g

    # ACCURACY
    accuracy = (tp_g + tp_p) / np.sum(M)

    # PRECISIÓN
    prec_g = tp_g / (tp_g + fp_g) if (tp_g + fp_g) > 0 else 0
    prec_p = tp_p / (tp_p + fn_g) if (tp_p + fn_g) > 0 else 0

    # RECALL
    rec_g = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0
    rec_p = tp_p / (tp_p + fp_g) if (tp_p + fp_g) > 0 else 0

    # F1
    f1_g = 2 * prec_g * rec_g / (prec_g + rec_g) if (prec_g + rec_g) > 0 else 0
    f1_p = 2 * prec_p * rec_p / (prec_p + rec_p) if (prec_p + rec_p) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Precisión Gato": prec_g,
        "Recall Gato": rec_g,
        "F1 Gato": f1_g,
        "Precisión Perro": prec_p,
        "Recall Perro": rec_p,
        "F1 Perro": f1_p,
        "Matriz": M
    }

# AHORA ARMO LA TABLA COMPARATIVA

tabla = pd.DataFrame({
    metodo: {
        "Accuracy": met["Accuracy"],
        "Precisión Gato": met["Precisión Gato"],
        "Recall Gato": met["Recall Gato"],
        "F1 Gato": met["F1 Gato"],
        "Precisión Perro": met["Precisión Perro"],
        "Recall Perro": met["Recall Perro"],
        "F1 Perro": met["F1 Perro"]
    }
    for metodo, met in resultados.items()
}).T

tabla

# -----------------------------------------------
# ITEM 7 - TESTS (AGREGADO)   borrar despues     
# -----------------------------------------------
import numpy.testing as npt


def test_algoritmo_qr():
    print("\n--- Corriendo tests para Algoritmo 3 (QR) ---")
    
    # 1. Definimos un problema de juguete
    # X es (n x p) con n < p, como en el TP (n=2, p=3)
    X_test = np.array([
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0]
    ], dtype=float)

    # Y es (m x p) (m=1, p=3)
    Y_test = np.array([
        [5.0, 3.0, 4.0]
    ], dtype=float)

    # 2. Calculamos la solución real usando numpy
    # W = Y @ X_pinv
    X_pinv_real = np.linalg.pinv(X_test)
    W_real = Y_test @ X_pinv_real

    print(f"  W (real) esperada:\n {W_real}")

    # 3. Testeamos pinvHouseHolder
    try:
        Q_hh, R_hh = xTraspuestaQR(X_test, "RH")
        W_hh = pinvHouseHolder(Q_hh, R_hh, Y_test)
        
        # Comparamos tu W con la W real
        npt.assert_allclose(W_hh, W_real, atol=1e-8)
        print("  [PASÓ] pinvHouseHolder (RH) calcula W correctamente.")
    except Exception as e:
        print(f"  [FALLÓ] pinvHouseHolder (RH): {e}")

    # 4. Testeamos pinvGramSchmidt (después de arreglar alc.py)
    try:
        Q_gs, R_gs = xTraspuestaQR(X_test, "GS")
        
        if Q_gs is None:
            print("  [FALLÓ] pinvGramSchmidt (GS): 'xTraspuestaQR' devolvió None.")
            print("          (Revisa el 'if(esCuadrada(A))' en 'alc.py:QR_con_GS')")
            return

        W_gs = pinvGramSchmidt(Q_gs, R_gs, Y_test)
        
        # Comparamos tu W con la W real
        npt.assert_allclose(W_gs, W_real, atol=1e-8)
        print("  [PASÓ] pinvGramSchmidt (GS) calcula W correctamente.")
    except Exception as e:
        print(f"  [FALLÓ] pinvGramSchmidt (GS): {e}")


# --- Esto hace que el test se corra solo cuando ejecutas 'python main.py' ---
if __name__ == "__main__":
    
    # (Opcional) puedes comentar esto si no quieres cargar datos siempre
    # print("Cargando dataset real (Xt, Yt)...")
    # Xt, Yt, Xv, Yv = cargarDataSet(path)
    # print(f"  Xt shape: {Xt.shape}, Yt shape: {Yt.shape}")
    
    # Corremos los tests
    test_algoritmo_qr()
