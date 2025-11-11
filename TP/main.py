import numpy as np
import alc as alc
import os
path = os.path.join("template-alumnos", "dataset", "cats_and_dogs")

# --------------------------------------
# ITEM 1 - Lectura de Datos
# --------------------------------------


def cargarDataSet(path):
    #   --- Path segun train o val ---
    pathGatosT  = path + "\\train\\cats\\efficientnet_b3_embeddings.npy"
    pathPerrosT = path + "\\train\\dogs\\efficientnet_b3_embeddings.npy"
    pathGatosV  = path + "\\val\\cats\\efficientnet_b3_embeddings.npy"
    pathPerrosV = path + "\\val\\dogs\\efficientnet_b3_embeddings.npy"

    # --- Matrices cargadas
    gatosT  = np.load(pathGatosT)
    perrosT = np.load(pathPerrosT)
    gatosV  = np.load(pathGatosV)
    perrosV = np.load(pathPerrosV)

    # --- como el TP pide que las matrices tengan 2 elementos por columna que indiquen
    # --- si se trata de un gato o un perro, creamos matrices de la forma np.zeros((2, cantColumnas))
    columnasGatosT = gatosT.shape[1]
    columnasPerrosT = perrosT.shape[1]
    columnasGatosV = gatosV.shape[1]
    columnasPerrosV = perrosV.shape[1]

    YGatosT = np.zeros((2, columnasGatosT))
    YGatosT[0, :] = 1 
    YPerrosT = np.zeros((2, columnasPerrosT))
    YPerrosT[1, :] = 1
    YGatosV = np.zeros((2, columnasGatosV))
    YGatosV[0, :] = 1
    YPerrosV = np.zeros((2, columnasPerrosV))
    YPerrosV[1, :] = 1

    Xt = np.concatenate((gatosT, perrosT), 1)
    Yt = np.concatenate((YGatosT, YPerrosT), 1)
    Xv = np.concatenate((gatosV, perrosV), 1)
    Yv = np.concatenate((YGatosV, YPerrosV), 1)

    return Xt, Yt, Xv, Yv


Xt, Yt, Xv, Yv = cargarDataSet(path)



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
