import numpy as np
import alc as alc
import os
import time
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
    W = alc.multiplicacionMatricial(Y, V)
    return W

def pinvGramSchmidt(Q, R, Y):
    Q = alc.traspuesta(Q)
    filas_q, columnas_q = Q.shape
    Vt = np.zeros((filas_q, columnas_q)) 
    for c in range(columnas_q):
        Vt[:, c] = alc.res_tri(R, Q[:, c], inferior=False)
    V = alc.traspuesta(Vt) 
    W = alc.multiplicacionMatricial(Y, V)
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
# ITEM 6 - Evaluacion y Benchmarking     
# -----------------------------------------------
def evaluacionQR_HH(Xt, Yt, Xv, Yv):
    Q, R = xTraspuestaQR(Xt, "RH")
    W = pinvHouseHolder(Q, R, Yt)
    Ypred = alc.multiplicacionMatricial(W, Xv)
    filas, columnas = Ypred.shape
    print(filas, columnas)
    YpredClases = np.zeros((filas, columnas)) 
    
    for c in range(columnas):
        indice_max = 0
        valor_max = Ypred[0, c] 
        
        for f in range(1, filas): 
            if Ypred[f, c] > valor_max:
                valor_max = Ypred[f, c]
                indice_max = f
        
        if indice_max == 0:
            YpredClases[:, c] = [1.0, 0.0] # Gato
        else:
            YpredClases[:, c] = [0.0, 1.0] # Perro
            
    matrizConfusion = np.zeros((2, 2))
    #la matriz se vera de la forma:
    #[gatos predecidos correctamente,      gatos predecidos como perros]
    #[perros predecidos como gatos,     perros predecidos correctamente]
    for c in range(columnas):
        if YpredClases[0, c] == 1.0:
            pred = 0                    #en la columna 0 estan los predecidos como gatos
        else:
            pred = 1                    #en la columna 1 estan los predecidos como perros

        if Yv[0, c] == 1.0:
            real = 0                    #en la fila 0 estan los gatos reales
        else:   
            real = 1                    #en la fila 1 estan los perros reales

        matrizConfusion[real, pred] += 1
    
    print("--- Matriz Confusion QR con HH---")
    print(matrizConfusion)


def evaluacionQR_GS(Xt, Yt, Xv, Yv):
    Q, R = xTraspuestaQR(Xt, "GS")
    W = pinvGramSchmidt(Q, R, Yt)
    Ypred = alc.multiplicacionMatricial(W, Xv)
    filas, columnas = Ypred.shape
    print(filas, columnas)
    YpredClases = np.zeros((filas, columnas)) 

    print(f"Iniciando Ypred(Gram-Schmidt) para matriz {filas}x{columnas}...")
    for c in range(columnas):
        # --- BARRA DE PROGRESO ---
        # Imprime el progreso y vuelve al inicio de la línea con '\r'
        print(f"  Progreso Ypred: {c+1}/{columnas}", end="\r")

        indice_max = 0
        valor_max = Ypred[0, c] 
        
        for f in range(1, filas): 
            if Ypred[f, c] > valor_max:
                valor_max = Ypred[f, c]
                indice_max = f
        
        if indice_max == 0:
            YpredClases[:, c] = [1.0, 0.0] # Gato
        else:
            YpredClases[:, c] = [0.0, 1.0] # Perro


    print("--- Y PRED ---")
    print(Ypred)
    matrizConfusion = np.zeros((2, 2))
    #la matriz se vera de la forma:
    #[gatos predecidos correctamente,      gatos predecidos como perros]
    #[perros predecidos como gatos,     perros predecidos correctamente]

    print(f"Iniciando Matriz Confusion para matriz {filas}x{columnas}...")
    for c in range(columnas):
        print(f"  Progreso Confusion: {c+1}/{columnas}", end="\r")
        if YpredClases[0, c] == 1.0:
            pred = 0                    #en la columna 0 estan los predecidos como gatos
        else:
            pred = 1                    #en la columna 1 estan los predecidos como perros

        if Yv[0, c] == 1.0:
            real = 0                    #en la fila 0 estan los gatos reales
        else:   
            real = 1                    #en la fila 1 estan los perros reales

        matrizConfusion[real, pred] += 1
    
    print("--- Matriz Confusion QR con GS---")
    print(matrizConfusion)


print("--- Evaluacion con matriz completa de GS ---")
try:
    start_time = time.time()
    evaluacionQR_GS(Xt, Yt, Xv, Yv)
    print(f"    Evaluacion con matriz completa de GS terminado en: {time.time() - start_time:.4f} seg.")
    
except Exception as e:
    print(f"\n  [FALLÓ] pinvGramSchmidt (GS): {e}")



print("--- Iniciando test de evaluacion con matrices RECORTADAS (MIXTO) ---")

# 1. Definimos los tamaños de las rodajas
n_features = 200  # Usar 100 features (en lugar de 1536)
n_train = 400     # Usar 200 muestras de train (en lugar de 2000)
n_val_gatos = 50  # Usar 25 gatos de validación
n_val_perros = 50 # Usar 25 perros de validación

# 2. Creamos las matrices recortadas de ENTRENAMIENTO
Xt_slice = Xt[:n_features, :n_train]
Yt_slice = Yt[:, :n_train]

# 3. Creamos las matrices recortadas de VALIDACIÓN (Mixto)
#    (Yv tiene gatos de 0 a 499, y perros de 500 a 999)
Xv_gatos = Xv[:n_features, :n_val_gatos]
Xv_perros = Xv[:n_features, 500:(500 + n_val_perros)]
Xv_slice_mixto = np.concatenate((Xv_gatos, Xv_perros), axis=1)

Yv_gatos = Yv[:, :n_val_gatos]
Yv_perros = Yv[:, 500:(500 + n_val_perros)]
Yv_slice_mixto = np.concatenate((Yv_gatos, Yv_perros), axis=1)

print(f"Shapes (entrenamiento): {Xt_slice.shape}, {Yt_slice.shape}")
print(f"Shapes (validación mixta): {Xv_slice_mixto.shape}, {Yv_slice_mixto.shape}")

# 4. Llamamos a la función de evaluación CON LAS RODAJAS MIXTAS
evaluacionQR_GS(Xt_slice, Yt_slice, Xv_slice_mixto, Yv_slice_mixto)