import numpy as np
import alc as alc

# ------------------------
# Descomposicion QR
# ------------------------
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

