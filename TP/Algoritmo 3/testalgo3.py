import numpy as np
import alc as alc
# Importa tu otro archivo
import algo3 as algo3

# ------------------------
# PLACEHOLDERS
# ------------------------

# n=3 (features), p=5 (muestras). Cumple n < p para Algoritmo 3.
Xt_placeholder = np.array([
    [0.1, 0.5, 1.2, 0.8, 0.3],
    [1.1, 0.2, 0.7, 1.5, 1.0],
    [0.4, 1.3, 0.1, 0.2, 1.4]
], dtype=float)

# m=2 (clases), p=5 (muestras).
# Clases one-hot: [1,0]T es Gato, [0,1]T es Perro [cite: 186]
# Muestras: Gato, Perro, Perro, Gato, Perro
Yt_placeholder = np.array([
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 1.0]
], dtype=float)

print("Dimensiones de Xt:", Xt_placeholder.shape) # (3, 5)
print("Dimensiones de Yt:", Yt_placeholder.shape) # (2, 5)

# ------------------------
# TESTS PARA QR
# ------------------------

def test_qr(A, metodo):
    """
    Prueba la descomposicion QR para un metodo dado ("GS" o "RH").
    A es la matriz a descomponer (en tu caso, X.T).
    """
    print(f"\n--- Probando QR con método: {metodo} ---")
    
    # 1. Ejecutar la descomposición
    try:
        Q, R = alc.calculaQR(A, metodo, tol=1e-12)
        if Q is None or R is None:
            print("ERROR: La descomposición falló (retornó None).")
            return False
        print("Descomposición completada.")
    except Exception as e:
        print(f"ERROR: La descomposición falló con una excepción: {e}")
        return False

    # 2. Verificar propiedad de Reconstrucción (A = QR)
    A_reconstruida = alc.multiplicacion_matricial(Q, R)
    if alc.matricesIguales(A, A_reconstruida):
        print("Test RECONSTRUCCIÓN (A = QR): \tPASÓ")
    else:
        print("Test RECONSTRUCCIÓN (A = QR): \tFALLÓ")
        # Opcional: imprimir la diferencia
        # print(A - A_reconstruida)

    # 3. Verificar propiedad de Ortogonalidad (Q.T @ Q = I)
    n_q = Q.shape[1] # Número de columnas de Q
    identidad = np.eye(n_q)
    Q_trans_Q = alc.multiplicacion_matricial(alc.traspuesta(Q), Q)
    
    if alc.matricesIguales(identidad, Q_trans_Q):
        print("Test ORTOGONALIDAD (Q.T @ Q = I): \tPASÓ")
    else:
        print("Test ORTOGONALIDAD (Q.T @ Q = I): \tFALLÓ")
        # Opcional: imprimir la diferencia
        # print(identidad - Q_trans_Q)
    
    print("--------------------------------------")


# ------------------------
# EJECUTAR LOS TESTS
# ------------------------

# Usamos los placeholders de antes.
# Recordar: Alg. 3 calcula QR sobre X.T 
A_para_test = alc.traspuesta(Xt_placeholder)

# A_para_test tiene dimensiones (5, 3) (p > n, "alta")
# Tu QR_con_GS puede no estar preparada para matrices no cuadradas
# Tu QR_con_HH sí debería estarlo.
# ¡Asegúrate que tus funciones en alc.py soporten matrices rectangulares!

# test_qr(A_para_test, "GS") # Descomenta cuando Gram-Schmidt soporte rectangulares
test_qr(A_para_test, "RH") # Householder usualmente es más robusto para esto


# ------------------------
# PROBAR EL ALGORITMO 3 COMPLETO
# ------------------------# --- EN TU SCRIPT DE TEST (testalgo3.py) ---

print("\n--- Probando Algoritmo 3 (Householder) ---")

# Obtenemos las dimensiones p (muestras) y n (features)
A_para_test = alc.traspuesta(Xt_placeholder)
p, n = A_para_test.shape # p=5, n=3

# 1. Descomponer (obtenemos Q completa y R completa)
Q_full, R_full = algo3.xTraspuestaQR(Xt_placeholder, "RH")
# Q_full tiene shape (5, 5), R_full tiene shape (5, 3)

if Q_full is not None:
    # 2. RECORTAR A TAMAÑO "ECONÓMICO"
    # Q_eco debe ser (p x n) -> (5, 3)
    Q_eco = Q_full[:, :n]
    
    # R_eco debe ser (n x n) -> (3, 3)
    R_eco = R_full[:n, :]

    # 3. Calcular W con las matrices recortadas
    W_h = algo3.pinvHouseHolder(Q_eco, R_eco, Yt_placeholder)
    
    print("Cálculo de W completado.")
    print("Dimensiones de W:", W_h.shape) # Debería ser (2, 3)
else:
    print("No se pudo calcular W porque QR falló.")