# En test_integracion_qr.py

import numpy as np
import numpy.testing as npt
import time
import os
import alc as alc 

# Importamos las funciones de tu main.py
from main import cargarDataSet, xTraspuestaQR, pinvHouseHolder, pinvGramSchmidt

print("--- Test de Integración (Ítem 1 + Ítem 4) ---")
print("Cargando datos reales del dataset...")

# --- 1. SETUP: Cargar datos reales ---
path = os.path.join("template-alumnos", "dataset", "cats_and_dogs")
try:
    Xt_full, Yt_full, _, _ = cargarDataSet(path)
except FileNotFoundError:
    print("\n[ERROR] No se encontró el dataset.")
    exit()
except Exception as e:
    print(f"\n[ERROR] Tu función cargarDataSet falló: {e}")
    exit()

# --- 2. CAMBIO CLAVE: Usar un slice "ancho" (n < p) ---
# X debe ser (n, p) donde n < p. 
# Hacemos X (100, 200) para que X^T sea (200, 100) (alta)
N_FEATURES = 100
N_MUESTRAS = 200 
Xt = Xt_full[:N_FEATURES, :N_MUESTRAS] # (100, 200)
Yt = Yt_full[:, :N_MUESTRAS]            # (2, 200)

print(f"  Datos cargados. Usando slice (features={N_FEATURES}, muestras={N_MUESTRAS}).")
print(f"  Shape de Xt (slice): {Xt.shape}")
print(f"  Shape de Yt (slice): {Yt.shape}")
print(f"  Shape de X^T (para QR): ({Xt.shape[1]}, {Xt.shape[0]})")


# --- 3. GROUND TRUTH: Calcular W real con NumPy ---
print("\nCalculando W (esperada) con np.linalg.pinv...")
W_real = Yt @ np.linalg.pinv(Xt)
print("  W (esperada) calculada.")

# --- 4. TEST: pinvHouseHolder ---
print("\n--- Testeando pinvHouseHolder ('RH') ---")
try:
    start_time = time.time()
    print("  Calculando QR(X^T) con Householder...")
    Q_hh, R_hh = xTraspuestaQR(Xt, "RH")
    print(f"    ... QR(X^T) terminado en {time.time() - start_time:.4f} seg.")
    
    print("  Calculando W_hh con pinvHouseHolder...")
    start_time_w = time.time()
    W_hh = pinvHouseHolder(Q_hh, R_hh, Yt)
    print(f"    ... W_hh calculado en {time.time() - start_time_w:.4f} seg.")
    
    npt.assert_allclose(W_hh, W_real, atol=1e-6)
    print("\n  [PASÓ] pinvHouseHolder (RH) calcula W correctamente con datos reales.")
    print(f"  (Tiempo total RH: {time.time() - start_time:.4f} seg.)")

except Exception as e:
    print(f"\n  [FALLÓ] pinvHouseHolder (RH): {e}")

# --- 5. TEST: pinvGramSchmidt ---
print("\n--- Testeando pinvGramSchmidt ('GS') ---")
try:
    start_time = time.time()
    print("  Calculando QR(X^T) con Gram-Schmidt...")
    Q_gs, R_gs = xTraspuestaQR(Xt, "GS")
    print(f"    ... QR(X^T) terminado en {time.time() - start_time:.4f} seg.")
    
    print("  Calculando W_gs con pinvGramSchmidt...")
    start_time_w = time.time()
    W_gs = pinvGramSchmidt(Q_gs, R_gs, Yt)
    print(f"    ... W_gs calculado en {time.time() - start_time_w:.4f} seg.")
    
    npt.assert_allclose(W_gs, W_real, atol=1e-6)
    print("\n  [PASÓ] pinvGramSchmidt (GS) calcula W correctamente con datos reales.")
    print(f"  (Tiempo total GS: {time.time() - start_time:.4f} seg.)")

except Exception as e:
    print(f"\n  [FALLÓ] pinvGramSchmidt (GS): {e}")