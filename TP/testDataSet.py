import numpy as np
import numpy.testing as npt
import os
from cargarDataSet import cargarDataSet # Importa tu función

print("Ejecutando tests (versión corregida) para cargarDataSet...")

# --- Configuración ---
path_base = os.path.join("template-alumnos", "dataset", "cats_and_dogs")

try:
    # --- Ejecución ---
    Xt, Yt, Xv, Yv = cargarDataSet(path_base)

    # --- Tests ---
    print("\n[TEST 1/4] Verificando tipos de dato...")
    assert isinstance(Xt, np.ndarray), "Xt no es un numpy.ndarray"
    assert isinstance(Yt, np.ndarray), "Yt no es un numpy.ndarray"
    assert isinstance(Xv, np.ndarray), "Xv no es un numpy.ndarray"
    assert isinstance(Yv, np.ndarray), "Yv no es un numpy.ndarray"
    print("  [PASÓ] Todos los return son 'numpy.ndarray'.")

    print("\n[TEST 2/4] Verificando dimensiones (shape)...")
    # AHORA CORREGIDO:
    # Train: 1000 gatos + 1000 perros = 2000
    # Val: 500 gatos + 500 perros = 1000
    npt.assert_equal(Xt.shape, (1536, 2000), "Shape de Xt incorrecto.")
    npt.assert_equal(Yt.shape, (2, 2000), "Shape de Yt incorrecto.")
    npt.assert_equal(Xv.shape, (1536, 1000), "Shape de Xv incorrecto.")
    npt.assert_equal(Yv.shape, (2, 1000), "Shape de Yv incorrecto.")
    print("  [PASÓ] Dimensiones de Xt, Yt, Xv, Yv son correctas.")

    print("\n[TEST 3/4] Verificando contenido de Targets (Yt e Yv)...")
    gato_target = np.array([1, 0])
    perro_target = np.array([0, 1])

    # --- Verificación Yt (Train - 2000 total) ---
    npt.assert_array_equal(Yt[:, 0], gato_target, "Yt: Primera col (gato) incorrecta.")
    npt.assert_array_equal(Yt[:, 999], gato_target, "Yt: Última col de gato incorrecta.")
    npt.assert_array_equal(Yt[:, 1000], perro_target, "Yt: Primera col de perro incorrecta.")
    npt.assert_array_equal(Yt[:, 1999], perro_target, "Yt: Última col (perro) incorrecta.")

    # --- Verificación Yv (Validación - 1000 total) ---
    npt.assert_array_equal(Yv[:, 0], gato_target, "Yv: Primera col (gato) incorrecta.")
    npt.assert_array_equal(Yv[:, 499], gato_target, "Yv: Última col de gato incorrecta.")
    npt.assert_array_equal(Yv[:, 500], perro_target, "Yv: Primera col de perro incorrecta.")
    npt.assert_array_equal(Yv[:, 999], perro_target, "Yv: Última col (perro) incorrecta.")
    print("  [PASÓ] El contenido y orden de los targets (gato/perro) es correcto.")

    print("\n[TEST 4/4] Verificando que todos los targets suman 1...")
    npt.assert_allclose(np.sum(Yt, axis=0), 1, err_msg="No todas las cols de Yt suman 1.")
    npt.assert_allclose(np.sum(Yv, axis=0), 1, err_msg="No todas las cols de Yv suman 1.")
    print("  [PASÓ] Todas las columnas de Yt e Yv suman 1.")

    print("\n-------------------------------------------")
    print("✅ ¡ÉXITO! Todos los tests pasaron.")
    print("-------------------------------------------")

except FileNotFoundError:
    print("\n[ERROR] No se encontraron los archivos .npy.")
    print(f"Asegúrate que la carpeta 'dataset' esté en: {os.path.abspath(os.path.dirname(__file__))}")
except AssertionError as e:
    print(f"\n❌ [FALLÓ] Un test no pasó: {e}")
except Exception as e:
    print(f"\n❌ [ERROR INESPERADO] {e}")