import numpy as np
path = "template-alumnos\dataset\cats_and_dogs"


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
