import numpy as np

#Ejercicio 1
def esCuadrada(A):
    filas = len(A)
    columnas = len(A[filas - 1])
    return filas == columnas

test1 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

test2 = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8]])

test3 = np.array([[42]])

test4 = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8]])


#Ejercicio 2
def triangSup(A):
    filas = len(A)
    columnas = len(A[filas - 1])
    U = np.zeros((filas, columnas))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if j > i:
                U[i][j] = A[i][j]
    return U
    



#Ejercicio 3
def triangInf(A): 
    filas = len(A)
    columnas = len(A[filas - 1])
    U = np.zeros((filas, columnas))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if j < i:
                U[i][j] = A[i][j]
    return U


#Ejercicio 4
def diagonal(A):
    filas = len(A)
    columnas = len(A[filas - 1])
    U = np.zeros((filas, columnas))
    for i in range(len(A)):
        for j in range(len(A[i])):
            if j == i:
                U[i][j] = A[i][j]
    return U



#Ejercicio 5
def traza(A):
    suma = 0
    for i in range(len(A)):
        for j in range(len(A[i])):
            if j == i:
                suma = suma + A[i][j]
    return suma

#Ejercicio 6
def traspuesta(A):
    filas = len(A)
    columnas = len(A[0]) 
    U = np.zeros((columnas, filas))
    for f in range(filas):
        for c in range(columnas):
            U[c][f] = A[f][c]
    return U


#Ejercicio 7
#def esSimetrica(A):
#    filas = len(A)
 #   columnas = len(A[filas - 1])
  #  traspuesta = traspuesta(A)
   # fila = 0
    #columna = 0
   # iguales = True
   # while((fila < filas and columna < columnas) or )
   # return A == traspuesta

#Ejercicio 8
def calcular(A, x):
    u = []
    filas = len(A)
    columnas = len(A[0])
    for fila in range(filas):
        suma = 0
        for columna in range(columnas):
            suma += A[fila][columna] * x[columna]
        u.append(suma)
    b = np.array(u)
    return b

#Ejercicio 9
def intercambiarFilas(A, i, j):
    filaI = np.array(A[i])
    filaJ = np.array(A[j])
    A[i] = filaJ
    A[j] = filaI
    return A

#print(intercambiarFilas(np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]),2,3))
#Ejercicio 10
def sumar_fila_multiplo(A, i, j, s):
    filaI = np.array(A[i])
    filaJ = np.array(A[j])
    for k in range(len(filaJ)):
        filaJ[k] = filaJ[k] * s
    A[i] = filaI + filaJ
    return A

#print(sumar_fila_multiplo(np.array([[0,1,2],[3,4,5],[6,7,8],[9,10,11]]),1,2,2))
#Ejercicio 11
def esDiagonalmenteDominante(A):
    filas = len(A)
    columnas = len(A[0])
    dominante = True
    for fila in range(filas):
        suma = 0
        for columna in range(columnas):
            if(fila != columna):
                suma += A[fila][columna]
        if (suma >= A[fila][fila]):
            dominante = False
            break
    return dominante

A = np.array([[ 80, 10, 2 ],
     [  1,  7, 0 ],
     [  2,  3, 10 ]])

#print(esDiagonalmenteDominante(A))

#Ejercicio 12
def matrizCirculante(v):
    U = np.zeros((len(v), len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if (j < i ):
                U[i][j] = v[len(v) - (i - j) ]
            else:
                U[i][j] = v[j-i]
    return np.array(U)

    

v = np.array([4,7,89,5,6])

#print(matrizCirculante(v))

#Ejercicio 13
def matrizVandermonde(v):
    n = len(v)
    U = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            U[i][j] = v[j] ** i
    return np.array(U)

v1 = np.array([1, 2, 3, 4, 5])

print(matrizVandermonde(v1))

#Ejercicio 14
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)


#Ejercicio 15
def matrizFibonacci(n):
    U = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            U[i][j] = fibonacci(i+j)
    return np.array(U)
    
#Ejercicio 16
def matrizHilbert(n):
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1/(i+j+1)
    return np.array(H)

