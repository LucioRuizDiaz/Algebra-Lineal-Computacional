import numpy as np
import matplotlib.pyplot as plt

#Ejercicio 1a
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


'''print(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
print(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
print(norma(np.random.rand(10),2)<=np.sqrt(10))
print(norma(np.random.rand(10),2)>=0)
'''
def normaliza(X, p):
    Y = []
    for x in X:
        Y.append([np.array(norma(x, p))])
    return Y

for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    print( np.allclose(norma(x,'inf'),1) )
