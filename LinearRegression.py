import numpy as np
import random
def featureNormalize(X):

    [m,n] = X.shape
    i=0
    j=0
    mean =[]
    std = []
    for j in range(n):
        mean.append(0.0)
        for i in range(m):
            mean[j]+=X[i,j]
        mean[j]/=m
        std.append(np.amax(X[:,j])-np.amin(X[:,j]))
        return X

def main():
    f = open('r_wpbc.data', 'r')

    np.set_printoptions(suppress=True)

    data = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        line = [float(i) for i in line]
        data.append(line)

    x = np.matrix(data)         #Matriz x (entradas) es 194*33


    y = []                      #Vector columna con salidas (Time)
    for i in range(194):
        y.append(x[i,32])

    x = np.delete(x, 32, 1)     #Matriz x (entradas) es 194*32  (vector y eliminado de x)


    t = random.sample(range(194),116)
    training_ds = x[t,:]        #Matriz con datos de entrenamiento (60% de los datos iniciales)

    t2 = []
    for element in range(194):
        if element not in t:
            t2.append(element)

    tests_ds = x[t2,:]          #Matriz con datos de prueba (40% de los datos iniciales)

    training_ds=featureNormalize(training_ds) #Normalizar datos del conjunto de entrenamiento

main()