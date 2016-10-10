import numpy as np
import random
import copy as c


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
        for k in range(m):
            X[k,j]=(X[k,j]-mean[j])/std[j]
    return X

def gradientDescent (X, y, theta, alpha, num_iters):
    J_histor = []
    error = 0.0
    [m,n] = X.shape         # m = numero de ejemplos de entrenamiento
    i=0
    theta_temp = []
    j=0
    for i in range (num_iters):
        for j in range (116):
            h = np.dot(X[j,:],theta)
            #print (h)
            error = h-y[j]
        error/=116


    return [theta,J_histor]


def pseudoinverse(X):                            #Función que calcula la pseudoinversa de una matriz
    Xt = np.transpose(X)
    Xinv = np.linalg.inv(X.dot(Xt))
    Xplus = Xinv.dot(X)
    return Xplus


def main():
    f = open('r_wpbc.data', 'r')

    np.set_printoptions(suppress=True)

    data = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        line = [float(i) for i in line]
        data.append(line)

    x = np.matrix(data)                           #Matriz x (entradas) es 194*33
    t = random.sample(range(194),116)
    x_training = x[t,:]                           #Matriz con datos de entrenamiento (60% de los datos iniciales)

    t2 = []
    for element in range(194):
        if element not in t:
            t2.append(element)

    x_test = x[t2,:]                              #Matriz con datos de prueba (40% de los datos iniciales)
    y_training = []
    y_test =[]
    for i in range(116):
        y_training.append(x_training[i,32])       #Matriz y de entrenamiento
    for i in range(78):
        y_test.append(x_test[i,32])               #Matriz y de pruebas

    x_training = np.delete(x_training,32,1)       #Matriz x de entrenamiento
    x_test = np.delete(x_test,32,1)               #Matriz x de pruebas


    x_training = featureNormalize(x_training)    #Normalizar datos del conjunto de entrenamiento
    x_training = np.concatenate((np.ones((116,1)),x_training),axis=1)

    x_test = np.concatenate((np.ones((78,1)),x_test),axis=1)
    theta = np.zeros((33,1))
    alpha = 0.09
    iterat = 400
    temp = gradientDescent(x_training,y_training,theta,alpha,iterat)
    theta = np.copy(temp[0])
    J = []
    J = c.deepcopy(temp[1])
    del temp[:]


main()

