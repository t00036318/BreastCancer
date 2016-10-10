import numpy as np
import random
#import matplotlib.pyplot as plt
def featureNormalize(X):

    [m,n] = X.shape

    mean = X.mean(0)
    std = np.amax(X, axis= 0) - np.amin(X, axis= 0)

    X = (X-mean)/std

    return [X,mean,std]

def testNormalize(Xt,mean,sigma):
    Xt = (Xt-mean)/sigma
    return (Xt)

def computeCost(X,y,theta):
    [m,n]= y.shape
    J = (1/(2*m))*sum(np.power((X*theta-y),2))
    return J

def gradientDescent (X, y, theta, alpha, num_iters):
    J_histor = np.zeros((num_iters,1))
    [m,n] = X.shape         # m = numero de ejemplos de entrenamiento
    i=0
    theta_temp = []
    j=0
    for i in range (num_iters):
        h = X*theta
        error=h-y
        theta_temp = np.transpose(((alpha/m)*(np.transpose(error)*X)))
        theta = theta-theta_temp
        J_histor [i] = computeCost(X,y,theta)
    return [theta,J_histor]

def Pseudoinverse(X,Y):                            #Función que calcula la pseudoinversa de una matriz
    Xplus=np.linalg.pinv(X)
    th = Xplus.dot(Y)
    return th

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


    t = random.sample(range(194),116)
    training_ds = x[t,:]        #Matriz con datos de entrenamiento (60% de los datos iniciales)

    t2 = []
    for element in range(194):
        if element not in t:
            t2.append(element)

    tests_ds = x[t2,:]          #Matriz con datos de prueba (40% de los datos iniciales)
    y_training=training_ds[:,32]
    y_test= tests_ds[:,32]
    x = np.delete(x,32,1)       #Matriz x (entradas) es 194*32  (vector y eliminado de x)
    training_ds = np.delete(training_ds,32,1)
    tests_ds = np.delete(tests_ds,32,1)
    [training_ds,mean,sigma]=featureNormalize(training_ds) #Normalizar datos del conjunto de entrenamiento
    print (tests_ds)
    tests_ds = testNormalize(tests_ds,mean,sigma)
    training_ds=np.concatenate((np.ones((116,1)),training_ds),axis=1)
    tests_ds = np.concatenate((np.ones((78,1)),tests_ds),axis=1)
    theta = np.zeros((33,1))
    alpha = 0.09
    iterat = 400
    [theta,J] = gradientDescent(training_ds,y_training,theta,alpha,iterat)
    theta2=Pseudoinverse(training_ds,y_training)
    #print ("Thetas calculados por el metodo de Gradiente Descendiente: \n", theta)
    print (tests_ds[1])
    print (tests_ds*theta)
    #print ("Thetas calculados por el metodo de la Pseudoinversa: \n",theta2)
    print("holi")
    print (tests_ds*theta2)
    """plt.plot(J)
    plt.ylabel('Cost J')
    plt.xlabel('Number of Iterations')
    plt.show()"""


main()