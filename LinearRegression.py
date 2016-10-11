import numpy as np
import random
import matplotlib.pyplot as plt
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

def MAPE(y,h):
    [m,n]=y.shape
    error=(y-h)/y
    error=np.abs(error)
    error=100/m*np.sum(error)
    return error

def main():

    f = open('ENB2012.csv', 'r')

    np.set_printoptions(suppress=True)

    data = []
    for line in f.readlines():
        line = line.strip()
        line = line.split(",")
        line = [float(i) for i in line]
        data.append(line)

    x = np.matrix(data)         #Matriz x (entradas) es 768*10

    [m,n] = x.shape
    t = random.sample(range(m),round((m*0.6)))
    training_ds = x[t,:]        #Matriz con datos de entrenamiento (60% de los datos iniciales)

    t2 = []
    for element in range(m):
        if element not in t:
            t2.append(element)

    tests_ds = x[t2,:]          #Matriz con datos de prueba (40% de los datos iniciales)
    y_training2=training_ds[:,n-1]
    y_training=training_ds[:,n-2]
    y_test2= tests_ds[:,n-1]
    y_test = tests_ds[:,n-2]
    x = np.delete(x,n-1,1)       #Matriz x (entradas) es 194*32  (vector y eliminado de x)
    x = np.delete(x,n-2,1)
    training_ds = np.delete(training_ds,n-1,1)
    training_ds = np.delete(training_ds,n-2,1)
    tests_ds = np.delete(tests_ds,n-1,1)
    tests_ds = np.delete(tests_ds,n-2,1)
    [training_ds,mean,sigma]=featureNormalize(training_ds) #Normalizar datos del conjunto de entrenamiento
    tests_ds = testNormalize(tests_ds,mean,sigma)
    training_ds=np.concatenate((np.ones((round((m*0.6)),1)),training_ds),axis=1)
    tests_ds = np.concatenate((np.ones(((m-round((m*0.6))),1)),tests_ds),axis=1)
    thetaG_y1 = np.zeros((n-1,1))
    thetaN_y1 = np.zeros((n-1,1))
    thetaG_y2 = np.zeros((n-1,1))
    thetaN_y2 = np.zeros((n-1,1))
    alpha = 0.09
    iterat = 400
    [thetaG_y1,J] = gradientDescent(training_ds,y_training,thetaG_y1,alpha,iterat)
    [thetaG_y2,J2] = gradientDescent(training_ds,y_training2,thetaG_y2,alpha,iterat)
    thetaN_y1 = Pseudoinverse(training_ds,y_training)
    thetaN_y2 = Pseudoinverse(training_ds,y_training2)
    '''
    print ("Thetas calculados por el metodo de Gradiente Descendiente para Heating load: \n", thetaG_y1)
    print ("Thetas calculados por el metodo de Gradiente Descendiente para Cooling load: \n", thetaG_y2)
    print ("Thetas calculados por el metodo de la Pseudoinversa para Heating load: \n",thetaN_y1)
    print ("Thetas calculados por el metodo de la Pseudoinversa para Cooling load: \n",thetaN_y2)
    '''
    errorG_y1=MAPE(y_test,tests_ds*thetaG_y1)
    errorG_y2=MAPE(y_test2,tests_ds*thetaG_y2)
    errorN_y1=MAPE(y_test,tests_ds*thetaN_y1)
    errorN_y2=MAPE(y_test2,tests_ds*thetaN_y2)
    plt.figure(1)
    plt.ylabel('Cost J')
    plt.xlabel('Number of Iterations')
    plt.title('Cambio de la funcion de costo Heating Load')
    plt.plot(J)
    plt.figure (2)
    plt.ylabel('Cost J')
    plt.xlabel('Number of Iterations')
    plt.title('Cambio de la funcion de costo Cooling Load')
    plt.plot(J2)
    plt.show()
    print("Error de las predicciones con Gradiente Descendiente para y1 (Heating Load)",errorG_y1)
    print("Error de las predicciones con Metodo de la Pseudoinversa para y1 (Heating Load)",errorN_y1)
    print("Error de las predicciones con Gradiente Descendiente para y2 (Cooling Load)",errorG_y2)
    print("Error de las predicciones con Metodo de la Pseudoinversa para y1 (Cooling Load)",errorN_y2)

main()