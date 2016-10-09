import numpy as np
import random

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
i=0
for fila in x:
    y.append(x[i,32])
    i += 1

x = np.delete(x, 32, 1)     # Matriz x (entradas) es 194*32  (vector y eliminado de x)


t = random.sample(range(194),116)
training_ds = x[t,:]        #Matriz con datos de  entrenamiento (60% de los datos iniciales)

