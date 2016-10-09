import numpy as np

f = open('r_wpbc.data', 'r')

np.set_printoptions(suppress=True)

data = []
for line in f.readlines():
    line = line.strip()
    line = line.split(",")
    line = [float(i) for i in line]
    data.append(line)

x = np.matrix(data)    #Matriz x (entradas) es 194*33
print(x)

y = []                 #Vector columna con salidas (Time)
i=0
for fila in x:
    y.append(x[i,32])
    i += 1
print(y)
