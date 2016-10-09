import numpy as np

f = open('r_wpbc.data', 'r')

data = []
for line in f.readlines():
    line = line.strip()
    line = line.split(",'")
    data.append(line)


x = np.matrix(data)    #Matriz x (entradas) es 194*33

print(x)


