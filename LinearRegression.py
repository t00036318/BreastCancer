import numpy as np

f = open('r_wpbc.data', 'r')

data = []
for line in f.readlines():
    line = line.strip()
    line = line.split(",'")
    for element in line:
        data.append(element)

x = np.matrix(data)    #Matriz x (entradas) es 194*33

print(x)


