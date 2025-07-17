import numpy as np

test = [2,3,5,1,5]

labels = ['a','b','c','d','e','f']

for i in range(len(test)):
    test[i] = labels[test[i]]
    
print(test)