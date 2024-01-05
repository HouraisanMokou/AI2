import numpy as np

a = np.array([3,5,6,8,4,6,6,1,5,6])
o = np.flip(np.argsort(a))
print(a)
print(o)
print(a[o])
