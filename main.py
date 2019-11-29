# coding: utf-8
import numpy as np
from numpy.linalg import norm  as F
from ACA_.aca import ACA
import matplotlib.pyplot as plt
import time
import random
import sys

# 初始化
'''
Z = np.array([[1, 0, 0, 5, 6], [3, 4, 0, -6, -1], [-3, 0, -2, -6, -1], [0, 4, 4, 2, 0]])
R = np.array(np.zeros([4, 5]))

i, j, v, u, f = [0], [], [], [], []
'''

e = [0.00001,0.0001,0.001,0.01,0.1,1,10]
temp = np.array([[20 * random.randint(-1, 1) * random.random() for i in range(1000)] for j in range(1000)])
t=[]
error=[]
b=[]
r=[]
print(sys.getsizeof(temp))
for n in e:
    Z = temp.copy()
    R = np.array(np.zeros([1000, 1000]))
    i, j, v, u, f = [0], [], [], [], []
    s_t=time.time()
    a=ACA(Z, R, f, u, v, i, j,n)
    a.solve()
    e_t=time.time()
    t.append(e_t-s_t)

    b.append(np.log10(sys.getsizeof(a.u)+sys.getsizeof(a.v)) )
    error.append(np.log10( F( np.dot(np.mat(a.u).T, a.v)-Z )   ))
    r.append(len(a.v))


print(t,error,b)

plt.figure()
plt.subplot(2,2,1)
plt.title('time')
plt.plot(t,label="time")
plt.legend()

plt.subplot(2,2,2)
plt.title('error')
plt.plot(error,label="error")
plt.legend()

plt.subplot(2,2,3)
plt.title('bytes')
plt.plot( np.subtract(np.log10(sys.getsizeof(temp)),b),label="b")
plt.legend()

plt.subplot(2,2,4)
plt.title('rank')
plt.plot(np.log10(r),label="r")
plt.legend()

plt.show()

'''
print(a.Z)
print("---=--")
print(a.R)
print("---=--")
print(a.i, a.j)
print("---=--")
print(a.u)
print("---=--")
print(a.v)
print("---=--")
print(a.f)

'''#打印


