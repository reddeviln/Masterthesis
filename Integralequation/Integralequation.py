import numpy as np
from matplotlib import pyplot

h=0.01
v1=1.5
v2=0.5

numsteps = 15

phiarray  = np.zeros((numsteps+1,))
beta = 0
phiarray[0] = 1.0
m = 0
tarray = np.asarray([h*(2**n - 1) for n in range(0, numsteps+1)])
for n in range(1,numsteps+1):
    m = (v1 * phiarray[n-1] + v2 * phiarray[n-1]**2)
    beta = beta * 0.5 +  m * h / 4
    phiarray[n]= (2**(-n) + beta - m * h / 2) * phiarray[n-1] + m * h / 2 * phiarray[0]
    phiarray[n] *= 1/(h + 2**(-n) + beta)
    print("Step {}: t[n]={} phiarray[n]={}".format(n,tarray[n], phiarray[n]))


pyplot.plot(tarray,phiarray)
pyplot.show()
print(abs(np.exp(-tarray)-phiarray))