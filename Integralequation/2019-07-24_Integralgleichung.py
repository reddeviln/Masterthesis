import numpy as np
from matplotlib import pyplot

deltat=0.01
tau0=1.0
v1=1.5
v2=0.5

numsteps = 5000

diffarray = np.zeros((numsteps+1,))
phiarray1  = np.zeros((numsteps+1,))

phiarray1[0] = 1.0

denom = 1.0/(tau0 + 0.5*deltat*(v1*phiarray1[0]+v2*phiarray1[0]**2+1))

for n in range(1,numsteps+1):
    diffarray[n] = phiarray1[n-1]
    for i in range(1,n):
        diffarray[n] += (v1*phiarray1[i]+v2*phiarray1[i]**2)*diffarray[n-i]
    diffarray[n] *= -deltat*denom
    phiarray1[n] = phiarray1[n-1] + diffarray[n]
    print("Step {}: diffarray[n]={} phiarray[n]={}".format(n,diffarray[n],phiarray1[n]))
h=deltat
numsteps=14
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
pyplot.plot(deltat*np.arange(20001),phiarray1)
pyplot.title("Approximations of $\phi$")
pyplot.xlabel("t")
pyplot.legend(["$\phi$ using new algorithm","$\phi$ using $\mathcal{O}(n^2)$ algorithm"])
pyplot.show()
