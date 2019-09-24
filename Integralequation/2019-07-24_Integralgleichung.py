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

pyplot.plot((deltat*np.arange(5001)),phiarray1)
pyplot.xlabel("$t$")
pyplot.ylabel("$\phi$")
pyplot.show()
