import numpy as np
from matplotlib import pyplot as plt
import math

splot = np.arange(-6, 6, 0.01)
result1=[]
result2=[]

for s in splot:

    ds = 0.001
    s0 = s - 5
    stilde = np.arange(s0, s-ds, ds)
    F1st = np.exp(math.pi/2*np.sinh(stilde))
    F1s = np.exp(math.pi/2*np.sinh(s))
    F1 = np.arcsinh(2 / math.pi * np.log((F1s - F1st)))
    diffF1 = F1-stilde
    int1 = np.argmin(abs(diffF1))
    result1.append(stilde[int1])

    F2st = np.exp(stilde)
    F2s = np.exp(s)
    F2 = np.log(F2s - F2st)
    diffF2 = F2-stilde
    int2 = np.argmin(abs(diffF2))
    result2.append(stilde[int2])

plt.plot(splot, splot-result1)
plt.plot(splot, splot-result2)
plt.legend(['$f(u)= \exp(\pi/2 * \sinh(u))$','$f(u)=\exp(u)$'])
plt.xlabel('$u$')
plt.ylabel("$u-F(u*)$")
plt.show()


