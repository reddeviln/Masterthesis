import numpy as np
from matplotlib import pyplot as plt

x=0.001*np.arange(2000)
plt.plot(x,x)
plt.plot(0,0,marker='o')
plt.plot([0.4,0.4],[0,0.4],marker='o',linestyle='')
plt.plot([0.8,0.8,0.8],[0,0.4,0.8],marker='o',linestyle='')
plt.plot([1.2,1.2,1.2,1.2],[0,0.4,0.8,1.2],marker='o',linestyle='')
plt.plot([1.6,1.6,1.6,1.6,1.6],[0,0.4,0.8,1.2,1.6],marker='o',linestyle='')
plt.plot([2,2,2,2,2,2],[0,0.4,0.8,1.2,1.6,2],marker='o',linestyle='')
plt.xlabel("$t$")
plt.ylabel("$t'$")
plt.show()