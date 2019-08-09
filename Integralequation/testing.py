import numpy as np
from matplotlib import pyplot
import math
s=3

ds = 0.01
s0 = s-5
stilde = np.asarray([s0+n*ds  for n in range(0,500) ])
estilde = np.asarray(np.exp(math.pi/2*np.sinh(stilde)))
estilde1 = np.exp(stilde)
sarray1 = [np.exp(s) for n in range(0,500)]
sarray = [np.exp(math.pi/2*np.sinh(s)) for n in range (0,500)]
F1 = np.arcsinh(2 / math.pi * np.log((sarray - estilde)))
F2 = np.log(sarray1-estilde1)

pyplot.plot(stilde, F1)
pyplot.plot(stilde, F2)
pyplot.legend(['$f(u)=\exp(\pi/2*\sinh(u))$','$f(u)=\exp(u)$'])
pyplot.xlabel("$u'$")
pyplot.ylabel("$F(u')$")
pyplot.title("For $u=3$")
pyplot.show()
