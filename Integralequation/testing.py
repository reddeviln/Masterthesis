import numpy as np
from matplotlib import pyplot
import math
ses=[3]

for s in ses:
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
pyplot.legend(['$f(u)=\exp(\pi/2*\sinh(u)$' , '$f(u)=\exp(u)$'])
pyplot.xlabel("$u'$")
pyplot.ylabel("$F(u')$")
pyplot.axhline(y=2,xmin=0,xmax=0.87, ls='dashed',color='grey')
pyplot.axhline(y=1,xmin=0,xmax=0.925, ls='dashed',color='grey')
pyplot.axhline(y=0,xmin=0,xmax=0.95, ls='dashed',color='grey')
pyplot.axvline(x=2.55,ymin=0,ymax=0.75, ls='dashed', color='grey')
pyplot.axvline(x=2.85,ymin=0,ymax=0.55, ls='dashed', color='grey')
pyplot.axvline(x=2.95,ymin=0,ymax=0.35, ls='dashed', color='grey')
pyplot.text(2.3,-2.1,'$w^3_2$',color='grey')
pyplot.text(2.6,-2.1,'$w^3_1$',color='grey')
pyplot.text(2.8,-2.35,'$w^3_0$',color='grey')
pyplot.show()
#pyplot.savefig('HistoryF.pdf')
