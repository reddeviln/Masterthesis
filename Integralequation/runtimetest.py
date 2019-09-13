import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
import math

# setting properties of the simulation
v1 = 1.5
v2 = 0.5
T = 50
phi0 = 1.
u0 = -7
numStepss = np.arange(2,20001)
def f(ui):
    return np.where(ui == -np.inf, 0, np.exp(ui))


def difff(fi):
    result = np.zeros(len(fi) - 1)
    for i in range(0, len(result)):
        result[i] = fi[i + 1] - fi[i]
    return result


def F(ui, uprime):
    return np.log(f(ui) - f(uprime))

lstar = []
jstar =[]
for numSteps in numStepss:
    u = np.linspace(u0, np.log(T), numSteps)  # equaldistant points u first discretization
    fn = np.zeros(len(u) + 1)
    fn = f(np.concatenate(([-np.inf], u)))  # calculate f(u_i) but include f(-inf)
    diff_f = difff(fn)  # we often just need the difference in f(u_i) - f(u_{i-1}) so we calculate this in advance
    phi = np.zeros(numSteps)  # allocate solution array
    phi[0] = phi0
    h = u[1] - u[0]
    # calculate lstar which is how many intervals of the first discretization the second discretization spreads at the start

    lstar.append(int(np.ceil((u[1] - F(u[1], u[0])) / h)))

    # calculate the area where the two discretizations are overlapping

    # find jstar which is the index of the first sst which lies in the interval [u_n,u_[n+1]]
    for j in range(lstar[-1],0,-1 ):
        condition1 = u[lstar[-1]] - F(u[lstar[-1]], u[lstar[-1] - j]) <= h
        condition2 = u[lstar[-1]] - F(u[lstar[-1]], u[lstar[-1] - j + 1]) > h
        if condition1 and condition2:
            jstar.append(j)
            break
    print(numSteps)
plt.plot(numStepss,lstar)
plt.plot(numStepss,numStepss)
plt.legend(['lstar','Identity'])
plt.xlabel('N')
plt.show()