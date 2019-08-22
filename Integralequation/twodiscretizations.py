import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate
import math

v1 = 1.5
v2 = 0.5
T = 50
tau0 = 1.0
plt.subplot(211)
for numsteps in [1000, 3000]:
    deltat = T / numsteps

    diffarray = np.zeros((numsteps + 1,))
    phiarray1 = np.zeros((numsteps + 1,))

    phiarray1[0] = 1.0

    denom = 1.0 / (tau0 + 0.5 * deltat * (v1 * phiarray1[0] + v2 * phiarray1[0] ** 2 + 1))

    for n in range(1, numsteps + 1):
        diffarray[n] = phiarray1[n - 1]
        for i in range(1, n):
            diffarray[n] += (v1 * phiarray1[i] + v2 * phiarray1[i] ** 2) * diffarray[n - i]
        diffarray[n] *= -deltat * denom
        phiarray1[n] = phiarray1[n - 1] + diffarray[n]
        print("Step {}: diffarray[n]={} phiarray[n]={}".format(n, diffarray[n], phiarray1[n]))
    plt.plot(deltat * np.arange(numsteps + 1), phiarray1)


def m(v1i, v2i, phii):
    return v1i * phii + v2i * phii ** 2


def f(ui):
    return np.exp(ui)


def diffu(ui):
    return np.exp(ui)


def l(ni):
    if ni == 0:
        return 0
    return np.ceil(np.log(ni))
    # return np.ceil(ni)


def F(ui, ustrich):
    return np.log(f(ui) - f(ustrich))


# lm = 1+4**(-4)
# v1=(2-1/lm)/lm
# v2=(1/lm**2)
hs = [0.005]
numStepss = []
error = []
sums = [0 for n in range(0, len(hs))]
o = 0
for h in hs:

    u0 = -7
    numSteps = int(np.ceil((np.log(T) - u0) / h))
    sums[o] = [0 for y in range(0, numSteps - 1)]
    print('Number of steps', numSteps - 1)
    numStepss.append(numSteps)
    phi = np.zeros(numSteps)
    phi[0] = 1.
    u = np.linspace(u0, np.log(T), numSteps)
    diff = diffu(u)
    ln = l(range(0, numSteps)).astype(int)
    # ln = np.zeros(numSteps)
    ms = np.zeros(numSteps)
    k = np.zeros(numSteps, int)
    ms[0] = m(v1, v2, phi[0])
    u2strich = np.zeros((numSteps, numSteps))
    for i in range(0, numSteps):
        u2strich[i, :] = F(u[i], u)

    k[0] = 0
    k[1] = 1
    k[2] = 2
    y = 0
    for n in range(0, numSteps - 1):
        if y >= 1:
            sums[o][y] = sums[o][y - 1]
        ms[n] = m(v1, v2, phi[n])
        ln[n + 1] = min(max(ln[n+1],0), n + 1)
        print('Step', n, 'of ', numSteps - 2, 'phi', phi[n])
        if ln[n] <= 0:
            phi[n + 1] = phi[n] * (1 - h * diff[n]) + phi[0] * h * diff[n] * ms[n] - h * diff[n] * ms[n] * phi[n]
            utilde1 = u[0:n + 2]
            numbneghist1 = 0
            sums[o][y] += 1
            y += 1
        else:
            utilde = np.union1d(u[0:n + 2], u2strich[n + 1, n - ln[n + 1] + 1:n + 1])
            utilde = np.where(utilde < u0, u0, utilde)
            utilde, numbneghist = np.unique(utilde, return_counts=True)
            if numbneghist[0] >= ln[n + 1] + 1:
                k[n + 1] = int(n + 1)

            else:
                k[n + 1] = np.nonzero(np.isclose(utilde, u2strich[n + 1, n + 1-numbneghist[0]]))[0]-1
            phi[n + 1] = phi[n] * (1 - h * diff[n]) + phi[0] * h * diff[n] * ms[n]
            sums[o][y] += 1
            term1 = 0
            for j in range(k[n], k[n + 1]):
                term1 -= h * diff[j] * ms[j] * phi[j]
                sums[o][y] += 1
            phi[n + 1] += term1
            term1 = 0
            for j in range(k[n + 1], k[n]):
                term1 -= h * diff[j] * ms[j] * phi[j]
                sums[o][y] += 1
            phi[n + 1] += term1
            term1 = 0
            lastequiU = k[n + 1]
            for j in range(k[n + 1], n + ln[n + 1] + 1 - numbneghist[0]):
                check = np.isclose(utilde[j], u, rtol=1e-08, atol=1e-15)
                sums[o][y] += 1
                if check.any():
                    # check if utilde[j] is one of the equaldistant points
                    whichj = np.nonzero(check)[0]
                    term1 -= (utilde[j] - utilde[j - 1]) * diff[whichj] * ms[whichj] * phi[whichj]
                    lastequiU = whichj
                else:
                    indexphi = F(u[n + 1], utilde[j]) - u0
                    if indexphi < 0:
                        indexphi = 0
                    else:
                        indexphi *= 1 / h
                    term1 -= (utilde[j] - utilde[j - 1]) * diff[lastequiU] * ms[lastequiU] * phi[int(np.rint(indexphi))]
            phi[n + 1] += term1
            term1 = 0
            lastequiU = k[n]
            for j in range(k[n], n + ln[n] - numbneghist1):
                check = np.isclose(utilde[j], u, rtol=1e-08, atol=1e-15)
                sums[o][y] += 1
                if check.any():
                    # check if utilde[j] is one of the equaldistant points
                    whichj = np.nonzero(check)[0]
                    term1 += (utilde1[j] - utilde1[j - 1]) * diff[whichj] * ms[whichj] * phi[whichj]
                    lastequiU = whichj
                else:
                    indexphi = F(u[n], utilde1[j]) - u0
                    if indexphi < 0:
                        indexphi = 0
                    else:
                        indexphi *= 1 / h
                    term1 += (utilde1[j] - utilde1[j - 1]) * diff[lastequiU] * ms[lastequiU] * phi[
                        int(np.rint(indexphi))]
            phi[n + 1] += term1
            utilde1 = utilde
            numbneghist1 = numbneghist[0]
            y+=1
    # error.append(np.linalg.norm(np.interp(deltat * np.arange(22001),np.exp(u),  phi) - phiarray1, ord=2))
    error.append(np.linalg.norm(phi - np.exp(-np.exp(u)), ord=np.inf))
    o+=1
    plt.plot(np.exp(u), phi)
plt.legend([1000, 3000] + hs)
rate = []
rate.append("-")
for i in range(1, len(error)):
    rate.append(-np.log(error[i] / error[i - 1]) / np.log(numStepss[i] / numStepss[i - 1]))
combined = []
for i in range(0, len(error)):
    combined.append([numStepss[i], error[i], rate[i]])
print(tabulate(combined, headers=['Elements', 'L2 error', 'EOC']))
plt.subplot(212)
plt.plot(range(0,numStepss[0]-1), sums[0][:])
plt.show()
