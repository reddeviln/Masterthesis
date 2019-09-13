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
numStepss = [50,100,200,400,800]
# defining functions
def f(ui):
    return np.where(ui == -np.inf, 0, np.exp(ui))


def difff(fi):
    result = np.zeros(len(fi) - 1)
    for i in range(0, len(result)):
        result[i] = fi[i + 1] - fi[i]
    return result


def F(ui, uprime):
    return np.log(f(ui) - f(uprime))
error=[]
error_inf=[]
rate =[]
rate_inf =[]
for numSteps in numStepss:
    u = np.linspace(u0, np.log(T), numSteps)  # equaldistant points u first discretization
    fn = np.zeros(len(u) + 1)
    fn = f(np.concatenate(([-np.inf], u)))  # calculate f(u_i) but include f(-inf)
    diff_f = difff(fn)  # we often just need the difference in f(u_i) - f(u_{i-1}) so we calculate this in advance
    phi = np.zeros(numSteps)  # allocate solution array
    phi[0] = phi0
    h = u[1] - u[0]
    # calculate lstar which is how many intervals of the first discretization the second discretization spreads at the start

    lstar = int(np.ceil((u[1] - F(u[1], u[0])) / h))

    # calculate the area where the two discretizations are overlapping

    # find jstar which is the index of the first sst which lies in the interval [u_n,u_[n+1]]
    for j in range(1, lstar + 1):
        condition1 = u[lstar] - F(u[lstar], u[lstar - j]) <= h
        condition2 = u[lstar] - F(u[lstar], u[lstar - j + 1]) > h
        if condition1 and condition2:
            jstar = j
            break

    # put both discretizations together
    sst = np.concatenate((u[np.arange(1, lstar)], F(u[lstar], u[lstar - jstar:lstar])))

    perm = np.argsort(sst)
    sorted_sst = sst[perm]

    # remember which sst came from which discretization
    is_m_sst = perm < lstar - 1
    is_phi_sst = ~ is_m_sst
    count_m_sst = np.cumsum(is_m_sst)
    count_phi_sst = np.cumsum(is_phi_sst)
    N = 0
    M = 0
    if lstar == 1:
        print("We have l*=1")
        for n in range(0, numSteps - 1):
            print("Step", n, "of", numSteps - 1)
            N += (v1 * phi[n] + v2 * phi[n] ** 2) * diff_f[n]
            M += phi[n] * diff_f[n]
            # now defining coefficients for ax³+bx²+cx+d = 0
            a = -v2 * (diff_f[n + 1] - fn[n + 1])  # fn[n+1] is f(u_{n})
            b = -v1 * (diff_f[n + 1] - fn[n + 1]) - v2 * M
            c = -N - v1 * M - 1
            d = -M + phi[0] * (1 + N)
            # now coefficients for derivative bx² +cx +d = 0
            bdiff = 3 * a
            cdiff = 2 * b
            ddiff = c
            # now Newton iterations phi_{n+1}^{i+1} = phi_{n+1}^i - (ax³+bx²+cx+d)/(bdiff*x²+cdiff*x+ddiff)
            # starting with the previous step
            phi[n + 1] = phi[n]
            for i in range(0, 5):
                value = ((a * phi[n + 1] + b) * phi[n + 1] + c) * phi[n + 1] + d
                update = value / ((bdiff * phi[n + 1] + cdiff) * phi[n + 1] + ddiff)
                if abs(value) < 1e-12:
                    break
                else:
                    phi[n + 1] = phi[n + 1] - update
            print("Newton iteration finished in", i, "iterations (max. 4) with residual left:", value)
    else:

        print("We have l*=", lstar)
        S = 0
        M = 0
        O = (v2 * phi[0] + v1) * phi[0] * f(u0 - h * lstar)
        Q = phi[0] * f(u0 - h * (jstar + 1))
        for n in range(0, numSteps - 1):
            print("Step", n, "of", numSteps - 1)
            S += (v2 * phi[n] + v1) * phi[n] * diff_f[n]
            M += phi[n] * diff_f[n]
            O += (v2 * phi[max(n + 1 - lstar, 0)] + v1) * phi[max(n + 1 - lstar, 0)] * (
                        f(u0 + h * (n + 1 - lstar)) - f(u0 + h * (n - lstar)))
            Q += phi[max(n - jstar, 0)] * (f(u0 + h * (n - jstar)) - f(u0 + h * (n - 1 - jstar)))

            sum_ = 0

            for i in range(1, lstar + jstar - 2):
                sum_ += (v2 * phi[max(n + 2 - lstar + count_m_sst[i - 1], 0)] + v1) * phi[
                    max(n + 2 - lstar + count_m_sst[i - 1], 0)] \
                        * phi[n + 1 - count_phi_sst[i - 1]] * (
                                    f((n + 1 - lstar) * h + sorted_sst[i]) - f((n + 1 - lstar) * h + sorted_sst[i - 1]))
            # coefficients for ax²+bx+c =0
            tmp = phi[max(n + 1 - jstar, 0)] * (diff_f[n + 1] - f(u0 + (n - jstar + 1) * h))
            assert (tmp > 0.0)
            assert (diff_f[n + 1] - f(u0 + (n - lstar + 1) * h) > 0.0)
            a = -v2 * Q + v2 * phi[0] * diff_f[n + 1] - v2 * tmp
            b = -1 - v1 * Q + v1 * phi[0] * diff_f[n + 1] - diff_f[n + 1] - O - (v2 * phi[max(n + 2 - lstar, 0)] + v1) * \
                phi[max(n + 2 - lstar, 0)] * (diff_f[n + 1] - f(u0 + (n - lstar + 1) * h)) - v1 * tmp
            c = phi[0] * (1 + S) - M - sum_
            # now coefficients for derivative bx +c = 0
            bdiff = 2 * a
            cdiff = b
            # now Newton iterations
            # starting with the previous step
            phi[n + 1] = phi[n]

            for i in range(0, 5):
                value = (a * phi[n + 1] + b) * phi[n + 1] + c
                update = value / (bdiff * phi[n + 1] + cdiff)
                # assert (update <= 0.0)
                if abs(value) < 1e-12:
                    break
                else:
                    phi[n + 1] = phi[n + 1] - update
            print("Newton iteration finished in", i, "iterations (max. 4) with residual left:", value)
    error.append(np.linalg.norm(phiarray1-np.interp(deltat*np.arange(5001),np.exp(u),phi)))
    error_inf.append(np.linalg.norm(phiarray1 - np.interp(deltat * np.arange(5001), np.exp(u), phi),ord = np.inf))
rate = []
rate.append('-')
rate_inf.append('-')
n = len(error) -1
for i in range(1, n+1):
    rate.append(-np.log(error[i]/error[i-1])/np.log(numStepss[i]/numStepss[i-1]))
    rate_inf.append(-np.log(error_inf[i] / error_inf[i - 1]) / np.log(numStepss[i] / numStepss[i - 1]))
combined = []
for i in range (0,n+1):
    combined.append([numStepss[i], error[i], rate[i], error_inf[i], rate_inf[i]])
print(tabulate(combined, headers =['#Elements', 'L2 error', 'EOC', 'L-inf error', 'EOC']))
plt.plot(np.exp(u), phi)
plt.plot(deltat*np.arange(5001),phiarray1)
plt.legend(['l*alg','n^2'])
plt.show()
