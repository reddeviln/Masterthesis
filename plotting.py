import matplotlib.pylab as plt

numElem=[10,20,40,80,160]
error = [0.0597072, 0.0179654, 0.00538169, 0.00145236, 0.000381656]
plt.loglog(numElem, error)
plt.title("L2 convergence")
plt.xlabel("# Elements per radius")
plt.ylabel("L2 error")
plt.show()