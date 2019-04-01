import fenics as fe
from FEniCSSimulation import *
import matplotlib.pyplot as plt

heatEquation = FEniCSSimulation(0, Identity, 1)
outmesh = heatEquation.make_mesh(1, 10, 10)
fe.plot(outmesh)
plt.show()
