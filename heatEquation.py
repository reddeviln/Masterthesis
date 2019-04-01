import fenics as fe
from FEniCSSimulation import *
import matplotlib.pyplot as plt

heatEquation = FEniCSSimulation(0, Identity, 1)
heatEquation.make_mesh(1, 10, 10)
heatEquation.register_dofs('P',2,2)

def boundary(x, on_boundary):
    return on_boundary

heatEquation.boundary_condition('Dirichlet', Constant(0),heatEquation.V[0],boundary)
heatEquation.form_variational_problem()