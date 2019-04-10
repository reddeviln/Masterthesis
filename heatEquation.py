import fenics as fe
from FEniCSSimulation import *
import matplotlib.pyplot as plt
import numpy

s = Vector()
s.init(2)
s[0] = 0
s[1] = 1
heatEquation = FEniCSSimulation(Constant(0),Constant(s), Constant(1))
heatEquation.make_mesh(1, 10, 10)
heatEquation.register_dofs('P', 1)
heatEquation.register_dofs('P', 1, dim=2)



def boundary(x, on_boundary):
    return on_boundary


heatEquation.boundary_condition('Dirichlet', Constant(0), heatEquation.V[0], boundary)
heatEquation.impose_initial_condition(Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))',degree=2, a=5), 0)

heatEquation.form_variational_problem()

heatEquation.run_simulation(2,100)
