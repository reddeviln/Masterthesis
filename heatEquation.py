import fenics as fe
from FEniCSSimulation import *
import matplotlib.pyplot as plt
import numpy

s = Vector()
s.init(2)
s[0] = 100
s[1] = 1
heatEquation = FEniCSSimulation(Constant(-1),Constant(s), Constant(1))
heatEquation.make_mesh(1, 10, 10)
heatEquation.register_dofs('P', 1)
heatEquation.register_dofs('P', 1, dim=2)



def boundary(x, on_boundary):
    return on_boundary


heatEquation.boundary_condition('Dirichlet', Constant(0), heatEquation.V[0], boundary)
heatEquation.impose_initial_condition(, 0)

heatEquation.form_variational_problem_heat()

heatEquation.run_simulation(2,10, "output/heatequation/solution.pvd")
