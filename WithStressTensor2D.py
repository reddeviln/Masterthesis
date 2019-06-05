from FEniCSSimulation import *
import mshr





#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(-5), Constant(0) , Constant(0), Constant(0.9))
nCells = 95
domain=mshr.Circle(Point(0,0),sqrt(0.9))
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element))


def boundary(x, on_boundary):
    return on_boundary




# boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant(0), WithStressTensorSim.V[0].sub(2), boundary)

# initial condition

WithStressTensorSim.impose_initial_condition(Constant((0,0,0)))

# variational form
WithStressTensorSim.form_variational_problem_full2D(1, 0)

# run
parameters["form_compiler"]["cpp_optimize"] = True
WithStressTensorSim.run_simulation_full(0.02,2000,"output/withstress/solution.pvd",1.0E-10,25)

