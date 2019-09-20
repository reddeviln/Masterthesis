from FEniCSSimulation import *
import mshr





#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(-5), Constant(0) , Constant(0), Constant(1))
nCells = 40
domain=mshr.Circle(Point(0,0),1)
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
StressTensorElement3 = FiniteElement('P', triangle, 1)
StressTensorElement4 = FiniteElement('P', triangle, 1)
StressTensorElement5 = FiniteElement('P', triangle, 1)
StressTensorElement6 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, StressTensorElement3, StressTensorElement4, StressTensorElement5, StressTensorElement6,VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element))


def boundary(x, on_boundary):
    return on_boundary




# boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant(0), WithStressTensorSim.V[0].sub(6), boundary)

# initial condition

WithStressTensorSim.impose_initial_condition(Constant((0,0,0,0,0,0,0)))

# variational form
parameters["form_compiler"]["cpp_optimize"] = True
WithStressTensorSim.form_variational_problem_full2D([1,3,0.1], 0, 1/1000)

# run
WithStressTensorSim.run_simulation_full(1,1000,"output/withstress/solution.pvd")

