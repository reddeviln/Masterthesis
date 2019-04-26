from FEniCSSimulation import *
import mshr
import matplotlib.pyplot as plt



#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(-5), Constant(0) , Constant(1))
nCells = 40
domain=mshr.Rectangle(Point(0,-sqrt(9)),Point(2*sqrt(9),sqrt(9)))
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('DG', triangle , 0)
StressTensorElement2 = FiniteElement('DG', triangle, 0)
VelocityElements = FiniteElement('DG', triangle, 0)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element))


def boundary(x, on_boundary):
    return on_boundary and (near(x[1],sqrt(9)) or near(x[1],-sqrt(9)))


# boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant(0), WithStressTensorSim.V[0].sub(2), boundary)


# initial condition
WithStressTensorSim.impose_initial_condition(Constant((0,0,0)),0)

# variational form
WithStressTensorSim.form_variational_problem_UCM_DG(1)

# run
WithStressTensorSim.run_simulation_full(5,1000,"output/withstressDG/solution.pvd",1.0E-10,25)

