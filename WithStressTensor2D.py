from FEniCSSimulation import *
import mshr
import matplotlib.pyplot as plt


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

    def map(self, x, y):
        y[0] = x[0] - 2 * sqrt(9)
        y[1] = x[1]


#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(0), Constant(0) , Constant(1))
nCells = 15
domain=mshr.Rectangle(Point(0,-sqrt(9)),Point(2*sqrt(9),sqrt(9)))
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element, constrained_domain=PeriodicBoundary()))


def boundary1(x, on_boundary):
    return on_boundary and near(x[1],sqrt(9))


def boundary2(x, on_boundary):
    return on_boundary and  near(x[1],-sqrt(9))




# boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant(10), WithStressTensorSim.V[0].sub(2), boundary1)
WithStressTensorSim.boundary_condition('Dirichlet', Constant(0),WithStressTensorSim.V[0].sub(2), boundary2)

# initial condition
WithStressTensorSim.impose_initial_condition(Constant((0,0,0)),0)

# variational form
WithStressTensorSim.form_variational_problem_full2D(1)

# run
WithStressTensorSim.run_simulation_full(2,1750,"output/withstress/solution.pvd",1.0E-10,25)

