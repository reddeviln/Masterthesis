from FEniCSSimulation import *
import mshr

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return (near(x[1], -1) or near(x[1], 1)) and on_boundary

    def map(self, x, y):
        y[0] = x[0]
        y[1] = x[1] + 2

#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(0), Constant(0) , Constant(0.1), Constant(0.9))
nCells = 40
domain = mshr.Rectangle(Point(-1,1), Point(1,-1))
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element, constrained_domain=PeriodicBoundary()))


class DirichletBoundaryRight(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 1) and on_boundary


class DirichletBoundaryLeft(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], -1) and on_boundary




DB1 = DirichletBoundaryLeft()
dbc1 = DirichletBC(WithStressTensorSim.V[0].sub(2), Constant(0), DB1)
DB2 = DirichletBoundaryRight()
dbc2 = DirichletBC(WithStressTensorSim.V[0].sub(2), Constant(10), DB2)
# boundary conditions
WithStressTensorSim.bc = [dbc1, dbc2]


# initial condition

WithStressTensorSim.impose_initial_condition(Constant((0,0,0)))

# variational form
WithStressTensorSim.form_variational_problem_full2D(4, 0, 0.001)

# run
parameters["form_compiler"]["cpp_optimize"] = True
WithStressTensorSim.run_simulation_full(2,2000,"output/withstressPeriodic/solution.pvd")

