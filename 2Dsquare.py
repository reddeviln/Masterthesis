from FEniCSSimulation import *
import mshr

class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return (near(x[0], -1) or near(x[0], 1)) and on_boundary

    def map(self, x, y):
        y[0] = x[0] + 2
        y[1] = x[1]

#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(-3), Constant(0) , Constant(0), Constant(1))
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


class DirichletBoundaryTop(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1) and on_boundary


class DirichletBoundaryBottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], -1) and on_boundary




DB1 = DirichletBoundaryTop()
dbc1 = DirichletBC(WithStressTensorSim.V[0].sub(2), Constant(0), DB1)
DB2 = DirichletBoundaryBottom()
dbc2 = DirichletBC(WithStressTensorSim.V[0].sub(2), Constant(0), DB2)
# boundary conditions
WithStressTensorSim.bc = [dbc1, dbc2]


# initial condition

WithStressTensorSim.impose_initial_condition(Constant((0,0,0)))

Tend = 12
dt = 1.25e-3
numsteps = 9600
# variational form
WithStressTensorSim.form_variational_problem_full2D(1, 0, dt)

# run
parameters["form_compiler"]["cpp_optimize"] = True
WithStressTensorSim.run_simulation_full(Tend,numsteps,"output/2Dsquare/solution1E.pvd")

