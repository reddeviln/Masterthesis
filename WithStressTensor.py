from FEniCSSimulation import *
import mshr
import matplotlib.pyplot as plt

#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(3), Constant(0) , Constant(0.1))
nCells = 4
domain=mshr.Cylinder(Point(0,0,0),Point(2,0,0),1.,1.)
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh
# dofs
StressTensorElement1 = FiniteElement('P', tetrahedron, 1)
StressTensorElement2 = FiniteElement('P', tetrahedron, 1)
StressTensorElement3 = FiniteElement('P', tetrahedron, 1)
VelocityElements = FiniteElement('P', tetrahedron, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, StressTensorElement3, VelocityElements)
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element))


def boundary(x, on_boundary):
    return on_boundary and not (near(x[0],0) or near(x[0],2))


# boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant(0), WithStressTensorSim.V[0].sub(3), boundary)

# initial condition
WithStressTensorSim.impose_initial_condition(Constant((0,0,1,0)),0)

# variational form
WithStressTensorSim.form_variational_problem_onebyone(2)

# run
WithStressTensorSim.run_simulation_onebyone(2,10,"output/withstress/solution.pvd",1.0E-5,25)

