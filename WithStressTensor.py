from FEniCSSimulation import *
import mshr

#init and mesh
WithStressTensorSim = FEniCSSimulation(Constant(3), Constant(0) , Constant(0.01))
nCells = 10
domain=mshr.Cylinder(Point(0,0,0),Point(2,0,0),1,1)
mesh = mshr.generate_mesh(domain, nCells)
WithStressTensorSim.mesh = mesh

# dofs
StressTensorElements = VectorElement('P', tetrahedron, 1)
VelocityElements = FiniteElement('P', tetrahedron, 1)
element = StressTensorElements * VelocityElements
WithStressTensorSim.V.append(FunctionSpace(WithStressTensorSim.mesh, element))

#boundary conditions
WithStressTensorSim.boundary_condition('Dirichlet', Constant((0,0,1,0)), WithStressTensorSim.V[0], 'near(x[1],0)')
WithStressTensorSim.boundary_condition('Dirichlet', Constant((0,0,1,0)), WithStressTensorSim.V[0], 'near(x[1],1)')

#initial condition
WithStressTensorSim.impose_initial_condition(Constant((0,0,1,0)),0)

#variationalform
WithStressTensorSim.form_variational_problem_onebyone(2)

#run
WithStressTensorSim.run_simulation_onebyone(2,10,"output/withstress/solution.pvd",1.0E-5,25)

