from FEniCSSimulation import *
import mshr
import matplotlib.pyplot as plt




#init and mesh
WithStressTensorSim =[]
numElem =[]
Solutions =[]
Spaces = []
domain = mshr.Circle(Point(0,0),sqrt(0.9))
StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
print('Starting up')
for i in range(0,2):
    nCells = 10*pow(2,i+1)
    print('N = ',nCells)
    mesh = mshr.generate_mesh(domain, nCells)
    WithStressTensorSim.append(FEniCSSimulation(Constant(-5), Constant(0), Constant(0), Constant(0.9)))
    WithStressTensorSim[i].mesh = mesh
    # dofs
    WithStressTensorSim[i].V.append(FunctionSpace(WithStressTensorSim[i].mesh, element))
    t=0


    def boundary(x, on_boundary):
        return on_boundary


    u_D = Expression(('-sin(2*pi*(x[0]-t))-cos(2*pi*(x[1]-t))','-sin(2*pi*(x[0]-t))-cos(2*pi*(x[1]-t))', 'sin(2*pi*(x[0]-t))+cos(2*pi*(x[1]-t))'), t=t, degree=4)
    # boundary conditions
    WithStressTensorSim[i].boundary_condition('Dirichlet', u_D, WithStressTensorSim[i].V[0], boundary)

    u_I = Expression(('-sin(2*pi*(x[0]))-cos(2*pi*(x[1]))','-sin(2*pi*(x[0]))-cos(2*pi*(x[1]))', 'sin(2*pi*(x[0]))+cos(2*pi*(x[1]))'), degree=4)

    # initial condition
    WithStressTensorSim[i].impose_initial_condition(u_I)

    # variational form
    WithStressTensorSim[i].form_variational_problem_full2D(1, 1)

    # run
    parameters["form_compiler"]["cpp_optimize"] = True
    WithStressTensorSim[i].run_simulation_full(1,100,"output/withstressRes/solution.pvd",u_D)
    numElem.append(nCells)
    Solutions.append(WithStressTensorSim[i].result)
    Spaces.append(WithStressTensorSim[i].V[0])
print('running postprocessing')
error, rate = run_postprocessing(u_D,Solutions, numElem, Spaces)

from tabulate import tabulate

n = len(rate)
combined = []
for i in range (0,n):
    combined.append([numElem[i], error[i], rate[0]])
print(tabulate(combined, headers =['#Elements per radius', 'L2 error', 'EOC']))


