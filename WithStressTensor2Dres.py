from FEniCSSimulation import *
import mshr
import matplotlib.pyplot as plt




#init and mesh
WithStressTensorSim =[]
numElem =[]
Solutions =[]
Spaces = []

StressTensorElement1 = FiniteElement('P', triangle, 1)
StressTensorElement2 = FiniteElement('P', triangle, 1)
VelocityElements = FiniteElement('P', triangle, 1)
element = MixedElement(StressTensorElement1, StressTensorElement2, VelocityElements)
print('Starting up')
for i in range(0,5):
    nCells = 5*pow(2,i+1)
    print('N = ',nCells)
    domain = mshr.Circle(Point(0, 0), 1, nCells)
    mesh = mshr.generate_mesh(domain, nCells)
    WithStressTensorSim.append(FEniCSSimulation(Constant(2), Constant(0), Constant(0), Constant(0.9)))
    WithStressTensorSim[i].mesh = mesh
    # dofs
    WithStressTensorSim[i].V.append(FunctionSpace(WithStressTensorSim[i].mesh, element))
    t=0


    def boundary(x, on_boundary):
        return on_boundary


    u_I = Expression(('-2*x[0]*(exp(-(pow(x[0], 2) + pow(x[1], 2)-1))-1)', '-2*x[1]*(exp(-(pow(x[0], 2) + pow(x[1], 2)-1))-1)', 'exp(-(pow(x[0], 2) + pow(x[1], 2)-1))-1'), element=WithStressTensorSim[i].V[0].ufl_element())

    # boundary conditions
    WithStressTensorSim[i].boundary_condition('Dirichlet', Constant((0,0,0)), WithStressTensorSim[i].V[0], boundary)


    # initial condition
    WithStressTensorSim[i].impose_initial_condition(u_I)

    # variational form
    WithStressTensorSim[i].form_variational_problem_full2D(4, 1, 0.01)

    # run
    parameters["form_compiler"]["cpp_optimize"] = True
    WithStressTensorSim[i].run_simulation_full(1,100,"output/withstressRes/solution.pvd")
    numElem.append(nCells)
    Solutions.append(WithStressTensorSim[i].result)
    Spaces.append(WithStressTensorSim[i].V[0])
print('running postprocessing')
(error, rate) = run_postprocessing(u_I, Solutions, numElem, Spaces)

from tabulate import tabulate

n = len(rate)
combined = []
for i in range (0,n):
    combined.append([numElem[i], error[i], rate[i]])
print(tabulate(combined, headers =['#Elements per radius', 'L2 error', 'EOC']))
plt.loglog(numElem, error)
plt.title("Convergence in L2")
plt.xlabel("# Elements per diameter")
plt.ylabel("L2-error")
plt.show()

