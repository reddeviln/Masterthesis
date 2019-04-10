from fenics import *
import mshr
import numpy as np
class FEniCSSimulation:
    """Lowlevelclass for using Non-Newtonian Fluids with Navier-Stokes"""

    def __init__(self, gradP_in, sigma_in, mu_in):
        """Constructor"""

        self.gradP = gradP_in
        self.sigma = sigma_in
        self.mu = mu_in
        self.bc = []
        self.V = []
        self.u = []
        self.v = []
        self.mesh = None
        self.a =[]

    def make_mesh(self, KindOfMesh, nCellsX, nCellsY):
        """generates the mesh  and returns it"""

        #KindOfMesh = 0 => UnitSquareMesh
        #KindOfMesh = 1 => CircleMesh

        if (KindOfMesh == 0):
            mesh = UnitSquareMesh(nCellsX , nCellsY)
        elif (KindOfMesh ==1):
            circle = mshr.Circle(Point(0,0),1)
            mesh = mshr.generate_mesh(circle,nCellsX)
        else:
            print("KindOfMesh has value", KindOfMesh, "which is not valid.")
            exit(1)
        self.mesh=mesh


    def register_dofs(self, TypeOfFunctions, degree, **dim):
        """define all the ansatzfunctions and function spaces and saves them in self"""

        if 'dim' in dim:
            V = VectorFunctionSpace(self.mesh,TypeOfFunctions, degree, dim=dim['dim'])
        else:
            V = FunctionSpace(self.mesh, TypeOfFunctions, degree)

        self.u.append(TrialFunction(V))
        self.v.append(TestFunction(V))
        self.V.append(V)

    def boundary_condition(self, TypeOfBoundary, expression, space, boundary):
        """define different boundary conditions (iteratively by calling this method as often as needed"""

        if TypeOfBoundary == 'Dirichlet':
            self.bc.append(DirichletBC(space, expression, boundary))

    def impose_initial_condition(self, expression, whichSpace):
        """impose the initial condition for the problem"""
        self.u_n = interpolate(expression, self.V[whichSpace])

    def form_variational_problem_heat(self):
        """define the variational problem for the heat equation with additions"""
        dt = 0.1
        sigma_int = interpolate(self.sigma, self.V[1])
        F = self.u[0]*self.v[0]*dx+self.mu*dt*dot(grad(self.u[0]),grad(self.v[0]))*dx-(self.u_n*self.v[0])*dx\
            +dt*(self.gradP*self.v[0])*dx+ dt*dot(sigma_int,grad(self.v[0]))*dx
        self.a = lhs(F)
        self.L = rhs(F)

    def getSigma(self, C, dim):
        """get sigma from C"""
        return C-Constant((1,1,1))



    def form_variational_problem_full(self,Lambda):
        """define the variational problem for the equations with stress tensor"""
        dt = Constant(0)
        Lambda=2
        C, u = TrialFunctions(self.V[0])
        CV, v = TestFunctions(self.V[0])
        F = [0 for x in range(4)]
        self.a = [0 for x in range(4)]
        self.L = [0 for x in range(4)]
        F[0] = u*v*dx-self.u_n[3]*v*dx+dt*(self.gradP*v*dx\
            + dot(self.getSigma(C, 3),grad(v))*dx)\
            + Constant(self.mu)*dot(grad(u),grad(v))*dx
        F[1] = (C[0]-self.u_n[0]+1/Lambda*(C[0]-1))*CV[0]*dx+dt/Lambda*u*v.dx(1)*dx
        F[2] = (C[1] - self.u_n[1] + 1 / Lambda * (C[1] - 1)) * CV[1] * dx + dt / Lambda * u * v.dx(2) * dx
        F[3] = 2*dot(grad(u),C)*CV[2]*dx

        self.a[0] = lhs(F[0])
        self.a[1] = lhs(F[1])
        self.a[2] = lhs(F[2])
        self.a[3] = lhs(F[3])

        self.L[0] = rhs(F[0])
        self.L[1] = rhs(F[1])
        self.L[2] = rhs(F[2])
        self.L[3] = rhs(F[3])
        self.a = np.asarray(self.a)


    def run_simulation(self, T_end, num_steps,filename):
        """runs the actual simulation"""

        dt = T_end/num_steps
        vtkfile = File (filename)
        u = Function(self.V[0])
        t = 0


        for n in range(num_steps):
            t += dt
            A = assemble(self.a)
            b = assemble(self.L)
            [bcu.apply(A) for bcu in self.bc]
            [bcu.apply(b) for bcu in self.bc]
            solve(A, u.vector(),b)
            vtkfile << (u,t)
            self.u_n.assign(u)
