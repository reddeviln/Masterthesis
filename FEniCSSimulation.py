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



    def form_variational_problem_onebyone(self,Lambda):
        """define the variational problem for the equations with stress tensor"""
        dt = Constant(0)
        Lambda = Constant(Lambda)
        self.C, self.u1 = TrialFunctions(self.V[0])
        self.CV, self.v1 = TestFunctions(self.V[0])
        self.F = [0 for x in range(4)]

        self.F[0] = self.u1 * self.v1 * dx-self.u_n[3] * self.v1* dx + dt*(self.gradP * self.v1 * dx\
             + dot(self.getSigma(self.C, 3), grad(self.v1)) * dx)\
             + Constant(self.mu) * dot(grad(self.u1), grad(self.v1)) * dx
        ones = interpolate(Constant((1,1,1,1)), self.V[0])
        self.F[1] = (self.C[0] - self.u_n.vector()[0] + 1 / Lambda *(self.C[0])- Constant(1)) * self.CV[0] * dx + dt / Lambda * self.u1 * self.CV[0].dx(1) * dx
        self.F[2] = (self.C[1] - self.u_n[1] + 1 / Lambda * (self.C[1] - 1)) * self.CV[1] * dx + dt / Lambda * self.u1 * self.CV[1].dx(2) * dx
        self.F[3] = (self.C[2] - self.u_n[2] + 1 / Lambda * (self.C[2] - 1)) * self.CV[2] * dx - 2 * dt * (self.u1.dx(0)*self.C[0] + self.u1.dx(1)*self.C[1]) * self.CV[2] * dx

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

    def run_simulation_onebyone(self, T_end, num_steps,filename, tolerance, maxiter):
        """run the actual simulation for with fixpoint iteration"""

        dt = T_end / num_steps
        vtkfile = File(filename)
        U = Function(self.V[0])
        u_k = self.u_n[3]
        t = 0
        tol = tolerance
        eps = 1
        iter = 0
        for n in range(num_steps):
            t += dt

            while eps > tol and iter < maxiter:

                solve(self.F[1] == 0, U, self.bc)
                C, u = split(U)
                solve(self.F[2], U.vector())
                C, u = split(U)
                #self.a[3].u = u
                A3 = assemble(self.a[3])
                b3 = assemble(self.L[3])
                [bcu.apply(A3) for bcu in self.bc]
                [bcu.apply(b3) for bcu in self.bc]
                solve(A3, U.vector(), b3)
                C, u = split(U)
                #self.a[0].u = u
               # self.a[0].C = C
                A0 = assemble(self.a[0])
                b0 = assemble(self.L[0])
                [bcu.apply(A0) for bcu in self.bc]
                [bcu.apply(b0) for bcu in self.bc]
                solve(A0, U.vector(), b0)
                C, u = split(U)
                diff = u.vector().array()- u_k.vector().array()
                eps = np.linalg.norm(diff, ord=np.Inf)
                print('iter=%d: norm=%g' % (iter, eps))
                u_k.assign(u)
                iter += 1

            vtkfile << (u, t)
            self.u_n.assign(U)




