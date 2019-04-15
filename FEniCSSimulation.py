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

    def getSigma(self, C1, C2, C3, Lambda):
        """get sigma from C"""

        return 0.5/Lambda*as_vector([C1-1, C2-1, C3-1])



    def form_variational_problem_onebyone(self,Lambda):
        """define the variational problem for the equations with stress tensor"""
        dt = Constant(0.1)
        Lambda = Constant(Lambda)
        C1, C2, C3, u = TrialFunctions(self.V[0])
        CV1, CV2, CV3, v = TestFunctions(self.V[0])
        self.F = [0 for x in range(4)]
        self.u_k = Function(self.V[0])
        un1, un2, un3, un4 = split(self.u_n)
        uk1, uk2, uk3, uk4 = split(self.u_k)

        self.F[0] = u * v * dx - un4 * v * dx + dt*(self.gradP * v * dx\
             + dot(self.getSigma(uk1, uk2, uk3,  Lambda), grad(v)) * dx)\
             + Constant(self.mu) * dot(grad(u), grad(v)) * dx \
             + (C1 - un1 + 1 / Lambda * (C1 - 1)) * CV1 * dx\
             + (C2 - un2 + 1 / Lambda * (C2 - 1)) * CV2 * dx \
             + (C3 - un3 + 1 / Lambda * (C3 - 1)) * CV3 * dx \
             - dt *(1/Lambda*u.dx(0) * CV1 * dx \
             + 1 / Lambda * u.dx(1) * CV2 * dx + 2 * (u.dx(0) * C1 + u.dx(1) * C2) * CV3 * dx)



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
        self.u_k = self.u_n
        t = 0
        tol = tolerance
        eps = 1.0
        iter = 0
        for n in range(num_steps):
            t += dt

            while eps > tol and iter < maxiter:

                #A = assemble(lhs(self.F[0]))
                #b = assemble(rhs(self.F[0]))
                #[bcu.apply(A) for bcu in self.bc]
                #[bcu.apply(b) for bcu in self.bc]
                solve(self.F[0] == 0, U, self.bc)
                diff = np.abs((U.vector() - self.u_k.vector()).max())
                eps = diff
                self.u_k.assign(U)
                print('iter=%d: norm=%g' % (iter, eps))
                solve(lhs(self.F[1]) == rhs(self.F[1]), U, self.bc)
                diff = np.abs((U.sub(0).vector().get_local() - self.u_k.sub(0).vector().get_local()))
                eps = diff
                print('iter=%d: norm=%g' % (iter, eps))
                self.u_k.assign(U)
                solve(lhs(self.F[2]) == rhs(self.F[2]), U, self.bc)
                self.CFunction, self.UFunction = split(U)
                diff = np.abs((U.vector() - self.u_k.vector()).max())
                eps = diff
                print('iter=%d: norm=%g' % (iter, eps))
                solve(lhs(self.F[3]) == rhs(self.F[3]), U, self.bc)
                self.u_k.assign(U)
                diff = np.abs((U.vector() - self.u_k.vector()).max())
                eps = diff
                print('iter=%d: norm=%g' % (iter, eps))

                self.u_k.assign(U)
                iter += 1

            vtkfile << (U.sub(0), t)
            self.u_n.assign(U)




