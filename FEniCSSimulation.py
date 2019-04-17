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

    def  getSigma3D(self, C1, C2, C3, Lambda):
        """get sigma from C"""

        return 0.5/Lambda*as_vector([C1-1, C2-1, C3-1])

    def  getSigma2D(self, C1, C2, Lambda):
        """get sigma from C"""

        return 0.8*as_vector([C1-1, C2-1])

    def form_variational_problem_full3D(self,Lambda):
        """define the variational problem for the equations with stress tensor"""
        dt = Constant(0.1)
        Lambda = Constant(Lambda)
        self.U = Function(self.V[0])
        C1, C2, C3, u = split(self.U)
        CV1, CV2, CV3, v = TestFunctions(self.V[0])
        un1, un2, un3, un4 = split(self.u_n)

        self.F = u * v * dx - un4 * v * dx + dt*(self.gradP * v * dx\
            + dot(self.getSigma3D(un1, un2, un3,  Lambda), grad(v)) * dx\
            + Constant(self.mu) * dot(grad(u), grad(v)) * dx) \
            + (C1 - un1 + dt / Lambda * (C1 - 1)) * CV1 * dx\
            + (C2 - un2 + dt / Lambda * (C2 - 1)) * CV2 * dx \
            + (C3 - un3 + dt / Lambda * (C3 - 1)) * CV3 * dx \
            - dt *(1/Lambda*u.dx(0) * CV1 * dx \
            + 1 / Lambda * u.dx(1) * CV2 * dx + 2 * (u.dx(0) * C1 + u.dx(1) * C2) * CV3 * dx)

    def form_variational_problem_full2D(self,Lambda):
        """define the variational problem for the equations with stress tensor"""
        dt = Constant(0.1)
        Lambda = Constant(Lambda)
        self.U = Function(self.V[0])
        C1, C2, u = split(self.U)
        CV1, CV2, v = TestFunctions(self.V[0])
        un1, un2, un3 = split(self.u_n)

        self.F = u * v * dx - un3 * v * dx + dt*(self.gradP * v * dx\
            + dot(self.getSigma2D(un1, un2, Lambda), grad(v)) * dx\
            + Constant(self.mu) * dot(grad(u), grad(v)) * dx) \
            + (C1 - un1 + dt / Lambda * (C1 - 1)) * CV1 * dx\
            + (C2 - un2 + dt / Lambda * (C2 - 1)) * CV2 * dx \
            - dt *(1 / Lambda * u.dx(0) * CV1 * dx \
            + 1 / Lambda * u.dx(1) * CV2 * dx)

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

    def run_simulation_full(self, T_end, num_steps,filename, tolerance, maxiter):
        """run the actual simulation for with newton iteration"""

        dt = T_end / num_steps
        vtkfile = File(filename)
        t = 0
        tol = tolerance
        eps = 1.0
        iter = 0
        for n in range(num_steps):
            t += dt

            solve(self.F == 0, self.U, self.bc, solver_parameters={"newton_solver": {"absolute_tolerance": tol, "relative_tolerance": 1e-16}})
            print("time: ",t)
            vtkfile << (self.U.sub(2), t)
            self.u_n.assign(self.U)




