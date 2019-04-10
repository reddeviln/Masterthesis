from fenics import *
import mshr
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

    def form_variational_problem(self):
        """define the variational problem"""
        dt = 0.1
        sigma_int = interpolate(self.sigma, self.V[1])
        F = self.u[0]*self.v[0]*dx+self.mu*dt*dot(grad(self.u[0]),grad(self.v[0]))*dx-(self.u_n*self.v[0])*dx\
            +dt*(self.gradP*self.v[0])*dx+ dt*dot(sigma_int,grad(self.v[0]))*dx
            #
        self.a = lhs(F)
        self.L = rhs(F)

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
