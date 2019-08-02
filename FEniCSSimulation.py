from fenics import *
import mshr
import matplotlib.pyplot as plt
class FEniCSSimulation:
    """Lowlevelclass for using Non-Newtonian Fluids with Navier-Stokes"""

    def __init__(self, gradP_in, sigma_in, mu_in, mup_in):
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
        self.mup = mup_in

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

    def impose_initial_condition(self, expression):
        """impose the initial condition for the problem"""
        self.u_n = interpolate(expression, self.V[0])

    def form_variational_problem_heat(self):
        """define the variational problem for the heat equation with additions"""
        dt = 0.1

        sigma_int = interpolate(self.sigma, self.V[1])
        F = self.u[0]*self.v[0]*dx+self.mu*dt*dot(grad(self.u[0]),grad(self.v[0]))*dx-(self.u_n*self.v[0])*dx\
            +dt*(self.gradP*self.v[0])*dx+ dt*dot(sigma_int,grad(self.v[0]))*dx
        self.a = lhs(F)
        self.L = rhs(F)

    def residual(self, t, Lambda, CV1, CV2, v):
        """defines the residual for UCM"""

        x = SpatialCoordinate(self.mesh)
        normx = x[0]**2 + x[1]**2
        exy = exp(-(normx-1))
        R = - 4 * self.mup * (exy* (1 - normx) - 1) - self.gradP
        R23 =   - 2 * x * (exy * (1 - 1 / Lambda) + 1 / Lambda)
        result =  R * v * dx + dot(R23 , as_vector([CV1,CV2])) * dx

        return result


    def form_variational_problem_full2D(self, Lambda, residualon, dt):
        """define the variational problem for the equations with stress tensor"""
        t = 0
        Lambda = Constant(Lambda)
        self.U = TrialFunction(self.V[0])
        C1, C2, u = split(self.U)
        CV1, CV2, v = TestFunctions(self.V[0])
        un1, un2, un3 = split(self.u_n)

        if residualon == 1:
            self.F = u * v * dx - un3 * v * dx + dt*(self.gradP * v * dx + self.residual(t, Lambda, CV1, CV2, v) \
                + dot(self.mup*as_vector([C1, C2]), grad(v)) * dx\
                + Constant(self.mu) * dot(grad(u), grad(v)) * dx) \
                + (C1 - un1 + dt / Lambda * C1) * CV1 * dx\
                + (C2 - un2 + dt / Lambda * C2) * CV2 * dx \
                - dt * dot(grad(u), as_vector([CV1, CV2])) * dx
        elif residualon == 0:
            self.F = u * v * dx - un3 * v * dx + dt * (self.gradP * v * dx \
                     + dot(self.mup * as_vector([C1, C2]), grad(v)) * dx \
                     + Constant(self.mu) * dot(grad(u), grad(v)) * dx) \
                     + (C1 - un1 + dt / Lambda * C1) * CV1 * dx \
                     + (C2 - un2 + dt / Lambda * C2) * CV2 * dx \
                     - dt * (u.dx(0) * CV1 * dx \
                     + u.dx(1) * CV2 * dx)
        else:
            raise ValueError('residualon needs to be 0 or 1 but it was ', residualon)



    def form_variational_problem_UCM_DG(self, Lambda):
        """define the variational problem for UCM with DG for stability"""

        dt = Constant(0.1)
        self.n =FacetNormal(self.mesh)
        self.U =TrialFunction(self.V[0])
        self.C1, self.C2, self.u = split(self.U)
        self.phi = TestFunction(self.V[0])
        self.CV1, self.CV2, self.v = split(self.phi)
        un1, un2 , un3 = split(self.u_n)
        Flux = as_tensor([[- self.mup * self.C1, -self.mup * self.C2], [-1/Lambda * self.u, 0], [0, -1/Lambda * self.u]])
        jumpu = as_tensor([jump(self.U), jump(self.U)])
        ubnd  = as_tensor([self.U, self.U])
        numFlux = avg(Flux) - sqrt(self.mup) / ( 2 * sqrt(Lambda)) * jumpu.T
        numFluxbnd = Flux - sqrt(self.mup) / (2 * sqrt(Lambda)) * ubnd.T
        self.F = dot(self.U, self.phi) * dx - dot(self.u_n, self.phi) * dx + dt * (-inner(Flux, grad(self.phi)) * dx
                + (self.gradP * self.v + 1 / Lambda * (self.C1) * self.CV1 + 1 / Lambda * (self.C2) * self.CV2) * dx
                + dot(numFlux[0,:], jump(self.v, self.n)) * dS +  dot(numFlux[1,:], jump(self.CV1, self.n)) * dS
                + dot(numFlux[2,:], jump(self.CV2, self.n)) * dS + dot(numFluxbnd[0,:], self.v * self.n) * ds
                + dot(numFluxbnd[1,:], self.CV1 * self.n) * ds + dot(numFluxbnd[2,:], self.CV2 * self.n) * ds)

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

    def run_simulation_full(self, T_end, num_steps,filename):
        """run the actual simulation for with newton iteration"""

        dt = T_end / num_steps
        print(dt)
        vtkfile = File(filename)
        t = 0
        U = Function(self.V[0], name="velocity")
        a = lhs(self.F)
        L = rhs(self.F)
        A = assemble(a)
        U.assign(self.u_n)
        vtkfile << (U.sub(2), t)
        [bcu.apply(A) for bcu in self.bc]
        for n in range(num_steps):
            t += dt
            b = assemble(L)
            [bcu.apply(b) for bcu in self.bc]
            solve(A, U.vector(), b)
            print("time: ",t)
            if n % 10 == 0:
                vtkfile << (U.sub(2), t)
            self.u_n.assign(U)
        self.result = U


def run_postprocessing(u_D, listofSolutions, listNumElem, Spaces):
    """run error and convergence analysis"""

    error = []
    rate = []
    vtkexact = File("output/withstressRes/exact.pvd")

    for i in range(0, len(listofSolutions)):
        u_e = interpolate(u_D, Spaces[i])

        #error.append(errornorm(u_e.sub(2), listofSolutions[i].sub(2), degree_rise=5))
        error.append(norm(u_e.sub(2).vector()-listofSolutions[i].sub(2).vector(),'linf'))
    vtkexact << u_e.sub(2)
    n = len(error)

    #vtkdiff = File("output/withstressRes/diff.pvd")
    #u_diff = Function.copy(u_e)
    #u_diff.assign(u_diff-listofSolutions[len(listofSolutions)-1])
    #vtkdiff <<u_diff.sub(2)
    from math import log as ln
    rate.append(0)
    for i in range(1, n):
        rate.append(-ln(error[i]/error[i-1])/ln(listNumElem[i]/listNumElem[i-1]))

    return error, rate








