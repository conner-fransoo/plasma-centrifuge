class NonlinearProblemWithCallback(NonlinearProblem):
    """Problem for the DOLFINx NewtonSolver with an external callback.

    It lets `NewtonSolver` to run an additional routine `external_callback`
    before vector and matrix assembly. This may be useful to perform additional
    calculations at each Newton iteration. In particular, external operators
    must be evaluated via this routine.
    """
    from typing import Callable, Optional

    def __init__(
        self,
        F: ufl.form.Form,
        u: fem.function.Function,
        bcs: list[fem.bcs.DirichletBC] = [],
        J: ufl.form.Form = None,
        form_compiler_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        external_callback: Optional[Callable] = lambda: None,
    ):
        super().__init__(F, u, bcs, J, form_compiler_options, jit_options)

        self.u = u
        self.external_callback = external_callback

    def form(self, x: PETSc.Vec) -> None:
        """This function is called before the residual or Jacobian is
        computed. This is usually used to update ghost values, but here
        we also use it to evaluate the external operators.

        Args:
           x: The vector containing the latest solution
        """
        # The following line is from the standard NonlinearProblem class
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # We also copy the current solution values and propagate to the next step
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        # The external operators are evaluated here
        self.external_callback()

from typing import Callable, Optional

class PETScNonlinearProblem:
    """Defines a nonlinear problem for `PETSc.SNES`.

    It lets `PETSc.SNES` to run an additional routine `external_callback`
    before vector and matrix assembly. This may be useful to perform additional
    calculations at each Newton iteration. In particular, external operators
    must be evaluated via this routine.
    """

    def __init__(
        self,
        u: fem.function.Function,
        F: ufl.form.Form,
        J: Optional[ufl.form.Form] = None,
        bcs: list[fem.bcs.DirichletBC] = [],
        external_callback: Optional[Callable] = lambda: None,
    ):
        self.u = u
        self.F_form = fem.form(F)
        if J is None:
            V = self.u.function_space
            J = ufl.derivative(self.F_form, self.u, ufl.TrialFunction(V))
        self.J_form = fem.form(J)
        self.bcs = bcs
        self.external_callback = external_callback

    def F(self, snes: PETSc.SNES, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        snes: the snes object
        x: Vector containing the latest solution.
        b: Vector to assemble the residual into.
        """
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.scatter_forward()

        # Call external functions, e.g. evaluation of external operators
        self.external_callback()

        with b.localForm() as b_local:
            b_local.set(0.0)
        fem.petsc.assemble_vector(b, self.F_form)

        fem.petsc.apply_lifting(b, [self.J_form], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x: PETSc.Vec, A: PETSc.Mat, P: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        x: Vector containing the latest solution.
        A: Matrix to assemble the Jacobian into.
        """
        A.zeroEntries()
        fem.petsc.assemble_matrix(A, self.J_form, self.bcs)
        A.assemble()
