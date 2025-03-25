import dolfinx
from mpi4py import MPI
from dolfinx import default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import (
    functionspace,
    Function,
    locate_dofs_topological,
    dirichletbc,
)
from dolfinx import log
from dolfinx.fem.petsc import LinearProblem
from ufl import TrialFunction, TestFunction, dx, grad, dot, ds
import ufl

from dolfinx.mesh import locate_entities_boundary, meshtags

import numpy as np


R_0 = 0.5
R1 = R_0 * 0.2
c = 1
c_0 = c * 0.8
I = 1  # total current
B0 = 1  # magnetic field
mu = 1  # permeability


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([c, R_0])], [50, 50]
)

V = functionspace(mesh, ("Lagrange", 1))
tdim = mesh.topology.dim


def delta_function(x, a, mod=np):
    return 1 / (a * mod.sqrt(mod.pi)) * mod.exp(-((x / a) ** 2))


a_param = 1 / 100


def flux_left(r, mod=np):
    return -I * delta_function(r - R1, a=a_param, mod=mod) / (2 * mod.pi * r)


def flux_top(z, mod=np):
    return +I * delta_function(z - c_0, a=a_param, mod=mod) / (2 * mod.pi * R_0)


# import matplotlib.pyplot as plt

# # z = np.linspace(0, c, 100)
# # plt.plot(z, flux_top(z))

# r = np.linspace(0, R_0, 100)
# plt.plot(r, flux_left(r))

# plt.show()

facets_top = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], R_0))
facets_left = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
facets_near_cathode = locate_entities_boundary(
    mesh,
    tdim - 1,
    lambda x: np.logical_and(
        np.isclose(x[1], R1, atol=0.01, rtol=0), np.isclose(x[0], 0)
    ),
)
dofs_near_cathode = locate_dofs_topological(V, tdim - 1, facets_near_cathode)

# create facet markers
facet_indices, facet_markers = [], []
for marker, facets in zip([1, 2], [facets_top, facets_left]):
    for facet in facets:
        facet_indices.append(facet)
        facet_markers.append(marker)

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
fdim = mesh.topology.dim - 1
facet_tag = meshtags(
    mesh, fdim, facet_indices[sorted_facets], facet_markers[sorted_facets]
)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)


u = Function(V)
v = TestFunction(V)
conductivity = dolfinx.fem.Constant(mesh, 0.01)

source_term = dolfinx.fem.Constant(mesh, -1.0)

x = ufl.SpatialCoordinate(mesh)
a = conductivity * dot(grad(u), grad(v)) * dx
L = (
    +flux_left(r=x[1], mod=ufl) * v * ds(2)
    + flux_top(z=x[0], mod=ufl) * v * ds(1)
    # + source_term * v * dx
)

constrain_cathode = dirichletbc(dolfinx.fem.Constant(mesh, -50.0), dofs_near_cathode, V)
bcs = [constrain_cathode]
# potential = Function(V)
# problem = LinearProblem(a, L, bcs=bcs, u=potential)
# uh = problem.solve()

F = 0
F += a
F += -L
problem = NonlinearProblem(F, u, bcs=bcs)
potential = u

solver = NewtonSolver(MPI.COMM_WORLD, problem)
# solver.convergence_criterion = "incremental"

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)

from dolfinx.io import VTXWriter

writer = VTXWriter(MPI.COMM_WORLD, "potential.bp", [potential], "BP5")
writer.write(0.0)
