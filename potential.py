import dolfinx
from mpi4py import MPI
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import (
    functionspace,
    Function,
    locate_dofs_topological,
    dirichletbc,
)
from dolfinx import log
from ufl import TestFunctions, dx, grad, dot, ds
import ufl
from basix.ufl import element
from dolfinx.mesh import locate_entities_boundary, meshtags

import numpy as np


R_0 = 6e-2
R1 = R_0 * 0.2
c = 4 * R_0
c_0 = c * 0.7
I = 12  # total current
B0 = 0.13  # magnetic field
mu = 1  # permeability
viscosity = 3  # viscosity Pa s
sigma = 700  # conductivity S/m
sigma_perp = sigma


mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([c, R_0])], [250, 100]
)


V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,)))
tdim = mesh.topology.dim


def delta_function(x, a, mod=np):
    return 1 / (a * mod.sqrt(mod.pi)) * mod.exp(-((x / a) ** 2))


a_param = 1 / 100


def flux_left(r, mod=np):
    return -I * delta_function(r - R1, a=a_param, mod=mod) / (2 * mod.pi * r)


def flux_top(z, mod=np):
    return +I * delta_function(z - c_0, a=a_param, mod=mod) / (2 * mod.pi * R_0)


facets_top = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], R_0))
facets_left = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
facets_near_cathode = locate_entities_boundary(
    mesh,
    tdim - 1,
    lambda x: np.logical_and(
        np.isclose(x[1], R1, atol=0.01, rtol=0), np.isclose(x[0], 0)
    ),
)
dofs_near_cathode = locate_dofs_topological(V.sub(0), tdim - 1, facets_near_cathode)
dofs_left = locate_dofs_topological(V.sub(1), tdim - 1, facets_left)
dofs_top = locate_dofs_topological(V.sub(1), tdim - 1, facets_top)

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

# variational formulation
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

u = Function(V)
phi, v_theta = ufl.split(u)
phi_v, v_theta_test = TestFunctions(V)

source_term = dolfinx.fem.Constant(mesh, -1.0)

x = ufl.SpatialCoordinate(mesh)

F = 0

# potential
F += sigma * dot(grad(phi), grad(phi_v)) * dx  # TODO make this cylindrical
F += flux_left(r=x[1], mod=ufl) * phi_v * ds(2)
F += flux_top(z=x[0], mod=ufl) * phi_v * ds(1)
F += -B0 / x[1] * (x[1] * v_theta).dx(1) * phi_v * dx

# velocity
F += (
    viscosity * dot(grad(v_theta), grad(v_theta_test)) * dx
)  # TODO make this cylindrical
F += -sigma_perp * B0 * (-phi.dx(1) + v_theta * B0) * v_theta_test * dx


# we constrain the annode to around -50 because the problem is ill-posed
constrain_cathode = dirichletbc(
    dolfinx.fem.Constant(mesh, -50.0), dofs_near_cathode, V.sub(0)
)
non_slip_left = dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_left, V.sub(1))
non_slip_top = dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_top, V.sub(1))
bcs = [non_slip_left, non_slip_top, constrain_cathode]

problem = NonlinearProblem(F, u, bcs=bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(u)

from dolfinx.io import VTXWriter

potential, velocity = u.split()
potential = potential.collapse()
velocity = velocity.collapse()

writer = VTXWriter(MPI.COMM_WORLD, "potential.bp", [potential], "BP5")
writer.write(0.0)


writer = VTXWriter(MPI.COMM_WORLD, "velocity.bp", [velocity], "BP5")
writer.write(0.0)
