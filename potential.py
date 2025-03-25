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
from ufl import TestFunctions, dx, grad, dot, ds, inner
import ufl
from basix.ufl import element
from dolfinx.mesh import locate_entities_boundary, meshtags

import numpy as np


R_0 = 7.2
R1 = R_0 * 0.05
c = 4 * R_0
c_0 = c * 0.7
I = 100  # total current
B0 = 0.13  # magnetic field
mu = 1  # permeability
viscosity = 3  # viscosity Pa s
sigma = 700  # conductivity Ohm-1 m-1
beta_eff = 8
voltage = 58.0  # V


def sigma_perp_fun(sigma, beta_e):
    """Eq 6 from Winjakker Thesis"""
    return sigma * 1 / (1 + beta_e**2)


sigma_perp = sigma_perp_fun(sigma, beta_eff)


# def sigma_H(sigma, beta_e):
#     """Eq 6 from Winjakker Thesis"""
#     return sigma * beta_e / (1 + beta_e**2)


def hartmann_number(B0, R_0, sigma, mu):
    """Eq 1 from Winjakker Thesis"""
    return B0 * R_0 * np.sqrt(sigma / mu)


print(f"Hartmann number: {hartmann_number(B0, R_0, sigma, mu)}")
print(f"beta_eff: {beta_eff}")
print(f"sigma: {sigma}")
print(f"sigma_perp: {sigma_perp}")

mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, [np.array([0.0, 0.0]), np.array([c, R_0])], [250, 200]
)


V = functionspace(mesh, element("Lagrange", mesh.topology.cell_name(), 1, shape=(2,)))


def delta_function(x, a, mod=np):
    return mod.exp(-((x / a) ** 2)) / (a * mod.sqrt(mod.pi))


a_param = 1 / 100


def flux_left(r, mod=np):
    return -I * delta_function(r - R1, a=a_param, mod=mod) / (2 * mod.pi * r)


def flux_top(z, mod=np):
    return +I * delta_function(z - c_0, a=a_param, mod=mod) / (2 * mod.pi * R_0)


tdim = mesh.topology.dim
facets_top = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], R_0))
facets_left = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[0], 0))
facets_bot = locate_entities_boundary(mesh, tdim - 1, lambda x: np.isclose(x[1], 0))
facets_near_cathode = locate_entities_boundary(
    mesh,
    tdim - 1,
    lambda x: np.logical_and(np.isclose(x[1], R1, rtol=0.1), np.isclose(x[0], 0)),
)

dofs_near_cathode = locate_dofs_topological(V.sub(0), tdim - 1, facets_near_cathode)
dofs_left = locate_dofs_topological(V.sub(1), tdim - 1, facets_left)
dofs_top = locate_dofs_topological(V.sub(1), tdim - 1, facets_top)
dofs_bot = locate_dofs_topological(V.sub(1), tdim - 1, facets_bot)

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
z, r = x[0], x[1]

# potential
sigma_tensor = ufl.as_tensor([[sigma / sigma_perp, 0], [0, 1]])

F += inner(sigma_tensor * grad(phi), grad(phi_v)) * r * dx
F += flux_left(r=r, mod=ufl) * phi_v * ds(2)
F += flux_top(z=z, mod=ufl) * phi_v * ds(1)
F += -sigma_perp * B0 / r * (r * v_theta).dx(1) * phi_v * r * dx

# velocity
F += viscosity * dot(grad(v_theta), grad(v_theta_test)) * r * dx
F += sigma_perp * B0 * (-phi.dx(1) + v_theta * B0) * v_theta_test * r * dx


# we constrain the annode to around -50 because the problem is ill-posed
assert len(dofs_near_cathode) > 0, "No dofs found near cathode"
constrain_cathode = dirichletbc(
    dolfinx.fem.Constant(mesh, -voltage), dofs_near_cathode, V.sub(0)
)
non_slip_left = dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_left, V.sub(1))
non_slip_top = dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_top, V.sub(1))
non_slip_bot = dirichletbc(dolfinx.fem.Constant(mesh, 0.0), dofs_bot, V.sub(1))
bcs = [
    non_slip_left,
    non_slip_top,
    non_slip_bot,
    constrain_cathode,
]

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


# integrate current density

current_left = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(-sigma * potential.dx(0) * 2 * ufl.pi * r * ds(2))
)
current_top = dolfinx.fem.assemble_scalar(
    dolfinx.fem.form(-sigma_perp * potential.dx(1) * 2 * ufl.pi * r * ds(1))
)

print(f"Expected current: {I}")
print(f"Current left: {current_left}")
print(f"Current top: {current_top}")
