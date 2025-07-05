from mpi4py import MPI
import os
import numpy as np
import time

# 🚀 Bind each MPI rank to a unique GPU before importing JAX
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ✅ Set visible GPU for this rank BEFORE importing jax
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

# ✅ Now import JAX after setting CUDA visibility
import jax
import jax.numpy as jnp

# 🖥️ Confirm GPU assignment
print(f"[rank {rank}] Using JAX devices: {jax.devices()}")

# ✅ Domain parameters
NX, NY = 400, 300
NSTEPS = 10000
omega = 1.7
u_max = 0.1
nu = (1 / omega - 0.5) / 3

# ✅ Divide domain among ranks (along X)
assert NX % size == 0
NXs = NX // size

# ✅ Lattice weights and directions
dtype = jnp.float32
w = jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype)
c = jnp.array([[0,1,0,-1,0,1,-1,-1,1], [0,0,1,0,-1,1,1,-1,-1]], dtype)

# ✅ Lattice routines
@jax.jit
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u * u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    return wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[jnp.newaxis, :, :])

@jax.jit
def collide(g):
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    return g + omega * (feq - g), u

def stream(g):
    shifts = [(0,0), (0,1), (1,0), (0,-1), (-1,0),
              (1,1), (1,-1), (-1,-1), (-1,1)]
    return jnp.stack([jnp.roll(g[i], shift=shifts[i], axis=(0,1)) for i in range(9)])

# ✅ MPI halo exchange
def mpi_halo(f):
    left = np.array(f[:, 0, :])
    right = np.array(f[:, -1, :])
    left_recv = np.empty_like(left)
    right_recv = np.empty_like(right)

    comm.Sendrecv(right, dest=(rank + 1) % size, recvbuf=left_recv, source=(rank - 1) % size)
    comm.Sendrecv(left, dest=(rank - 1) % size, recvbuf=right_recv, source=(rank + 1) % size)

    f = f.at[:, 0, :].set(left_recv)
    f = f.at[:, -1, :].set(right_recv)
    return f

@jax.jit
def step_local(f):
    f = stream(f)
    f, _ = collide(f)
    return f

# ✅ Initialize
x = jnp.arange(NX) + 0.5
y = jnp.arange(NY) + 0.5
X, Y = jnp.meshgrid(x, y)
u0 = u_max * jnp.sin(2 * jnp.pi * Y.T / NY)
rho0 = jnp.ones((NX, NY), dtype=dtype)
f0 = np.array(equilibrium(rho0, jnp.array([u0, jnp.zeros_like(u0)])))

# ✅ Extract slice for this rank
f_loc = f0[:, rank*NXs:(rank+1)*NXs, :]

# ✅ Main loop
start = time.time()
for _ in range(NSTEPS):
    f_loc = step_local(f_loc)
    f_loc = mpi_halo(f_loc)
end = time.time()

# ✅ BLUPS calculation
elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

total_blups = comm.reduce(blups, op=MPI.SUM, root=0)

if rank == 0:
    print(f"Ran on {size} GPUs: {total_blups:.3f} BLUPS")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Domain size: NX={NX}, NY={NY}")
    print(f"Number of steps: {NSTEPS}")
    print(f"Omega: {omega}, Viscosity: {nu:.4e}")

MPI.Finalize()
