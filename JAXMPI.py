import time
import math
import os
import socket
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit

# ✅ Distributed initialization using MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# 🔧 MODIFIED: Use consistent env variable for coordinator address
coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "localhost:1234")  # 🔧 CHANGED
os.environ["JAX_PROCESS_ID"] = str(rank)
os.environ["JAX_NUM_PROCESSES"] = str(size)
os.environ["JAX_COORDINATOR"] = coordinator

# ✅ ADDED: Initialize distributed JAX early
if size > 1 and "JAX_DIST_INITIALIZED" not in os.environ:
    from jax import distributed
    distributed.initialize(
        coordinator_address=coordinator,  # 🔧 MODIFIED
        num_processes=size,
        process_id=rank
    )
    os.environ["JAX_DIST_INITIALIZED"] = "1"

# ✅ Log hardware
print("Starting JAX PJIT simulation...")
print(f"Rank: {rank}, Size: {size}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print("All visible devices:", jax.devices())
print("Local devices:", jax.local_devices())
print(f"JAX process {jax.process_index()} out of {jax.process_count()} is running on host {socket.gethostname()} with devices: {jax.local_devices()}")
print(f"JAX platform: {jax.default_backend()}")

# ✅ Simulation parameters
NX, NY = 40000, 40000
NSTEPS = 10000
omega = 1.7
u_max = 0.1
nu = (1 / omega - 0.5) / 3

# ✅ Lattice parameters
dtype = jnp.float32
w = jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype)
c = jnp.array([[0,1,0,-1,0,1,-1,-1,1], [0,0,1,0,-1,1,1,-1,-1]], dtype)

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

@jax.jit
def stream(g):
    shifts = [(0,0), (0,1), (1,0), (0,-1), (-1,0),
              (1,1), (1,-1), (-1,-1), (-1,1)]
    return jnp.stack([jnp.roll(g[i], shift=shifts[i], axis=(0,1)) for i in range(9)])

# ✅ Initialize grid
x = jnp.arange(NX) + 0.5
y = jnp.arange(NY) + 0.5
X, Y = jnp.meshgrid(x, y, indexing='ij')
u0 = u_max * jnp.sin(2 * jnp.pi * Y / NY)
rho0 = jnp.ones((NX, NY), dtype=dtype)
v0 = jnp.zeros_like(u0)
u_init = jnp.array([u0, v0])
f0 = equilibrium(rho0, u_init).astype(dtype)

# ✅ Set up sharding across all devices (multi-process)
devices = jax.devices()
num_devices = len(devices)
px = int(math.floor(math.sqrt(num_devices)))
while num_devices % px != 0:
    px -= 1
py = num_devices // px
print(f" Using 2D mesh shape: ({px}, {py})")

mesh = Mesh(mesh_utils.create_device_mesh((px, py)), axis_names=('x', 'y'))

with mesh:
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    f = jax.device_put(f0, sharding)

    def lbm_step(f):
        f = stream(f)
        f, _ = collide(f)
        return f

    lbm_step = pjit(
        lbm_step,
        in_shardings=P(None, 'x', 'y'),
        out_shardings=P(None, 'x', 'y'),
    )

    print(" Effective sharding of `f`: ", f.sharding)
    print(" `f` array is sharded across:", f.devices())

    start = time.time()
    for _ in range(NSTEPS):
        f = lbm_step(f)
    end = time.time()

# ✅ Performance metrics
elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

print(f"\n Rank {rank}: Ran on {num_devices} device(s): {blups:.3f} BLUPS")
print(f"  Elapsed time: {elapsed:.2f} seconds")
print(f" Domain size: NX={NX}, NY={NY}")
print(f" Number of steps: {NSTEPS}")
print(f"  Omega: {omega}, Viscosity: {nu:.4e}")
