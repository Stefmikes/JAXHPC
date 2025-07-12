import time
import math
import os
import sys
import socket
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.experimental import mesh_utils, multihost_utils  
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit
from mpi4py import MPI

#  MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

#  JAX Distributed Environment Setup
coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS", "localhost:1234")
os.environ["JAX_PROCESS_ID"] = str(rank)
os.environ["JAX_NUM_PROCESSES"] = str(size)
os.environ["JAX_COORDINATOR"] = coordinator

if size > 1 and "JAX_DIST_INITIALIZED" not in os.environ:
    from jax import distributed
    distributed.initialize(
        coordinator_address=coordinator,
        num_processes=size,
        process_id=rank
    )
    os.environ["JAX_DIST_INITIALIZED"] = "1"

print("Starting JAX PJIT simulation...")
print(f"Rank: {rank}, Size: {size}")
print("All visible devices:", jax.devices())
print("Local devices:", jax.local_devices())
print(f"Process {jax.process_index()} on {socket.gethostname()} using {jax.local_devices()}")
print(f"JAX backend: {jax.default_backend()}")

# ✅ Simulation parameters
NX, NY = 300, 300
NSTEPS = 2000
omega = 0.16
u_max = 0.1
nu = (1 / omega - 0.5) / 3

# 🔄 Local domain decomposition based on process grid
# num_devices = jax.process_count()
# px = int(math.floor(math.sqrt(num_devices)))
# while num_devices % px != 0:
#     px -= 1
# py = num_devices // px

# Ensure domain is divisible by px and py
# assert NX % px == 0 and NY % py == 0, f"NX/NY not divisible by px/py: {NX},{NY} / {px},{py}"


#Try 1 Dimension mesh decomposition
px = size  # Decompose only in X direction
py = 1     # No decomposition in Y

assert NX % px == 0, f"NX not divisible by number of processes: NX={NX}, px={px}"

local_NX = NX // px
local_NY = NY

# New local shape with halos
local_NX_halo = local_NX + 2
local_NY_halo = local_NY + 2

print(f"Rank {rank}: jax.process_count() = {jax.process_count()}")
print(f"Rank {rank}: MPI size = {size}")
print(f"Global domain: NX={NX}, NY={NY}")
print(f"Process grid: px={px}, py={py}")
print(f"Local domain: local_NX={local_NX}, local_NY={local_NY}")
print(f"Expected total size from shards: local_NX*px = {local_NX * px}, local_NY*py = {local_NY * py}")

# ✅ Lattice constants
dtype = jnp.float32
w = jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype)
c = jnp.array([[0,1,0,-1,0,1,-1,-1,1],
               [0,0,1,0,-1,1,1,-1,-1]], dtype)

@jax.jit
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u * u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    return wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[jnp.newaxis, :, :])

@jax.jit
def collide(g):
    # eps = 1e-12
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    return g + omega * (feq - g), u

shifts = [(int(c[0, i]), int(c[1, i])) for i in range(9)]

@jax.jit
def stream(f):
    f_new = jnp.zeros_like(f)
    for i in range(9):
        dx, dy = shifts[i]
        f_new = f_new.at[i].set(jnp.roll(f[i], shift=(dx, dy), axis=(0,1)))
    return f_new
# @jax.jit
# def stream(f):
#     f_new = jnp.zeros_like(f)
#     for i in range(9):
#         dx, dy = shifts[i]
#         shifted = f[i]
#         if dx == 1:
#             shifted = shifted[:-1, :]
#             f_new = f_new.at[i, 1:, :].set(shifted)
#         elif dx == -1:
#             shifted = shifted[1:, :]
#             f_new = f_new.at[i, :-1, :].set(shifted)
#         elif dy == 1:
#             shifted = shifted[:, :-1]
#             f_new = f_new.at[i, :, 1:].set(shifted)
#         elif dy == -1:
#             shifted = shifted[:, 1:]
#             f_new = f_new.at[i, :, :-1].set(shifted)
#         else:
#             f_new = f_new.at[i].set(f[i])  # No shift
#     return f_new

opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def apply_bounce_back(f):
    # Only interior boundary (exclude halos)
    f = f.at[:, 1:-1, 1].set(f[opposite, 1:-1, 1])       # left edge
    f = f.at[:, 1:-1, -2].set(f[opposite, 1:-1, -2])     # right edge
    f = f.at[:, 1, 1:-1].set(f[opposite, 1, 1:-1])       # bottom edge

    # # Corners if needed
    f = f.at[:, -2, 1].set(f[opposite, -2, 1])           # top-left corner
    f = f.at[:, -2, -2].set(f[opposite, -2, -2])         # top-right corner
    f = f.at[:, 1, 1].set(f[opposite, 1, 1])             # bottom-left corner
    f = f.at[:, 1, -2].set(f[opposite, 1, -2])           # bottom-right corner            
    return f

def apply_top_lid_velocity(f, u_lid=jnp.array([-u_max, 0.0])):
    rho_wall = (f[0, 1:-1, -2] + f[1, 1:-1, -2] + f[3, 1:-1, -2] +
                2 * (f[4, 1:-1, -2] + f[7, 1:-1, -2] + f[8, 1:-1, -2]))
    incoming = [2, 5, 6]
    for i in incoming:
        i_opp = opposite[i]
        ci_dot_u = c[0, i] * u_lid[0] + c[1, i] * u_lid[1]
        correction = 6.0 * w[i] * rho_wall * ci_dot_u
        f = f.at[i, 1:-1, -2].set(f[i_opp, 1:-1, -2] - correction)
        f = jnp.maximum(f, 0.0)
    return f

ix = jax.process_index()
iy = 0

x_start = ix * local_NX
y_start = 0
x_end = x_start + local_NX
y_end = NY

print(f"Using 2D mesh shape: ({px}, {py})")
print(f"Global domain: {NX}x{NY}, Steps: {NSTEPS}")
print(f"Process {rank} handles local domain x:[{x_start}, {x_end}) y:[{y_start}, {y_end})")

# 🔄 Initialize only the local subdomain
x = jnp.arange(x_start, x_end) + 0.5
y = jnp.arange(y_start, y_end) + 0.5
X, Y = jnp.meshgrid(x, y, indexing='ij')


u0 = jnp.zeros((local_NX, local_NY), dtype)
#DEBUG
# print(f"Rank {rank}: local u0 shape: {u0.shape}")

rho0 = jnp.ones((local_NX, local_NY), dtype=dtype)
v0 = jnp.zeros_like(u0)
u_init = jnp.array([u0, v0])
f0 = equilibrium(rho0, u_init).astype(dtype)
# Start with interior equilibrium distribution
f0_inner = equilibrium(rho0, u_init).astype(dtype)
f0 = jnp.zeros((9, local_NX_halo, local_NY_halo), dtype=dtype)
f0 = f0.at[:, 1:-1, 1:-1].set(f0_inner)

# Initialize halos as copies of adjacent interior cells (simple zero-gradient BC for initialization)
f0 = f0.at[:, 0, 1:-1].set(f0[:, 1, 1:-1])       # left halo
f0 = f0.at[:, -1, 1:-1].set(f0[:, -2, 1:-1])     # right halo
f0 = f0.at[:, 1:-1, 0].set(f0[:, 1:-1, 1])       # bottom halo
f0 = f0.at[:, 1:-1, -1].set(f0[:, 1:-1, -2])     # top halo

# Corners (optional)
f0 = f0.at[:, 0, 0].set(f0[:, 1, 1])
f0 = f0.at[:, 0, -1].set(f0[:, 1, -2])
f0 = f0.at[:, -1, 0].set(f0[:, -2, 1])
f0 = f0.at[:, -1, -1].set(f0[:, -2, -2])

ndx, ndy = px, py  # process grid dims

comm_cart = comm.Create_cart((ndx, ndy), periods=(False, False))
coords = comm_cart.Get_coords(rank)

left_src = rank - 1 if coords[0] > 0 else MPI.PROC_NULL
left_dst = rank - 1 if coords[0] > 0 else MPI.PROC_NULL

right_src = rank + 1 if coords[0] < px - 1 else MPI.PROC_NULL
right_dst = rank + 1 if coords[0] < px - 1 else MPI.PROC_NULL
print(f"Rank {rank} neighbors: left_src={left_src}, right_src={right_src}")

def communicate(f_ikl):
    f_np = np.array(f_ikl)  # Ensure correct type

    # print(f"[Rank {rank}] Starting communicate()", flush=True, file=sys.stderr)

    # LEFT-RIGHT communication
    send_left = f_np[:, 1, :].copy()     # Shape: (Ny, Q)
    send_right = f_np[:, -2, :].copy()

    recv_left = np.empty_like(send_left)
    recv_right = np.empty_like(send_right)

    requests = []

    if left_src >= 0:
        req_send_left = comm_cart.Isend(send_left, dest=left_dst)
        req_recv_left = comm_cart.Irecv(recv_left, source=left_src)
        requests.extend([req_send_left, req_recv_left])

    if right_src >= 0:
        req_send_right = comm_cart.Isend(send_right, dest=right_dst)
        req_recv_right = comm_cart.Irecv(recv_right, source=right_src)
        requests.extend([req_send_right, req_recv_right])

    MPI.Request.Waitall(requests)

    if left_src >=0:
        f_np[:, 0, :] = recv_left
    if right_src >=0:
        f_np[:, -1, :] = recv_right

    return jnp.array(f_np)

local_devices = jax.local_devices()
print(f"Process {jax.process_index()} local devices:", local_devices)

local_mesh_shape = (len(local_devices),)  # 1D mesh for simplicity
local_device_mesh = mesh_utils.create_device_mesh(local_mesh_shape, devices=local_devices)

mesh = Mesh(local_device_mesh, axis_names=('x'))  # Use axis_names matching mesh dims

#  Device mesh setup (same as before)
# mesh = Mesh(mesh_utils.create_device_mesh((px, py)), axis_names=('x', 'y'))


with mesh:
    # sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    sharding = NamedSharding(mesh, P(None,'x'))  

    f = jax.device_put(f0, sharding)

    def lbm_collide_stream(f):
        f, _ = collide(f)
        f = stream(f)
        f = apply_bounce_back(f)
        f = apply_top_lid_velocity(f)
        return f

    lbm_collide_stream = pjit(
        lbm_collide_stream,
        in_shardings=P(None, 'x'),
        out_shardings=P(None, 'x'),
    )

    print("Sharding info:", f.sharding)

    def gather_velocity_field(rank, comm, comm_cart, u_np_local, local_NX, local_NY, NX, NY):
    # Gather all local velocity fields to root
        all_shards = comm.gather(u_np_local, root=0)
        if rank != 0:
            return None  # Only root reconstructs

    
        u_combined = np.zeros((2, NX, NY), dtype=u_np_local.dtype)

        for r, shard in enumerate(all_shards):
            coords = comm_cart.Get_coords(r)
            i_start = coords[0] * local_NX
            j_start = coords[1] * local_NY

        # Sanity check on shape
        if shard.shape != (2, local_NX, local_NY):
            shard = shard.reshape(2, local_NX, local_NY)
            assert shard.shape == (2, local_NX, local_NY), f"Shape still incorrect: {shard.shape}"

        # Place the shard in the full array
        u_combined[:, i_start:i_start + local_NX, j_start:j_start + local_NY] = shard

        return u_combined


    amp = []
    profiles = []
    os.makedirs("frames", exist_ok=True)

    start = time.time()

    for step in range(NSTEPS):
        f_cpu = f.addressable_data(0)  # Get CPU array for MPI communication            
                
        if size> 1:
            # print(f"[Rank {rank}] Step {step} communicating halos...", flush=True, file=sys.stderr)
            f_cpu = communicate(f_cpu)

        f = jax.device_put(f_cpu, f.sharding)  

        f = lbm_collide_stream(f)      

        if step % 100 == 0:
            rho = jnp.einsum('ijk->jk', f[:, 1:-1, 1:-1])
            u = jnp.einsum('ai,ixy->axy', c, f[:, 1:-1, 1:-1]) / rho
            # u_gathered = multihost_utils.process_allgather(u)
            # u_np = np.array(u_gathered)
            # all_shards = comm.gather(u_np, root=0)

            u_np_local = np.array(multihost_utils.process_allgather(u))
            # Trim extra leading dimensions
            while u_np_local.ndim > 3:
                u_np_local = u_np_local[0]
            assert u_np_local.shape == (2, local_NX, local_NY), f"Shape mismatch: {u_np_local.shape}"
            print(f"[Rank {rank}] local u shape: {u.shape}, after allgather: {u_np_local.shape}", flush=True)

            u_combined = gather_velocity_field(rank, comm, comm_cart, u_np_local, local_NX, local_NY, NX, NY)

            if rank == 0:

                u_x = u_combined[0]
                u_y = u_combined[1]

                speed = np.sqrt(u_x**2 + u_y**2)
                print(f"Step {step}: top lid max u_x = {u_x[:, -1].max():.4f}")

                ix = min(NX // 2, u_x.shape[0] - 1)
                iy = min(NY // 8, u_x.shape[1] - 1)
                amp.append(u_x[ix, iy])

                profiles.append(u_x[NX // 2, :].copy())  # now safe!

                # Meshgrid with correct dimensions for streamplot
                X, Y = np.meshgrid(np.arange(NY), np.arange(NX))
                xlim = (0, NY)
                ylim = (0, NX)

                plt.figure(figsize=(7, 6))
                plt.streamplot(X, Y, u_x.T, u_y.T, density=1.2, linewidth=1, arrowsize=1.5)
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.title(f'Lid-driven cavity flow (Steps:{step:05d})')
                plt.xlabel("X")
                plt.ylabel("Y")
                plt.axis("equal")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'frames/streamplot_{step:05d}.png')
                plt.close()

    end = time.time()

elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

print(f"\nRank {rank}:")
print(f"Time: {elapsed:.2f} s")
print(f"BLUPS: {blups:.3f}")
print(f"Domain: {NX}x{NY}, Steps: {NSTEPS}")
print(f"Viscosity: {nu:.4e}")

if rank == 0:
    import imageio
    for prefix in ['streamplot']:
        with imageio.get_writer(f'{prefix}.gif', mode='I', duration=0.5) as writer:
            for step in range(0, NSTEPS,100):
                filename = f'frames/{prefix}_{step:05d}.png'
                if os.path.exists(filename):
                    image = imageio.imread(filename)
                    writer.append_data(image)