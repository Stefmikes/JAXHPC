import time
import math
import os
import sys
import socket
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
NSTEPS = 10000
omega = 1.6
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

local_NX = NX //px
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
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    return g + omega * (feq - g), u

shifts = [(int(c[0, i]), int(c[1, i])) for i in range(9)]

@jax.jit
def stream(f):
    return jnp.stack([jnp.roll(f[i], shift=(shifts[i][0], shifts[i][1]), axis=(0,1)) for i in range(9)], axis=0)


opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def bounce_from_left(f):
    return f.at[:, 1:-1, 1].set(f[opposite, 1:-1, 1])
def bounce_from_right(f):
    return f.at[:, 1:-1, -2].set(f[opposite, 1:-1, -2])
def bounce_from_bottom(f):
    return f.at[:, 1, 1:-1].set(f[opposite, 1, 1:-1])
def corner_top_left(f):
    return f.at[:, 1, -2].set(f[opposite, 1, -2])
def corner_top_right(f):
    return f.at[:, -2, -2].set(f[opposite, -2, -2])
def corner_bottom_left(f):
    return f.at[:, 1, 1].set(f[opposite, 1, 1])
def corner_bottom_right(f):
    return f.at[:, -2, 1].set(f[opposite, -2, 1])


def apply_bounce_back(f, is_left, is_right, is_bottom):
    f = jax.lax.cond(is_left, lambda f: bounce_from_left(f), lambda f: f, f)
    f = jax.lax.cond(is_right, lambda f: bounce_from_right(f), lambda f: f, f)
    f = jax.lax.cond(is_bottom, lambda f: bounce_from_bottom(f), lambda f: f, f)
   
    #Only apply corners if at corresponding edges, to avoid out-of-bound errors
    f = jax.lax.cond(is_left & is_top_edge, corner_top_left, lambda f: f, f)
    f = jax.lax.cond(is_right & is_top_edge, corner_top_right, lambda f: f, f)
    f = jax.lax.cond(is_left & is_bottom, corner_bottom_left, lambda f: f, f)
    f = jax.lax.cond(is_right & is_bottom, corner_bottom_right, lambda f: f, f)
    return f

def apply_top_lid_velocity(f, is_top, u_lid=jnp.array([-u_max, 0.0])):
    def do_lid_velocity(f):
        rho_wall = (f[0, 1:-1, -2] + f[1, 1:-1, -2] + f[3, 1:-1, -2] +
                    2 * (f[4, 1:-1, -2] + f[7, 1:-1, -2] + f[8, 1:-1, -2]))

        incoming = jnp.array([2, 5, 6])

        def body(i, f):
            i_ = incoming[i]
            i_opp = opposite[i_]
            ci_dot_u = c[0, i_] * u_lid[0] + c[1, i_] * u_lid[1]
            correction = 6.0 * w[i_] * rho_wall * ci_dot_u
            f = f.at[i_, 1:-1, -2].set(f[i_opp, 1:-1, -2] - correction)
            return f

        f = jax.lax.fori_loop(0, len(incoming), body, f)
        return f

    return jax.lax.cond(is_top, do_lid_velocity, lambda f: f, f)


ix = jax.process_index()
iy = 0

x_start = ix * local_NX
y_start = 0
x_end = x_start + local_NX
y_end = NY

# ix = 0
# iy = jax.process_index()

# x_start = 0
# y_start = iy * local_NY
# x_end = NX
# y_end = y_start + local_NY

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

left_src, left_dst = comm_cart.Shift(direction=0, disp=-1)
right_src, right_dst = comm_cart.Shift(direction=0, disp=1)
bottom_src, bottom_dst = comm_cart.Shift(direction=1, disp=-1)
top_src, top_dst = comm_cart.Shift(direction=1, disp=1)

print(f"Rank {rank} neighbors: left_src={left_src}, right_src={right_src}")

is_left_edge = coords[0] == 0
is_right_edge = coords[0] == px - 1
is_bottom_edge = coords[1] == 0
is_top_edge = coords[1] == py - 1

is_left_edge = jnp.array(is_left_edge, dtype=bool)
is_right_edge = jnp.array(is_right_edge, dtype=bool)
is_bottom_edge = jnp.array(is_bottom_edge, dtype=bool)
is_top_edge = jnp.array(is_top_edge, dtype=bool)


def communicate(f_ikl, comm_cart, left_src, left_dst, right_src, right_dst,
                bottom_src, bottom_dst, top_src, top_dst,py):
    # print(f"[Rank {rank}] Starting communicate()", flush=True, file=sys.stderr)
    f_np = np.array(f_ikl)  # Ensure mutable array for MPI

    # print(f"[Rank {rank}] Communicating LEFT", flush=True)
    sendbuf_left = np.ascontiguousarray(f_np[:, 1, :])
    recvbuf_left = np.ascontiguousarray(f_np[:, -1, :])
    comm_cart.Sendrecv(sendbuf=sendbuf_left, dest=left_dst, sendtag=0,
                       recvbuf=recvbuf_left, source=left_src, recvtag=0)
    f_np[:, -1, :] = recvbuf_left  
    
    # print(f"[Rank {rank}] Communicating RIGHT", flush=True)
    sendbuf_right = np.ascontiguousarray(f_np[:, -2, :])
    recvbuf_right = np.ascontiguousarray(f_np[:, 0, :])
    comm_cart.Sendrecv(sendbuf=sendbuf_right, dest=right_dst, sendtag=1,
                       recvbuf=recvbuf_right, source=right_src, recvtag=1)
    f_np[:, 0, :] = recvbuf_right  
    
    # print(f"[Rank {rank}] Communicating BOTTOM", flush=True)
    if py > 1:
        sendbuf_bottom = np.ascontiguousarray(f_np[:, :, 1])
        recvbuf_bottom = np.ascontiguousarray(f_np[:, :, -1])
        comm_cart.Sendrecv(sendbuf=sendbuf_bottom, dest=bottom_dst, sendtag=2,
                       recvbuf=recvbuf_bottom, source=bottom_src, recvtag=2)
        f_np[:, :, -1] = recvbuf_bottom  
    
        # print(f"[Rank {rank}] Communicating TOP", flush=True)
        sendbuf_top = np.ascontiguousarray(f_np[:, :, -2])
        recvbuf_top = np.ascontiguousarray(f_np[:, :, 0])
        comm_cart.Sendrecv(sendbuf=sendbuf_top, dest=top_dst, sendtag=3,
                       recvbuf=recvbuf_top, source=top_src, recvtag=3)
        f_np[:, :, 0] = recvbuf_top  

    return jnp.array(f_np)

local_devices = jax.local_devices()
print(f"Process {jax.process_index()} local devices:", local_devices)

local_mesh_shape = (len(local_devices),)  # 1D mesh for simplicity
local_device_mesh = mesh_utils.create_device_mesh(local_mesh_shape, devices=local_devices)

mesh = Mesh(local_device_mesh, axis_names=('x',))  # Use axis_names matching mesh dims

#  Device mesh setup (same as before)
# mesh = Mesh(mesh_utils.create_device_mesh((px, py)), axis_names=('x', 'y'))


with mesh:
    # sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    sharding = NamedSharding(mesh, P(None,'x', None))  

    f = jax.device_put(f0, sharding)

    def lbm_collide_stream(f, is_left, is_right, is_bottom, is_top):
        f, _ = collide(f)
        f = stream(f)
        f = apply_bounce_back(f, is_left, is_right, is_bottom)
        f = apply_top_lid_velocity(f, is_top)
        return f

    lbm_collide_stream = pjit(
        lbm_collide_stream,
        in_shardings=(P(None, 'x', None), P(), P(), P(), P()),
        out_shardings=P(None, 'x', None),
    )

    print("Sharding info:", f.sharding)

    amp = []
    profiles = []
    os.makedirs("frames", exist_ok=True)

    start = time.time()

    for step in range(NSTEPS):
        f.block_until_ready()  # Ensure f is ready before proceeding
        f_cpu = f.addressable_data(0)  # Get CPU array for MPI communication            
        comm_cart.barrier()     
        if size > 1:
            # print(f"[Rank {rank}] Step {step} communicating halos...", flush=True, file=sys.stderr)
            f_cpu = communicate(
                    f_cpu, comm_cart,
                    left_src, left_dst, right_src, right_dst,
                    bottom_src, bottom_dst, top_src, top_dst,
                    py
            )
            # if not is_left_edge:
            #     print(f"[Rank {rank}] Sent to left: {f_cpu[:,1,:]}")
            #     print(f"[Rank {rank}] Received left halo: {f_cpu[:,0,:]}")
            #     diff_left = jnp.abs(f_cpu[:,1,:] - f_cpu[:,0,:])  # inner vs received halo
            #     print(f"[STEP:{step}] [Rank {rank}] Max left halo mismatch: {diff_left.max()}")

            # For right boundary
            # if not is_right_edge:
            #     print(f"[Rank {rank}] Sent to right: {f_cpu[:,-2,:]}")
            #     print(f"[Rank {rank}] Received right halo: {f_cpu[:,-1,:]}")
            #     diff_right = jnp.abs(f_cpu[:,-2,:] - f_cpu[:,-1,:])  # inner vs received halo
            #     print(f"[STEP:{step}] [Rank {rank}] Max right halo mismatch: {diff_right.max()}")
        comm_cart.barrier() 
        f = jax.device_put(f_cpu, f.sharding)  

        f = lbm_collide_stream(f, is_left_edge, is_right_edge, is_bottom_edge, is_top_edge)
      
        # if step % 100 == 0:
        #     rho = jnp.einsum('ijk->jk', f[:, 1:-1, 1:-1])
        #     u = jnp.einsum('ai,ixy->axy', c, f[:, 1:-1, 1:-1]) / rho
        #     u_gathered = multihost_utils.process_allgather(u)
        #     u_np = np.array(u_gathered)
        #     all_shards = comm.gather(u_np, root=0)

        #     if rank == 0:

        #         try:
                    
        #             # Normalize shard shapes
        #             normalized_shards = []
        #             for shard in all_shards:
        #                 while shard.ndim > 3:
        #                     shard = shard[0]  # strip excess batch dim
        #                 assert shard.shape[0] == 2, f"Unexpected shard shape: {shard.shape}"
        #                 normalized_shards.append(shard)

        #             # Concatenate along sharded axis (X)
        #             u_combined = np.concatenate(normalized_shards, axis=1)

               
        #             print(f"Reconstructed shape: {u_combined.shape}")
        #             assert u_combined.shape == (2, NX, NY), \
        #                 f"u_combined.shape = {u_combined.shape}, expected (2, {NX}, {NY})"

        #         except Exception as e:
        #             print("Concatenation failed:", e)
        #             raise

        #         u_x = u_combined[0]
        #         u_y = u_combined[1]

        #         speed = np.sqrt(u_x**2 + u_y**2)
        #         print(f"Step {step}: top lid max u_x = {u_x[:, -1].max():.4f}")

        #         ix = min(NX // 2, u_x.shape[0] - 1)
        #         iy = min(NY // 8, u_x.shape[1] - 1)
        #         amp.append(u_x[ix, iy])

        #         profiles.append(u_x[NX // 2, :].copy())  # now safe!

        #         # Meshgrid with correct dimensions for streamplot
        #         X, Y = np.meshgrid(np.arange(NY), np.arange(NX))
        #         xlim = (0, NY)
        #         ylim = (0, NX)
                
        #         # Plot
        #         fig, ax = plt.subplots(figsize=(7, 6))

        #         # Streamplot
        #         ax.streamplot(X, Y, u_x.T, u_y.T, density=1.2, linewidth=1, arrowsize=1.5)

        #         # Labels and aesthetics
        #         ax.set_xlim(xlim)
        #         ax.set_ylim(ylim)
        #         ax.set_title(f'Lid-driven cavity flow (Steps:{step:05d})')
        #         ax.set_xlabel("X")
        #         ax.set_ylabel("Y")
        #         ax.axis("equal")
        #         ax.grid(True)
        #         plt.tight_layout()
        #         plt.savefig(f'frames/streamplot_{step:05d}.png')
        #         plt.close()

    end = time.time()

elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

print(f"\nRank {rank}:")
print(f"Time: {elapsed:.2f} s")
print(f"BLUPS: {blups:.3f}")
print(f"Domain: {NX}x{NY}, Steps: {NSTEPS}")
print(f"Viscosity: {nu:.4e}")

# if rank == 0:
#     import imageio
#     for prefix in ['streamplot']:
#         with imageio.get_writer(f'{prefix}.gif', mode='I', duration=0.5) as writer:
#             for step in range(0, NSTEPS,100):
#                 filename = f'frames/{prefix}_{step:05d}.png'
#                 if os.path.exists(filename):
#                     image = imageio.imread(filename)
#                     writer.append_data(image)