import time
import math
import os
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
NX, NY = 400, 400
NSTEPS = 20000 
omega = 1.0
u_max = 0.1
nu = (1 / omega - 0.5) / 3

# 🔄 Local domain decomposition based on process grid
num_devices = jax.process_count()
px = int(math.floor(math.sqrt(num_devices)))
while num_devices % px != 0:
    px -= 1
py = num_devices // px

# Ensure domain is divisible by px and py
assert NX % px == 0 and NY % py == 0, f"NX/NY not divisible by px/py: {NX},{NY} / {px},{py}"

local_NX = NX // px
local_NY = NY // py

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
    feq = wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[jnp.newaxis, :, :])
    return jnp.maximum(feq, 1e-10)

@jax.jit
def collide(g):
    rho = jnp.einsum('ijk->jk', g)
    rho = jnp.maximum(rho, 1e-6)  # prevent divide-by-zero
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    g_new = g + omega * (feq - g)
    return jnp.maximum(g_new, 1e-10), u

shifts = [(int(c[0, i]), int(c[1, i])) for i in range(9)]

@jax.jit
def stream(f):
    return jnp.stack([jnp.roll(f[i], shift=(shifts[i][0], shifts[i][1]), axis=(0,1)) for i in range(9)], axis=0)

opposite = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

def apply_bounce_back(f):
    f_copy = f.copy()  # make it writeable
    # Bounce-back on left boundary (x=0)
    f = f.at[:, 1:-1, 0].set(f_copy[opposite, 1:-1, 0])
    # Bounce-back on right boundary (x=-1)
    f = f.at[:, 1:-1, -1].set(f_copy[opposite, 1:-1, -1])
    # Bounce-back on bottom boundary (y=0)
    f = f.at[:, 0, 1:-1].set(f_copy[opposite, 0, 1:-1])
    # Bounce-back on top boundary (y=-1)
    f = f.at[:, -1, 1:-1].set(f_copy[opposite, -1, 1:-1])
    f = f.at[:, 0, 0].set(f[opposite, 0, 0])
    f = f.at[:, -1, 0].set(f[opposite, -1, 0])
    f = f.at[:, 0, -1].set(f[opposite, 0, -1])
    f = f.at[:, -1, -1].set(f[opposite, -1, -1])
    return f

def apply_top_lid_velocity(f, u_lid=jnp.array([-u_max, 0.0])):
    rho_wall = (f[0, :, -1] + f[1, :, -1] + f[3, :, -1] +
                2 * (f[4, :, -1] + f[7, :, -1] + f[8, :, -1]))
    incoming = [2, 5, 6]
    for i in incoming:
        i_opp = opposite[i]
        ci_dot_u = c[0, i] * u_lid[0] + c[1, i] * u_lid[1]
        correction = 6.0 * w[i] * rho_wall[1:-1] * ci_dot_u
        f = f.at[i, 1:-1, -1].set(f[i_opp, 1:-1, -1] - correction)
    return f

def clamp_f(f):
    f = jnp.where(jnp.isnan(f), 1e-12, f)  # Replace NaNs
    f = jnp.maximum(f, 1e-12)                # Avoid negative or zero
    return f

ix = jax.process_index() % px
iy = jax.process_index() // px


x_start = ix * local_NX
y_start = iy * local_NY
x_end = x_start + local_NX
y_end = y_start + local_NY

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

f0 = jnp.pad(f0, ((0, 0), (1, 1), (1, 1)))  # Add halo in x and y
# 🔍 Initial NaN diagnostics
rho_test = jnp.einsum('ijk->jk', f0)
rho_test = jnp.maximum(rho_test, 1e-6)
u_test = jnp.einsum('ai,ixy->axy', c, f0) / rho_test

print(f"[Rank {rank}] Initial NaN diagnostics:")
print("NaNs in f0:", jnp.isnan(f0).sum())
print("NaNs in rho:", jnp.isnan(rho_test).sum())
print("NaNs in u:", jnp.isnan(u_test).sum())

print("Initial min/max values:")
print("rho min/max:", rho_test.min(), rho_test.max())
print("u min/max:", u_test.min(), u_test.max())

def halo_exchange(f):
    # Assume f shape is (9, NX_local + 2, NY_local + 2)
    # Extract halo faces (interior edges)
    send_left = f[:, 1, 1:-1]
    send_right = f[:, -2, 1:-1]
    send_down = f[:, 1:-1, 1]
    send_up = f[:, 1:-1, -2]

    # Allocate buffers for receiving
    recv_left = np.empty_like(send_left)
    recv_right = np.empty_like(send_right)
    recv_down = np.empty_like(send_down)
    recv_up = np.empty_like(send_up)

    # Neighbor process ranks
    left = rank - 1 if ix > 0 else MPI.PROC_NULL
    right = rank + 1 if ix < px - 1 else MPI.PROC_NULL
    down = rank - px if iy > 0 else MPI.PROC_NULL
    up = rank + px if iy < py - 1 else MPI.PROC_NULL

    # Start non-blocking communication
    reqs = []
    reqs.append(comm.Isend(send_left, dest=left, tag=0))
    reqs.append(comm.Isend(send_right, dest=right, tag=1))
    reqs.append(comm.Isend(send_down, dest=down, tag=2))
    reqs.append(comm.Isend(send_up, dest=up, tag=3))
    reqs.append(comm.Irecv(recv_left, source=left, tag=1))
    reqs.append(comm.Irecv(recv_right, source=right, tag=0))
    reqs.append(comm.Irecv(recv_down, source=down, tag=3))
    reqs.append(comm.Irecv(recv_up, source=up, tag=2))

    # Wait for all to complete
    MPI.Request.Waitall(reqs)

    # Fill ghost cells using NumPy indexing (not JAX)
    f = f.copy()  # make it writeable
    f[:, 0, 1:-1] = recv_left
    f[:, -1, 1:-1] = recv_right   
    f[:, 1:-1, 0] = recv_down     
    f[:, 1:-1, -1] = recv_up      

    return f

mesh = Mesh(mesh_utils.create_device_mesh((px, py)), axis_names=('x', 'y'))


with mesh:
    sharding = NamedSharding(mesh, P(None, 'x', 'y'))
    f = jax.device_put(f0, sharding)

    def lbm_step(f):
    # 1. Streaming step
        f1 = clamp_f(stream(f))
    
    # 2. Collision step
        f2, _ = collide(f1)
        f2 = clamp_f(f2)
    
    # 3. Bounce-back boundary condition
        f3 = clamp_f(apply_bounce_back(f2))
    
    # 4. Apply lid (moving wall) velocity boundary
        f4 = clamp_f(apply_top_lid_velocity(f3))
    
        return f4

    
    # Debug version of lbm_step
    def debug_lbm_step(f, step=0, rank=0):
        def check_nan(label, arr):
            nan_count = jnp.isnan(arr).sum()
            if nan_count > 0:
                print(f"[Rank {rank}] NaNs in {label} at step {step}: {nan_count}")
            return arr

        f1 = stream(f)
        check_nan("f after stream", f1)

        f2, _ = collide(f1)
        check_nan("f after collide", f2)

        f3 = apply_bounce_back(f2)
        check_nan("f after bounce-back", f3)

        f4 = apply_top_lid_velocity(f3)
        check_nan("f after top lid", f4)
        return f4


    lbm_step = pjit(
        lbm_step,
        in_shardings=P(None, 'x', 'y'),
        out_shardings=P(None, 'x', 'y'),
    )

    print("Sharding info:", f.sharding)
    # jax.debug.visualize_array_sharding(f.reshape(-1, f.shape[-1]))


    amp = []
    profiles = []
    os.makedirs("frames", exist_ok=True)

    start = time.time()

    for step in range(NSTEPS):
        # ↓ Move data from device to host (JAX → NumPy)
        f_host = jax.device_get(f)           # JAX → NumPy

        # ↓ Perform MPI-based halo exchange on CPU
        f_host = halo_exchange(f_host)

        # ↓ Move data back to device (NumPy → JAX)
        f = jax.device_put(f_host)           # NumPy → JAX

        # ↓ Perform JAX-accelerated LBM step (stream → collide → BC)
        # f = lbm_step(f)
        f = debug_lbm_step(f, step=step, rank=rank)

        # print(f"Step {step}: f min={jnp.min(f):.2e}, max={jnp.max(f):.2e}, rho min={jnp.min(rho):.2e}, max={jnp.max(rho):.2e}, u min={jnp.min(u):.2e}, max={jnp.max(u):.2e}")


        # --- Debug NaN check after first LBM step ---
        if step == 0:
            f_check = jax.device_get(f)
            nan_count_post_step = np.isnan(f_check).sum()
            print(f"[Rank {rank}] Post-step 0 NaN count: {nan_count_post_step}")
            if nan_count_post_step > 0:
                print("NaNs introduced in lbm_step!")
                np.set_printoptions(precision=4, suppress=True, linewidth=140)
                print("Sample slice of f with NaNs:\n", f_check[:, 200, 200])


        # if jnp.isnan(f).any() or jnp.isnan(rho).any() or jnp.isnan(u).any():
        #     print(f"NaNs detected at step {step}")
        #     break

        if step % 100 == 0:
            nan_count_f = jnp.isnan(f).sum()
            rho = jnp.einsum('ijk->jk', f)
            nan_count_rho = jnp.isnan(rho).sum()

            rho = jnp.maximum(rho, 1e-6)  # Prevent division by zero
            u = jnp.einsum('ai,ixy->axy', c, f) / rho
            nan_count_u = jnp.isnan(u).sum()
            print(f"Step {step}: NaNs in f = {nan_count_f}, rho = {nan_count_rho}, u = {nan_count_u}")

            # u_local = u.addressable_data(0)
            u = u[:, 1:-1, 1:-1]  # remove halo before gathering
            # u_gathered = multihost_utils.process_allgather(u)
            # u_np = np.array(u_gathered)
            # all_shards = comm.gather(u_np, root=0)

            u_global = multihost_utils.process_allgather(u)  # shape: (num_devices, 2, local_NX, local_NY)


            if rank == 0:
                print("Gathered global u shape:", u_global.shape)
                # print(f"Gathered {len(all_shards)} shards, expecting {size}")
                # for i, shard in enumerate(all_shards):
                #     print(f"Shard {i} shape: {shard.shape}")
                try:
                    # all_shards is a flat list of shape (2, local_NX, local_NY) for each process
                    # Reconstruct a (px, py) grid of velocity fields
                    ordered_grid = [[None for _ in range(px)] for _ in range(py)]
                    for proc_id in range(size):
                        shard = np.array(u_global[proc_id])
                        ix = proc_id % px
                        iy = proc_id // px
                        ordered_grid[iy][ix] = shard

                    # Concatenate along Y (axis=3), then X (axis=2)
                    rows = [np.concatenate(row, axis=2) for row in ordered_grid]
                    u_combined = np.concatenate(rows, axis=1)

                    u_x = u_combined[0]
                    u_y = u_combined[1]

                    assert u_combined.shape == (2, NX, NY), f"u_combined.shape = {u_combined.shape}, expected (2, {NX}, {NY})"

                    speed = np.sqrt(u_x**2 + u_y**2)
                    print(f"Step {step}: top lid max u_x = {u_x[:, -1].max():.4f}")

                    ix = min(NX // 2, u_x.shape[0] - 1)
                    iy = min(NY // 8, u_x.shape[1] - 1)
                    amp.append(u_x[ix, iy])
                    profiles.append(u_x[NX // 2, :].copy())

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

                except Exception as e:
                    print("Concatenation failed:", e)
                    raise

                # plt.figure(figsize=(6,5))
                # plt.imshow(speed.T, origin='lower', cmap='plasma', extent=[0, NX, 0, NY])
                # plt.colorbar(label='Speed')
                # plt.title(f'Speed Magnitude at step {step}')
                # plt.xlabel('X')
                # plt.ylabel('Y')
                # plt.tight_layout()
                # plt.savefig(f'frames/speed_magnitude_{step:05d}.png')
                # plt.close()

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