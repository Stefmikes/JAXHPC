import time
import math
import os
import socket
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.experimental import mesh_utils, multihost_utils  # ✅ added multihost_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit
from mpi4py import MPI

# ✅ MPI Initialization
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ✅ JAX Distributed Environment Setup
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

# ✅ Log device and environment info
print("Starting JAX PJIT simulation...")
print(f"Rank: {rank}, Size: {size}")
print("All visible devices:", jax.devices())
print("Local devices:", jax.local_devices())
print(f"Process {jax.process_index()} on {socket.gethostname()} using {jax.local_devices()}")
print(f"JAX backend: {jax.default_backend()}")

# ✅ Simulation parameters
NX, NY = 4000, 6000
NSTEPS = 10000
omega = 1.7
u_max = 0.1
nu = (1 / omega - 0.5) / 3

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

# Outside JIT — after defining c:
shifts = [(int(c[0, i]), int(c[1, i])) for i in range(9)]

@jax.jit
def stream(g):
    return jnp.stack([jnp.roll(g[i], shift=shifts[i], axis=(0,1)) for i in range(9)])

# ✅ Initialize domain
x = jnp.arange(NX) + 0.5
y = jnp.arange(NY) + 0.5
X, Y = jnp.meshgrid(x, y, indexing='ij')
u0 = u_max * jnp.sin(2 * jnp.pi * Y / NY)
rho0 = jnp.ones((NX, NY), dtype=dtype)
v0 = jnp.zeros_like(u0)
u_init = jnp.array([u0, v0])
f0 = equilibrium(rho0, u_init).astype(dtype)

# ✅ Set up device mesh
# devices = jax.devices()
# num_devices = len(devices)
num_devices = jax.process_count()
px = int(math.floor(math.sqrt(num_devices)))
while num_devices % px != 0:
    px -= 1
py = num_devices // px
print(f"Using 2D mesh shape: ({px}, {py})")

mesh = Mesh(mesh_utils.create_device_mesh((px, py)), axis_names=('x', 'y'))

# ✅ Run simulation
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

    print("Sharding info:", f.sharding)

    amp = []
    profiles = []
    os.makedirs("frames", exist_ok=True)

    start = time.time()

    for step in range(NSTEPS):
        f = lbm_step(f)

        if step % 200 == 0:
            rho = jnp.einsum('ijk->jk', f)
            u = jnp.einsum('ai,ixy->axy', c, f) / rho

            u_gathered = multihost_utils.process_allgather(u)

            for i, arr in enumerate(u_gathered):
                print(f"Shape of u_gathered[{i}]: {arr.shape}")
            
            if rank == 0:
                print(f"Gathered {len(u_gathered)} shards")
                print(f"Mesh shape: px={px}, py={py}, total={px * py}")
                # ✅ FIXED: Reshape u_gathered into 2D shard layout
                try:
                   # Reconstruct mesh layout from gathered full replicas
                    assert len(u_gathered) == px * py, f"Expected {px*py} shards, got {len(u_gathered)}"

                    shards_2d = [u_gathered[i * py:(i + 1) * py] for i in range(px)]
                    rows = [jnp.concatenate(row, axis=2) for row in shards_2d]  # concatenate over Y
                    u_combined = jnp.concatenate(rows, axis=1)  # concatenate over X

                    # Final shape: (2, NX, NY)
                    u_combined = jnp.reshape(u_combined, (2, NX, NY))

                except Exception as e:
                    print("Concatenation failed:", e)
                    print("Number of gathered shards:", len(u_gathered))
                    for i, arr in enumerate(u_gathered):
                        print(f"Shard {i}: shape={arr.shape}")
                    raise

                
                shards_2d = [u_gathered[i * py:(i + 1) * py] for i in range(px)]

                # ✅ FIXED: First stack each list of 2D arrays into 3D arrays, then concatenate
                rows = [jnp.concatenate([jnp.expand_dims(shard, axis=0) for shard in row_shards], axis=0) for row_shards in shards_2d]

                # ✅ Now concatenate rows along axis=1 (NX axis)
                u_combined = jnp.concatenate(rows, axis=1)

                # ✅ Extract u_x (component 0) from 3D array of shape (px * py, NX, NY)
                # If necessary, reshape to final shape (2, NX, NY) based on how original data was stored
                u_combined = jnp.reshape(u_combined, (2, NX, NY))  # <-- final corrected shape
                u_host = np.array(u_combined[0])  # u_x

                amp.append(u_host[NX // 2, NY // 8])
                profile = u_host[NX // 2, :].copy()
                profiles.append(profile)

                plt.figure()
                plt.plot(profile)
                plt.title(f'Wave profile at step {step}')
                plt.xlabel('y')
                plt.ylabel('u_x Amplitude')
                plt.ylim(-u_max, u_max)
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f'frames/frame_{step:05d}.png')
                plt.close()

    end = time.time()

# ✅ Performance Metrics
elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

print(f"\nRank {rank}:")
print(f"Time: {elapsed:.2f} s")
print(f"BLUPS: {blups:.3f}")
print(f"Domain: {NX}x{NY}, Steps: {NSTEPS}")
print(f"Viscosity: {nu:.4e}")

# ✅ Plotting (only on rank 0)
if rank == 0:
    amp = np.array(amp)
    profiles = np.array(profiles)

    plt.figure()
    for profile in profiles:
        plt.plot(profile)
    plt.title('Wave decay over time')
    plt.xlabel('y')
    plt.ylabel('u_x Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(amp / amp[0])
    plt.title('Amplitude decay at (NX/2, NY/8)')
    plt.xlabel('Timestep index (every 200 steps)')
    plt.ylabel('Relative Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    import imageio
    with imageio.get_writer('wave_decay.gif', mode='I', duration=0.1) as writer:
        for step in range(0, NSTEPS, 200):
            filename = f'frames/frame_{step:05d}.png'
            if os.path.exists(filename):
                image = imageio.imread(filename)
                writer.append_data(image)
