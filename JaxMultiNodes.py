import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax, pmap
import time

devices = jax.devices()
print("All JAX devices:", devices)
print("Local devices:", jax.local_devices())
print("Global device count:", jax.device_count())

n_devices = jax.device_count() 
assert 400 % n_devices == 0, "NX must be divisible by number of devices"  # 🆕 ADDED

# dimensions of the 2D lattice and the Lattice parameters
NX = 400
NY = 300
NXs = NX // n_devices  

scale = 1
NSTEPS = 10000 * scale * scale
omega = 1.7
tau = 1 / omega
u_max = 0.1 / scale
nu = (1 / omega - 0.5) / 3
rho0 = 1.0

dtype = jnp.float32
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=dtype)
c = jnp.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],
               [0, 0, 1,  0, -1, 1,  1, -1, -1]], dtype=dtype)

@jax.jit  # You can experiment removing jit if needed with pmap
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u*u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    return wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[np.newaxis, :, :])

# 🆕 CHANGE: Vectorized streaming with explicit shifts (no fori_loop)
def stream(g):
    shifts = [(0,0), (0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,-1), (-1,1)]
    # g shape: (9, NXs, NY)
    g_shifted = jnp.stack([jnp.roll(g[i], shift=shifts[i], axis=(0,1)) for i in range(9)])
    return g_shifted

def collide(g):
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    return g + omega * (feq - g), u

# 🆕 ADDED: Halo exchange between devices after streaming
def halo_exchange(f):
    # f shape: (n_devices, 9, NXs, NY)
    left_halo = f[:, :, :1, :]    # left edge column
    right_halo = f[:, :, -1:, :]  # right edge column

    # Exchange halos with neighbors (periodic boundary assumed)
    left_recv = jax.lax.ppermute(left_halo, perm=[(i, (i - 1) % n_devices) for i in range(n_devices)])
    right_recv = jax.lax.ppermute(right_halo, perm=[(i, (i + 1) % n_devices) for i in range(n_devices)])

    # Set halos in local arrays
    f = f.at[:, :, 0, :].set(left_recv[:, :, 0, :])
    f = f.at[:, :, -1, :].set(right_recv[:, :, 0, :])
    return f

@pmap
def step(f):
    f = stream(f)
    f = halo_exchange(f)  # 🆕 ADDED: halo exchange after streaming
    f, u = collide(f)
    amp_t = u[0, NXs // 2, NY // 8]
    return f, amp_t

# Initialization split across devices
x = jnp.arange(NX) + 0.5
y = jnp.arange(NY) + 0.5
X, Y = jnp.meshgrid(x, y)
rho = jnp.ones((NX, NY), dtype=dtype)
n = 1
k = 2 * jnp.pi * n / NY
u = jnp.array([u_max * jnp.sin(k * Y.T), jnp.zeros((NX, NY))])
f = equilibrium(rho, u)

# 🆕 CHANGE: reshape to (n_devices, 9, NXs, NY) for pmap
f_init = f.reshape(9, n_devices, NXs, NY).transpose(1, 0, 2, 3)

def run_simulation(f_init, NSTEPS):
    def body(f, _):
        f, amp = step(f)
        return f, amp
    f_final, amps = lax.scan(body, f_init, None, length=NSTEPS)
    return f_final, amps

print(f"Domain size: NX={NX}, NY={NY}")
print(f"Number of steps: {NSTEPS}")
print(f"Omega: {omega}, Viscosity: {nu:.4e}")

start = time.time()
f_final, amps = run_simulation(f_init, NSTEPS)
jax.block_until_ready(f_final)
elapsed_time = time.time() - start

total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed_time / 1e9

print(f"Performance: {blups:.3f} BLUPS")
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Optional plotting (device 0)
# amps_host = jax.device_get(amps[:, 0])
# fig, ax = plt.subplots()
# ax.plot(amps_host / amps_host[0])
# ax.set_title("Amplitude decay (Device 0)")
# ax.set_xlabel("Time step")
# ax.set_ylabel("Normalized Amplitude")
# plt.show()