import jax
import jax.numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
import time
import os
import sys

if sys.platform in ['linux', 'darwin']:
    try:
        import jax.distributed
        distributed_env_vars = ['JAX_DIST_ADDR', 'JAX_DIST_PORT', 'JAX_DIST_PROCESS_COUNT', 'JAX_DIST_PROCESS_ID']
        if all(var in os.environ for var in distributed_env_vars):
            jax.distributed.initialize(
                coordinator_address=os.environ['JAX_DIST_ADDR'] + ":" + os.environ['JAX_DIST_PORT'],
                num_processes=int(os.environ['JAX_DIST_PROCESS_COUNT']),
                process_id=int(os.environ['JAX_DIST_PROCESS_ID']),
            )
            print(f"Distributed JAX initialized for process {jax.process_index()} of {jax.process_count()}")
        else:
            print("Distributed environment variables not set. Running in standalone mode.")
    except ImportError:
        print("jax.distributed not available, running in standalone mode.")
else:
    print(f"Skipping distributed initialization on unsupported platform: {sys.platform}")

print(f"Process index: {jax.process_index() if hasattr(jax, 'process_index') else 0}")
print(f"Process count: {jax.process_count() if hasattr(jax, 'process_count') else 1}")

# Now your devices include all devices on all nodes if distributed, else local devices
devices = jax.devices()
print("All JAX devices:", devices)
print("Local devices:", jax.local_devices())
print("Global device count:", jax.device_count())

# --- Parameters ---
NX = 600
NY = 400
NSTEPS = 10000  # Adjust for your case
omega = 1.7
scale = 1

print(f"Domain size: NX={NX}, NY={NY}")
print(f"Scale: {scale}")

n_devices = jax.device_count()
assert NX % n_devices == 0, "NX must be divisible by total number of devices"
NXs = NX // n_devices  # Sub-domain size per device

# Lattice weights and velocity vectors
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = jnp.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],
               [0, 0, 1,  0, -1, 1,  1, -1, -1]])

@jax.jit
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u * u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    feq = wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[jnp.newaxis, :, :])
    return feq

@jax.jit
def Stream(g, c):
    def body(i, g):
        shift = (c.T[i][0], c.T[i][1])
        g = g.at[i].set(jnp.roll(g[i], shift=shift, axis=(0, 1)))
        return g
    g = lax.fori_loop(1, 9, body, g)
    return g

@jax.jit
def Collide(g):
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    g = g + omega * (feq - g)
    return g

def init_subdomain(nx, ny):
    x = jnp.arange(nx) + 0.5
    y = jnp.arange(ny) + 0.5
    X, Y = jnp.meshgrid(x, y)
    rho = jnp.ones((nx, ny))
    n = 1
    k = 2 * jnp.pi * n / ny
    u = jnp.array([0.1 * jnp.sin(k * Y.T), jnp.zeros((nx, ny))])
    f = equilibrium(rho, u)
    return f, rho

def step_fn(f, _):
    f = Stream(f, c)
    f = Collide(f)
    rho = jnp.einsum('ijk->jk', f)
    u = jnp.einsum('ai,ixy->axy', c, f) / rho
    amp_t = u[0, NXs // 2, NY // 8]
    return f, amp_t

p_step_fn = jax.pmap(step_fn, axis_name='devices', devices=devices)

# Initialize domain
f_init, rho = init_subdomain(NX, NY)
f_init = f_init.reshape(9, n_devices, NXs, NY).transpose(1, 0, 2, 3)

def run_simulation(f_init, steps):
    def body_fn(carry, _):
        f = carry
        f, amp = p_step_fn(f, None)
        return f, amp

    f_final, amps = lax.scan(body_fn, f_init, None, length=steps)
    return f_final, amps

start = time.time()
f_final, amplitudes = run_simulation(f_init, NSTEPS)
jax.device_get(f_final)
elapsed_time = time.time() - start
print(f"Elapsed time: {elapsed_time:.3f}s")

# Calculate BLUPS
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed_time / 1e9
print(f"Performance: {blups:.3f} BLUPS (Billion Lattice Updates Per Second)")

# Only plot on main process
if hasattr(jax, 'process_index') and jax.process_index() == 0:
    amplitudes_host = jax.device_get(amplitudes[:, 0])
    f_host = jax.device_get(f_final)
    f_last = f_host[-1]

    f_last = f_last.transpose(1, 0, 2).reshape(9, NX, NY)
    rho = jnp.einsum('ijk->jk', f_last)
    u = jnp.einsum('ai,ixy->axy', c, f_last) / rho

    # plt.plot(amplitudes_host / amplitudes_host[0])
    # plt.title("Amplitude Decay (device 0)")
    # plt.xlabel("Time step")
    # plt.ylabel("Amplitude")
    # plt.show()

    # plt.plot(u[0, NX // 2, :])
    # plt.title("Wave decay (final velocity profile)")
    # plt.xlabel("y")
    # plt.ylabel("Amplitude")
    # plt.show()
