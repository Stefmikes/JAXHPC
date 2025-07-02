import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
import time

# LBM parameters
NX, NY = 60, 60  # square cavity
NSTEPS = 10000
omega = 0.7  # relaxation parameter
tau = 1 / omega
u_lid = 0.1  # lid velocity

# Lattice velocities and weights
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = jnp.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],  # cx
               [0, 0, 1,  0, -1, 1,  1, -1, -1]])  # cy

# Compute equilibrium distribution
@jax.jit
def equilibrium(rho, u):
    cu = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u * u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    feq = wrho * (1 + cu * (1 + 0.5 * cu) - 1.5 * usq[np.newaxis, :, :])
    return feq

# Streaming step
@jax.jit
def stream(f):
    def body(i, f):
        shift = (c.T[i][0], c.T[i][1])
        f = f.at[i].set(jnp.roll(f[i], shift=shift, axis=(0, 1)))
        return f
    f = lax.fori_loop(0, 9, body, f)
    return f

# Collision step
@jax.jit
def collide(f):
    rho = jnp.sum(f, axis=0)
    u = jnp.einsum('ai,ixy->axy', c, f) / rho
    feq = equilibrium(rho, u)
    f = f + omega * (feq - f)
    return f

# Initialize fields
rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
f = equilibrium(rho, u)

# Main time loop
start = time.time()

for step in range(NSTEPS):
    # Streaming
    f = stream(f)

    # Bounce-back boundaries (no-slip on walls)
    # South wall (y=0)
    f = f.at[2, :, 0].set(f[4, :, 0])
    f = f.at[5, :, 0].set(f[7, :, 0])
    f = f.at[6, :, 0].set(f[8, :, 0])

    # North wall (y=NY-1) - Moving lid
    rho_top = jnp.sum(f[:, :, -1], axis=0)  # shape (NX,)
    u_top = jnp.array([u_lid * jnp.ones_like(rho_top), jnp.zeros_like(rho_top)])
    feq_top = equilibrium(rho_top[None, :], u_top[:, None, :])[:, :, 0]  # careful broadcasting

    f = f.at[4, :, -1].set(feq_top[4, :])
    f = f.at[7, :, -1].set(feq_top[7, :])
    f = f.at[8, :, -1].set(feq_top[8, :])

    # West wall (x=0)
    f = f.at[1, 0, :].set(f[3, 0, :])
    f = f.at[5, 0, :].set(f[7, 0, :])
    f = f.at[8, 0, :].set(f[6, 0, :])

    # East wall (x=NX-1)
    f = f.at[3, -1, :].set(f[1, -1, :])
    f = f.at[7, -1, :].set(f[5, -1, :])
    f = f.at[6, -1, :].set(f[8, -1, :])

    # Collision
    f = collide(f)

jnp.linalg.norm(f).block_until_ready()
print("Elapsed time:", time.time() - start)

# Post-process: compute velocity field
rho = jnp.sum(f, axis=0)
u = jnp.einsum('ai,ixy->axy', c, f) / rho

# Plot velocity field
X, Y = np.meshgrid(np.arange(NX), np.arange(NY))
plt.figure(figsize=(6, 6))
plt.streamplot(X, Y, np.array(u[0]).T, np.array(u[1]).T, density=1.2)
plt.title("Lid-driven cavity flow")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
