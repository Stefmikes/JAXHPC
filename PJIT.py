import time
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental.pjit import pjit

# ✅ Log platform and available devices (added block)
print(f"JAX platform: {jax.default_backend()}")
all_devices = jax.devices()
print(f"JAX is using {len(all_devices)} devices:")
for i, d in enumerate(all_devices):
    print(f"  Device {i}: {d}")

# ✅ Domain setup
NX, NY = 400, 300
NSTEPS = 10000
omega = 1.7
u_max = 0.1
nu = (1 / omega - 0.5) / 3

# ✅ Lattice parameters
dtype = jnp.float32
w = jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36], dtype)
c = jnp.array([[0,1,0,-1,0,1,-1,-1,1], [0,0,1,0,-1,1,1,-1,-1]], dtype)

# ✅ Setup JAX Mesh
devices = mesh_utils.create_device_mesh((len(jax.devices()),))
mesh = Mesh(devices, axis_names=('x',))

# ✅ Lattice functions
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)
    usq = jnp.einsum('axy->xy', u * u)
    wrho = jnp.einsum('i,xy->ixy', w, rho)
    return wrho * (1 + cdot3u * (1 + 0.5 * cdot3u) - 1.5 * usq[jnp.newaxis, :, :])

def collide(g):
    rho = jnp.einsum('ijk->jk', g)
    u = jnp.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    return g + omega * (feq - g), u

def stream(g):
    shifts = [(0,0), (0,1), (1,0), (0,-1), (-1,0),
              (1,1), (1,-1), (-1,-1), (-1,1)]
    return jnp.stack([jnp.roll(g[i], shift=shifts[i], axis=(0,1)) for i in range(9)])

# ✅ Initialize grid and velocity
x = jnp.arange(NX) + 0.5
y = jnp.arange(NY) + 0.5
X, Y = jnp.meshgrid(x, y, indexing='ij')
u0 = u_max * jnp.sin(2 * jnp.pi * Y / NY)
rho0 = jnp.ones((NX, NY), dtype=dtype)
v0 = jnp.zeros_like(u0)
u_init = jnp.array([u0, v0])
f0 = equilibrium(rho0, u_init).astype(dtype)

# ✅ Main loop with mesh and sharding
with mesh:
    sharding = NamedSharding(mesh, P('x', None, None))
    f = jax.device_put(f0, sharding)

    # ✅ Wrap lbm_step with pjit inside mesh context
    def lbm_step(f):
        f = stream(f)
        f, _ = collide(f)
        return f

    lbm_step = pjit(lbm_step, in_shardings=P('x', None, None), out_shardings=P('x', None, None))

    start = time.time()
    for _ in range(NSTEPS):
        f = lbm_step(f)
    end = time.time()

# ✅ Performance metrics
elapsed = end - start
total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed / 1e9

print(f"Ran on {len(devices)} GPUs: {blups:.3f} BLUPS")
print(f"Elapsed time: {elapsed:.2f} seconds")
print(f"Domain size: NX={NX}, NY={NY}")
print(f"Number of steps: {NSTEPS}")
print(f"Omega: {omega}, Viscosity: {nu:.4e}")
