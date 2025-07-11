
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
import time

devices = jax.devices()
print("All JAX devices:", devices)
print("Local devices:", jax.local_devices())
print("Global device count:", jax.device_count())


@jax.jit
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)    # This is 3*c*u
    usq = jnp.einsum('axy->xy', u*u)                # This is u^2
    wrho=jnp.einsum('i,xy->ixy', w, rho)            
    feq = wrho * (1 + cdot3u*(1 + 0.5*cdot3u) - 1.5*usq[np.newaxis,:,:])
    return feq
# Define the streaming function
@jax.jit
def Stream(g, c):
    def body(i, g):
        shift = (c.T[i][0], c.T[i][1])
        g = g.at[i].set(jnp.roll(g[i], shift=shift, axis=(0, 1)))
        return g

    g = lax.fori_loop(1, 9, body, g)
    return g
# Define the scattering function
@jax.jit
def Collide(g):
    rho = jnp.einsum('ijk->jk',g)
    u = jnp.einsum('ai,ixy->axy',c,g)/rho
    feq = equilibrium(rho,u)
    g = g + omega*(feq-g)
    return g

# dimensions of the 2D lattice and the Lattice parameters
NX=8000 #int(input("nx = "))
NY=8000 #int(input("ny = "))
# simulation parameters
scale  = 1               # set simulation size
#NX     = 32*scale        # domain size
#NY     = NX
NSTEPS = 10000*scale*scale # number of simulation time steps
NMSG   = 50*scale*scale  # show messages every NMSG time steps
vis    = False           # show visualisation; set to False for performance measurements
NVIS   = NMSG            # show visualisation every NVIS time steps
omega  = 1.7
tau    = 1/omega               # relaxation time
u_max  = 0.1/scale      # maximum velocity
nu     = (1/omega-1/2)/3     # kinematic shear viscosity
rho0   = 1.0               # rest density
#Re     = NX*u_max/nu     # Reynolds number; not used in the simulation itself
#
# Define the wights and channel velocities

dtype = jnp.float32
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],dtype=dtype) # weights
c = jnp.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]],dtype=dtype) # velocities, y components
#
# Define the gridpoints
x = jnp.arange(NX)+0.5    # the position of the points is half-ways in the interval
y = jnp.arange(NY)+0.5
X,Y = jnp.meshgrid(x,y)
# Initialize the density with 1.0 and the velocity with a sinusoidal
rho=jnp.ones((NX,NY),dtype=dtype)
n = 1     # multiples of the basic wavenumber try to play with n
k = 2*jnp.pi*n/NY  # Wavenumbers
u = jnp.array([u_max*jnp.sin(k*Y.T),jnp.zeros((NX,NY))])
f = equilibrium(rho,u) # create a local equilibrium initial condition. Note the term "local" it will decay due to viscosity

# amp=jnp.array(u[0,NX//2,NY//8]) # The amplitud of u in time
amp = jnp.zeros(NSTEPS + 1).at[0].set(u[0, NX//2, NY//8])

print(f"Domain size: NX={NX}, NY={NY}")
print(f"Scale: {scale}")
print(f"Number of steps: {NSTEPS}")
print(f"Omega: {omega}")
print(f"Rho0: {rho0}")

start = time.time()

# --------------------------
# Simulation Kernel (Optimized)
# --------------------------
from functools import partial

@partial(jax.jit, static_argnums=1)
def run_simulation(f_init, NSTEPS):
    def body(carry, _):
        f, amp, n = carry
        f = Stream(f, c)
        f = Collide(f)
        amp = amp.at[n + 1].set(u[0, NX // 2, NY // 8])
        return (f, amp, n + 1), None

    amp0 = jnp.zeros(NSTEPS + 1).at[0].set(u[0, NX // 2, NY // 8])
    (f_final, amp, _), _ = lax.scan(body, (f_init, amp0, 0), None, length=NSTEPS)
    return f_final, amp


print(f"Domain size: NX={NX}, NY={NY}")
print(f"Number of steps: {NSTEPS}")
print(f"Omega: {omega}, Viscosity: {nu:.4e}")

# --------------------------
# Run Simulation
# --------------------------
start = time.time()
f_final, amp = run_simulation(f, NSTEPS)
jax.block_until_ready(f_final)
elapsed_time = time.time() - start

total_updates = NX * NY * NSTEPS
blups = total_updates / elapsed_time / 1e9

print(f"Performance: {blups:.3f} BLUPS")
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# # Plotting the amplitude decay
# fig, ax = plt.subplots()
# ax.plot(amp/amp[0])
# ax.set_title('Aplitude decay')
# ax.set_xlabel('Time t')
# ax.set_ylabel('Amplitude')




