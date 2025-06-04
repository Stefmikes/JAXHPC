# %%
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from jax import lax
import cProfile
import pstats

# %%
# Function to calculate the equilibrium distribution

# @profile
def equilibrium(rho, u):
    cdot3u = 3 * jnp.einsum('ai,axy->ixy', c, u)    # This is 3*c*u
    usq = jnp.einsum('axy->xy', u*u)                # This is u^2
    wrho=jnp.einsum('i,xy->ixy', w, rho)            
    feq = wrho * (1 + cdot3u*(1 + 0.5*cdot3u) - 1.5*usq[np.newaxis,:,:])
    return feq
# Define the streaming function
# @profile
def Stream(g, c):
    def body(i, g):
        shift = (c.T[i][0], c.T[i][1])
        g = g.at[i].set(jnp.roll(g[i], shift=shift, axis=(0, 1)))
        return g

    g = lax.fori_loop(1, 9, body, g)
    return g
# Define the scattering function
# @profile
def Collide(g):
    rho = jnp.einsum('ijk->jk',g)
    u = jnp.einsum('ai,ixy->axy',c,g)/rho
    feq = equilibrium(rho,u)
    g = g + omega*(feq-g)
    return g

# %%
# dimensions of the 2D lattice and the Lattice parameters
NX=60 #int(input("nx = "))
NY=40 #int(input("ny = "))
# simulation parameters
scale  = 1               # set simulation size
#NX     = 32*scale        # domain size
#NY     = NX
NSTEPS = 200*scale*scale # number of simulation time steps
NMSG   = 50*scale*scale  # show messages every NMSG time steps
vis    = False           # show visualisation; set to False for performance measurements
NVIS   = NMSG            # show visualisation every NVIS time steps
omega  = 1.5
tau    = 1/omega               # relaxation time
u_max  = 0.1/scale      # maximum velocity
nu     = (1/omega-1/2)/3     # kinematic shear viscosity
rho0   = 1.0               # rest density
#Re     = NX*u_max/nu     # Reynolds number; not used in the simulation itself
#
# Define the wights and channel velocities
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
c = jnp.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]]) # velocities, y components
#
# Define the gridpoints
x = jnp.arange(NX)+0.5    # the position of the points is half-ways in the interval
y = jnp.arange(NY)+0.5
X,Y = jnp.meshgrid(x,y)
# Initialize the density with 1.0 and the velocity with a sinusoidal
rho=jnp.ones((NX,NY))
n = 1           # multiples of the basic wavenumber try to play with n
k = 2*jnp.pi*n/NY  # Wavenumbers
u = jnp.array([u_max*jnp.sin(k*Y.T),jnp.zeros((NX,NY))])
f = equilibrium(rho,u) # create a local equilibrium initial condition. Note the term "local" it will decay due to viscosity

# @profile
def main():
    global f, amp, u  # Ensure we modify the same variables
    amp = jnp.array(u[0, NX // 2, NY // 8])
    fig, ax = plt.subplots()
    ax.plot(u[0, NX // 2, :])
    ax.set_title('Wave decay')
    ax.set_xlabel('y')
    ax.set_ylabel('Amplitude')
    for n in range(1, 101):
        f = Stream(f, c)
        f = Collide(f)
        if n % 100 == 0:
            u = jnp.einsum('ai,ixy->axy', c, f) / rho
            ax.plot(u[0, NX // 2, :])
            amp = jnp.append(amp, u[0, NX // 2, NY // 8])

    # Plot amplitude decay
    fig, ax = plt.subplots()
    ax.plot(amp / amp[0])
    ax.set_title('Amplitude decay')
    ax.set_xlabel('Time t')
    ax.set_ylabel('Amplitude')

if __name__ == "__main__":
    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(15)  # Top 15 time-consuming calls
