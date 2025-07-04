import numpy as np
#import scipy.sparse as sp
#import scipy.sparse.linalg as la
#from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import time

# Function to calculate the equilibrium distribution
def equilibrium(rho, u):
    cdot3u = 3 * np.einsum('ai,axy->ixy', c, u)    # This is 3*c*u
    usq = np.einsum('axy->xy', u*u)                # This is u^2
    wrho=np.einsum('i,xy->ixy', w, rho)            
    feq = wrho * (1 + cdot3u*(1 + 0.5*cdot3u) - 1.5*usq[np.newaxis,:,:])
    return feq
# Define the streaming function
def Stream(g):
    for i in range(1,9):
        g[i,:,:] = np.roll(g[i,:,:], c.T[i], axis=[0,1])
    return g
# Define the scattering function
def Collide(g):
    rho = np.einsum('ijk->jk',g)
    u = np.einsum('ai,ixy->axy',c,g)/rho
    feq = equilibrium(rho,u)
    g = g + omega*(feq-g)
    return g

# ### Set up the simulation
# To this end we also have to define the array holding the streaming channels,
# read in the grid size parameters

# dimensions of the 2D lattice and the Lattice parameters
NX=400 #int(input("nx = "))
NY=300 #int(input("ny = "))
# simulation parameters
scale  = 1               # set simulation size
#NX     = 32*scale        # domain size
#NY     = NX
NSTEPS = 200*scale*scale # number of simulation time steps
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
w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]) # weights
c = np.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],  # velocities, x components
              [0, 0, 1,  0, -1, 1,  1, -1, -1]]) # velocities, y components
#
# Define the gridpoints
x = np.arange(NX)+0.5    # the position of the points is half-ways in the interval
y = np.arange(NY)+0.5
X,Y = np.meshgrid(x,y)
# Initialize the density with 1.0 and the velocity with a sinusoidal
rho=np.ones((NX,NY))
n = 1           # multiples of the basic wavenumber try to play with n
k = 2*np.pi*n/NY  # Wavenumbers
u = np.array([u_max*np.sin(k*Y.T),np.zeros((NX,NY))])
f = equilibrium(rho,u) # create a local equilibrium initial condition. Note the term "local" it will decay due to viscosity

# %%
amp=np.array(u[0,NX//2,NY//8]) # The amplitud of u in time
#
start = time.time()

# fig, ax = plt.subplots()
# ax.plot(u[0,NX//2,:])
# ax.set_title('Wave decay')
# ax.set_xlabel('y')
# ax.set_ylabel('Amplitude')
for n in range(1,10001):
    f = Stream(f)
    f = Collide(f)
    if n%100==0:
        #Tmeasure=np.append(Tmeasure,np.array(time))z
        u = np.einsum('ai,ixy->axy',c,f)/rho  
        # ax.plot(u[0,NX//2,:])
        amp=np.append(amp,u[0,NX//2,NY//8])

    
end = time.time()
elapsed_time = end - start
total_updates = NX * NY * 10000  # NSTEPS = 10000 here
blups = total_updates / elapsed_time / 1e9

print(f"Elapsed time: {elapsed_time:.3f} seconds")
print(f"Performance: {blups:.3f} BLUPS (Billion Lattice Updates Per Second)")

# Plotting the amplitude decay
# fig, ax = plt.subplots()
# ax.plot(amp/amp[0])
# ax.set_title('Aplitude decay')
# ax.set_xlabel('Time t')
# ax.set_ylabel('Amplitude')

# %%



