import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio

# Create frames directory if it doesn't exist
os.makedirs('frames', exist_ok=True)
# === Functions ===
def equilibrium(rho, u):
    cdot3u = 3 * np.einsum('ai,axy->ixy', c, u)
    usq = np.einsum('axy->xy', u*u)
    wrho = np.einsum('i,xy->ixy', w, rho)
    feq = wrho * (1 + cdot3u*(1 + 0.5*cdot3u) - 1.5*usq[np.newaxis,:,:])
    return feq

def Stream(g):
    for i in range(1,9):
        g[i,:,:] = np.roll(g[i,:,:], c.T[i], axis=[0,1])
    return g

def Collide(g):
    rho = np.einsum('ijk->jk', g)
    u = np.einsum('ai,ixy->axy', c, g) / rho
    feq = equilibrium(rho, u)
    g = g + omega * (feq - g)
    return g

# === Simulation Setup ===
NX, NY = 60, 40
NSTEPS = 200
omega = 1.7
u_max = 0.1
rho0 = 1.0

w = np.array([4/9] + [1/9]*4 + [1/36]*4)
c = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],
              [0, 0, 1,  0, -1, 1,  1, -1, -1]])

x = np.arange(NX) + 0.5
y = np.arange(NY) + 0.5
X, Y = np.meshgrid(x, y)

rho = np.ones((NX, NY))
k = 2 * np.pi / NY
u = np.array([u_max * np.sin(k * Y.T), np.zeros((NX, NY))])
f = equilibrium(rho, u)

# === Tracking amplitude and profiles ===
amp = [u[0, NX//2, NY//8]]
profiles = [u[0, NX//2, :].copy()]  # store initial wave

# === Main loop ===
start = time.time()

profiles = []

for n in range(1, 10001):
    f = Stream(f)
    f = Collide(f)

    if n % 100 == 0:
        u = np.einsum('ai,ixy->axy', c, f) / rho
        profiles.append(u[0, NX//2, :].copy())
        amp.append(u[0, NX//2, NY//8])  

               # Plot current profile and save as PNG
        plt.figure()
        plt.plot(u[0, NX//2, :])
        plt.title(f'Wave profile at step {n}')
        plt.xlabel('y')
        plt.ylabel('Amplitude')
        plt.ylim(-u_max, u_max)  # <-- FIXED y-axis scale
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'frames/frame_{n:05d}.png')
        plt.close()

print(f"Collected {len(profiles)} profiles")
 # store profile along y
end = time.time()

# === Performance ===
elapsed_time = end - start
blups = NX * NY * NSTEPS / elapsed_time / 1e9
print(f"Elapsed time: {elapsed_time:.3f} seconds")
print(f"Performance: {blups:.3f} BLUPS")

# Plot all collected profiles
fig, ax = plt.subplots()
for profile in profiles:
    ax.plot(profile)  # each profile is a snapshot of u at different times

ax.set_title('Wave decay')
ax.set_xlabel('y')
ax.set_ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot amplitude decay ===
fig, ax = plt.subplots()
amp = np.array(amp)
ax.plot(amp/amp[0])
ax.set_title('Aplitude decay')
ax.set_xlabel('Time t')
ax.set_ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

with imageio.get_writer('wave_decay.gif', mode='I', duration=0.1) as writer:
    for n in range(100, 10001, 100):
        filename = f'frames/frame_{n:05d}.png'
        image = imageio.imread(filename)
        writer.append_data(image)

