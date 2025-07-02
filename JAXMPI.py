from mpi4py import MPI
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NX = 600
NY = 400
NSTEPS = 1000
omega = 1.7

assert NX % size == 0, "NX must be divisible by number of MPI processes"
NX_local = NX // size

# Lattice velocities and weights
w = jnp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
c = jnp.array([[0, 1, 0, -1,  0, 1, -1, -1,  1],
               [0, 0, 1,  0, -1, 1,  1, -1, -1]])

def equilibrium(rho, u):
    cu = jnp.einsum('ia,axy->ixy', c, u)
    usq = jnp.sum(u * u, axis=0)
    feq = w[:, None, None] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*usq)
    return feq

def init_domain(nx, ny):
    x = jnp.arange(nx) + 0.5
    y = jnp.arange(ny) + 0.5
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    rho = jnp.ones((nx, ny))
    u = jnp.zeros((2, nx, ny))
    u = u.at[0].set(0.1 * jnp.sin(2 * jnp.pi * Y / ny))
    f = equilibrium(rho, u)
    return f

@jax.jit
def stream(f):
    f_stream = jnp.empty_like(f)
    for i in range(9):
        dx, dy = c[:, i]
        f_stream = f_stream.at[i].set(jnp.roll(f[i], shift=(dx, dy), axis=(0, 1)))
    return f_stream

@jax.jit
def collide(f):
    rho = jnp.sum(f, axis=0)
    u = jnp.einsum('ia,ixy->axy', c, f) / rho
    feq = equilibrium(rho, u)
    f_new = f + omega * (feq - f)
    return f_new, u, rho

def exchange_boundaries(f_host):
    left, right = rank - 1, rank + 1
    send_left = np.array(f_host[:, 1, :])
    send_right = np.array(f_host[:, -2, :])

    recv_left = np.empty_like(send_left)
    recv_right = np.empty_like(send_right)

    reqs = []

    if left >= 0:
        reqs.append(comm.Isend(send_left, dest=left, tag=11))
        reqs.append(comm.Irecv(recv_left, source=left, tag=12))
    if right < size:
        reqs.append(comm.Isend(send_right, dest=right, tag=12))
        reqs.append(comm.Irecv(recv_right, source=right, tag=11))

    MPI.Request.Waitall(reqs)

    if left >= 0:
        f_host[:, 0, :] = recv_left
    if right < size:
        f_host[:, -1, :] = recv_right

    return jnp.array(f_host)

def main():
    f = init_domain(NX_local + 2, NY)  # +2 for halo cells
    f_host = np.array(f)
    amplitudes = []

    start = time.time()
    for step in range(NSTEPS):
        f = stream(f)
        f_host = np.array(f)
        f_host = exchange_boundaries(f_host)
        f = jnp.array(f_host)
        f, u, rho = collide(f)

        amp = float(u[0, NX_local // 2, NY // 8])
        amplitudes.append(amp)

    elapsed = time.time() - start
    total_updates = NX * NY * NSTEPS
    blups = total_updates / elapsed / 1e9
    print(f"[Rank {rank}] Time: {elapsed:.2f}s | BLUPS: {blups:.3f}")

    # # Gather amplitudes to rank 0
    # all_amps = comm.gather(amplitudes, root=0)
    # if rank == 0:
    #     plt.plot(all_amps[0] / all_amps[0][0])
    #     plt.title("Amplitude Decay (Rank 0)")
    #     plt.xlabel("Time step")
    #     plt.ylabel("Amplitude")
    #     plt.grid()
    #     plt.show()

if __name__ == "__main__":
    main()
