##### Python code for Toda system: To get the effective velocity, velocity correlation, diffusion coefficient of quasiparticles - ensemble averaged. 

## Importing required packages
##### Python code for Toda system: To get the effective velocity, velocity correlation, diffusion coefficient of quasiparticles - ensemble averaged. 

## Importing required packages
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, qr
from scipy.stats import gamma
from numba import njit, prange
from scipy.integrate import cumulative_trapezoid

# some parameters
N =500          # Number of particles
β = 1              # Inverse temperature
P = 1              # Pressure
m = 1              # mass of particles

realisations=10**6        # Number of realizations 
dt = 0.08                
Tfinal = 40
steps = int(np.round(Tfinal / dt))
t_grid = np.linspace(0, Tfinal, steps + 1)

xmin=-6
xmax=6
dx=0.025
bin_edges = np.arange(xmin, xmax,dx)
l_bin=len(bin_edges) - 1
bin_centers = (bin_edges[:-1] + bin_edges[1:])/2

# function to get the initial positions and momenta of particles
def initial_condition(N):
    u = gamma.rvs(a=P, scale=1/(4*β), size=N)
    r = -np.log(4*u)
    l = np.sum(r)
    q = np.concatenate(([0.0], np.cumsum(r[:-1])))   # replaces the for loop
    p = np.random.normal(loc=0, scale=np.sqrt(1/β), size=N)
    return q, r, p, l

# function to get the stretch, given 'q' at any time 
@njit                                                
def compute_r(q, l):
    N = len(q)
    r = np.roll(q, -1) - q
    r[-1] = (q[0] + l) - q[-1]
    return r

# function to get the Lax matrix
def build_L(p, q, l):
    M = np.diag(0.5 * p)                             # diagonal in one shot
    dq  = np.diff(q)                                     # q[j+1] - q[j], shape (N-1,)
    w   = 0.5 * np.exp(-dq / 2)
    w0  = 0.5 * np.exp(-(q[0] + l - q[-1]) / 2)
    M  += np.diag(w,  1)
    M  += np.diag(w, -1)
    M[0, -1] = M[-1, 0] = w0
    return M


# function to get the Lax matrix pair
def build_B(q, l):
    B  = np.zeros((len(q), len(q)))
    dq = np.diff(q)
    w  = -0.5 * np.exp(-dq / 2)         # upper diagonal (negative)
    w0 =  0.5 * np.exp(-(q[0] + l - q[-1]) / 2)
    B += np.diag(w,  1)
    B += np.diag(-w, -1)                 # antisymmetric: lower = +w
    B[0, -1] =  w0
    B[-1, 0] = -w0
    return B

# Velocity matrix
def build_V(p, q, l):
    M = np.diag(p.copy())
    dq      = np.diff(q)
    w       = 0.5 * np.exp(-dq / 2) * dq
    wrap    = q[0] + l - q[-1]
    w0      = 0.5 * np.exp(-wrap / 2) * wrap
    M      += np.diag(w,  1)
    M      += np.diag(w, -1)
    M[0, -1] = M[-1, 0] = w0
    return M

# function to get the Forces on the particles
@njit
def forces(q, l):
    r  = compute_r(q, l)
    er = np.exp(-r)
    return np.roll(er, 1) - er           # np.roll(er,1)[j] == er[(j-1)%N]

# function to get the quasiparticle position
@njit
def fun_Q(q,psi,l):
    Q=np.zeros(N)
    for alpha in range(N):
        Q[alpha]=np.sum(q * np.abs(psi[:, alpha])**2)   
    return Q

# function to get the evoltion of p,q, psi - RK4 method
# @njit
def rk4_step_p_q_psi(q,p,psi,l):
    
    k1_q = p
    k1_p = forces(q, l)
    B = build_B(q, l)
    k1_psi = B @ psi

    k2_q = p + 0.5 * dt * k1_p
    k2_p = forces(q+k1_q*0.5*dt, l)
    B = build_B(q+k1_q*0.5*dt, l)
    k2_psi = B @ (psi + 0.5 * dt * k1_psi)

    k3_q =  p + 0.5 * dt * k2_p
    k3_p = forces(q+k2_q*0.5*dt, l)
    B = build_B(q+k2_q*0.5*dt, l)
    k3_psi = B @ (psi + 0.5 * dt * k2_psi)

    k4_q = p + dt * k3_p
    k4_p = forces(q+k3_q*dt, l)
    B = build_B(q+k3_q*dt, l)
    k4_psi = B @ (psi + dt * k3_psi)

    q_new = q + (dt / 6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
    p_new = p + (dt / 6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    psi_new = psi + (dt / 6.0) * (k1_psi + 2*k2_psi + 2*k3_psi + k4_psi)

    # normalize each eigenvector (column-wise)
    # vectorised column-wise normalisation
    norms    = np.sqrt(np.sum(np.real(psi_new * np.conj(psi_new)),axis=0, keepdims=True))   # shape (1, n_eig)
    psi_new /= norms
    return q_new, p_new, psi_new
    
    
### velocity correlator storage: initial velocity 
v0_total      = np.zeros(l_bin)
count_v0      = np.zeros(l_bin, dtype=int)
Cv_total      = np.zeros((steps + 1, l_bin))

####### velocity at all times and count storage
v_total     = np.zeros((steps + 1, l_bin))
count_bins  = np.zeros((steps + 1, l_bin), dtype=int)
r_stretch   = np.zeros(realisations)

for i in range(realisations):
    #print(f"Realisation {i}/{realisations}", flush=True)

    q, r_, p, l = initial_condition(N)
    r_stretch[i] = np.sum(r_)

    L = build_L(p, q, l)
    eigvals1, eigvecs = np.linalg.eigh(L)
    eigvals=2*eigvals1
    idx = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx]
    psi = eigvecs[:, idx].copy()

    bin_indices = np.digitize(eigvals_sorted, bin_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < l_bin)

    for step in range(steps + 1):

        V=build_V(p,q,l)
        v_alpha = np.zeros(N)

        VP = V @ psi
        v_alpha = np.real(np.sum(np.conj(psi) * VP, axis=0))
    
        # to store initial velocity
        if step == 0:
            v_alpha_0 = v_alpha.copy()
            np.add.at(v0_total,    bin_indices[valid], v_alpha_0[valid])
            np.add.at(count_v0,    bin_indices[valid], 1)

        np.add.at(v_total[step],    bin_indices[valid], v_alpha[valid])
        np.add.at(count_bins[step], bin_indices[valid], 1)

        # velocity-velocity correlator
        np.add.at(Cv_total[step],bin_indices[valid],v_alpha[valid]*v_alpha_0[valid])

        if step < steps:
            q, p, psi = rk4_step_p_q_psi(q, p, psi, l)

# Mean velocity at t = 0
v0_mean = np.zeros(l_bin)
mask0 = count_v0 > 0
v0_mean[mask0] = v0_total[mask0] / count_v0[mask0]

# Averaging over realisations
veff_mean = np.zeros_like(v_total)
mask = count_bins > 0
veff_mean[mask] = v_total[mask] / count_bins[mask]

# Velocity correlator:  < v(t) v(0) >
Cv_mean = np.zeros_like(Cv_total)
Cv_mean[mask] = Cv_total[mask] / count_bins[mask]

# Connected correlator:C_v(t) = <v(t)v(0)> - <v(t)><v(0)>
Cv_connected = np.zeros_like(Cv_mean)
for b in range(l_bin):
    if mask0[b]:
        Cv_connected[:, b] = (Cv_mean[:, b]- veff_mean[:, b] * v0_mean[b])

D_lambda = np.zeros(l_bin)                    # D(lambda) = integration of C_v(t) dt from 0 to final time, using trapezoidal for integration
D_t = np.zeros_like(Cv_connected)        # cumulative time integral : diffusion constant as a function of time

for b in range(l_bin):
    D_lambda[b] = np.trapezoid(Cv_connected[:, b], t_grid)
    D_t[:, b]  = cumulative_trapezoid(Cv_connected[:, b], t_grid, initial=0)   

l_avg=np.sum(r_stretch)/(realisations*N)                ### Average stretch per particle
v_eff_Rtavg = np.mean(veff_mean, axis=0)