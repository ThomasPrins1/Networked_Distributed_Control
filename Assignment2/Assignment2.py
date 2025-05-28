# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 12:59:05 2025

@author: user
@studentID: 5885221
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from sympy import symbols
import sympy as sp
from math import isclose
import cvxpy as cp
from collections import Counter
np.random.seed(5885221)


""" Variables """
tolerance = 1e-2
eta = 1e-4
a = 5 # first digit student number
b = 8 # third digit student number
c = 1 # last digid digit student number
A = np.array([[0,(0.5-c)],[(0.2+a-b),-1]])
B = np.array([[1],[0]])
print("A=",A)
print("B=",B)
# sampling interval
#h = 0.5
# delay
tau = 0
# iterations
N = 75
kappaSamples = 21

# eigenvalues K:
k1 = 3
k2 = -3.4/2.8
Kappa1 = np.array([[k1], [k2]]).T
# eigenvalues K2:
k1 = 3
k2 = -1.4/2.8
Kappa2 = np.array([[k1], [k2]]).T

""" Functions """
def calculate_FG(A,B,h,tau,include_delay):
    # F values:
    n = A.shape[0]
    A_full_size = np.shape(A)
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    if singular == A_full_size[0]:
        Fx = expm(A*h)
        Fu1 = np.linalg.solve(A, expm(A * (h - tau)) - np.eye(n)) @ B
        Fu2 = np.zeros((n,m))
        Fu = np.hstack((Fu1, Fu2))
        G1 = np.linalg.solve(A, expm(A * h) - expm(A * (h - tau))) @ B
        if include_delay == True:
            G_added = np.zeros((n,m))
            F = np.block([[Fx, Fu], [np.eye(n), np.zeros((n, n))]])
        else:
            G_added = np.eye(n,m)
            F = np.block([[Fx, Fu], [np.zeros((n, n)), np.zeros((n, n))]])
        G = np.vstack((G1, G_added))  
        return Fx,Fu,G1,F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def convex_solve_NCS_PL(F_vals):
    n = F_vals[0].shape[0]
    P = cp.Variable((n,n), symmetric = True)
    Q = cp.Variable((n,n), symmetric = True)

    constraints = [P >> 1e-4 * np.eye(n), Q >> 1e-4 * np.eye(n)]
    for F in F_vals:
        constraints.append(F.T @ P @ F - P + Q << 1e-4 * np.eye(n))
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(solver=cp.SCS)
    if prob.status == 'optimal':
        return True
    else:
        return False
    
def convex_solve_NCS_CC(A_vals, prob_matrix):
    n = len(A_vals)
    P_var_list = [cp.Variable((n, n), symmetric=True) for _ in range(4)]
    constraints = [P >> 1e-4 * np.eye(n) for P in P_var_list]
    #use MSS, p_ij needs to be set up as a matrix with the transition probabilities
    for i in range(n):
        expected = 0
        for j in range(n):
            expected += prob_matrix[i, j] * (A_vals[j].T @ P_var_list[j] @ A_vals[j])
        #expected = sum(prob_matrix[i, j] * (A_vals[j].T @ P_var_list[j] @ A_vals[j]) for j in range(n))
        constraints.append(P_var_list[i] - expected >> 1e-4 * np.eye(n))
    prob = cp.Problem(cp.Minimize(cp.trace(P_var_list[0])), constraints)
    prob.solve(solver=cp.SCS)
    if prob.status == 'optimal':
        return True
    else:
        return False
    
def convex_solve_NCS_Poly(F_list):
    n = F_list[0].shape[0]
    P = cp.Variable((n, n), symmetric=True)
    gamma = 1e-6
    constraints = [P >> 1e-4 * np.eye(n)]
    #print(F_list)
    for F in F_list:
        constraints.append(F.T @ P @ F - P << -gamma * P)
    
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(solver=cp.SCS)
    if prob.status == 'optimal':
        return True
    else:
        return False

def switched_state(h, state, A, B, K1_partial, K2_partial):
    rng = np.random.default_rng() # this needs to be a random state
    mode = rng.choice(2, 1)
    if state == True:
        _,_,_,F1,G1 = calculate_FG(A, B, h, 0, False)
        K1 = np.hstack((K1_partial, np.zeros_like(K1_partial)))
        _,_,_,F2,G2 = calculate_FG(A, B, h, 0, False)
        K2 = np.hstack((K2_partial, np.zeros_like(K2_partial)))
    elif state == False:
        _,_,_,F1,G1 = calculate_FG(A, B, 2*h, 0, False)
        K1 = np.hstack((K2_partial, np.zeros_like(K2_partial)))
        _,_,_,F2,G2 = calculate_FG(A, B, 2*h, 0, False)
        K2 = np.zeros((1, 4))  # zero control input
    A_stab1 = F1 - G1 @ K1
    A_stab2 = F2 - G2 @ K2
    return A_stab1,A_stab2

def all_states(h, A, B, K1_partial, K2_partial):
    _,_,_,F1,G1 = calculate_FG(A, B, h, 0, False)
    K1 = np.hstack((K1_partial, np.zeros_like(K1_partial)))
    K2 = np.hstack((K2_partial, np.zeros_like(K2_partial)))
    _,_,_,F2,G2 = calculate_FG(A, B, 2*h, 0, False)

    F_stab1 = F1 - G1 @ K1
    F_stab2 = F1 - G1 @ K2
    F_stab3 = F2 - G2 @ K2
    F_stab4 = F2

    out = [
        F_stab3 @ F_stab2,  # Kappa1 → Kappa2
        F_stab4  @ F_stab2,  # Kappa1 → zero
        F_stab3 @ F_stab1,  # Kappa2 → Kappa2
        F_stab4  @ F_stab1   # Kappa2 → zero
    ]
    return out

def compute_polytopic_F(A3, B3, Kappa, h):
    out = []
    tau_vals = np.linspace(0.1, h+0.1, 10)
    for tau in tau_vals:
        _,_,_,F,G = calculate_FG(A3,B3,h,tau,True)
        M = F - G @ Kappa
        #temp = (expm(M * (h - tau)) @ expm(M * tau))
        out.append(M)
    for i, F in enumerate(out):
        eigs = np.linalg.eigvals(F)
        rho = np.max(np.abs(eigs))
        #print(f"  τ[{i}] spectral radius = {rho:.4f}")
    return out,tau_vals


def checkElements(F,G,staticKappa_vals):
    out = []
    i=0
    for i, K in enumerate(staticKappa_vals):
        A_cl = F-G@(K.reshape(1, -1))
        eigenvalues = np.linalg.eigvals(A_cl)
        if (np.max(np.abs(eigenvalues)) < 1):
            out.append(i) # index of Kappa
    return out

""" Main Code """
""" Question 1 """

h_vals = np.linspace(0.01, 1, N)
spectral_radius_switch = []
stable_h = []


for i,h in enumerate(h_vals):
    print("progress Q1:",(i/N)*100,"%")
    #A_cl1,A_cl2 = switched_state(h,state,A,B,Kappa1,Kappa2)
    # cause periodic we can take x_k+2^e = A_cl2*A_cl1 x_k^e
    # This also means that we do not need to bother with convex solvers,
    # Since the actual sampling freq does not change over the period.
    # This gives a very conservative answer though. Instead we use convext optimization of a combination of all functions.
    # Meaning we search for a P & Q valid for all combinations of modes and states values for F_cl which are combined in F_list
    # so one convex solver uses constraints including all combinations for F.
    F_list = all_states(h,A,B,Kappa1,Kappa2)
    #spectral_radius_switch.append(temp)
    stability = convex_solve_NCS_PL(F_list)
    if stability:
        stable_h.append(h)
stable_h_e = [h for h, r in zip(h_vals, stable_h) if r < 1]
print("Stable h values :[", min(stable_h_e),",",max(stable_h_e),"]")

""" Question 2 """
# Markov chain made in drawio and "prob_matrix" shows the probabilities of transitions
#
n = A.shape[0]
p = 0.2
q = 0.3
# modes is:
## 0 |Kappa1 → Kappa2 prob()
## 1 |Kappa1 → zero
## 2 |Kappa2 → Kappa2
## 3 |Kappa2 → zero
prob_matrix = np.block([[np.zeros((n,n)), np.array([[q,1-q],[q,1-q]])],[np.array([[p,1-p],[p,1-p]]), np.zeros((n,n))]])
stable_h_stoch = []
state = 0
# need an A matrix for each specific state somehow
# this is defined by A = F-G*K, where K will change for each iteration (use all_state from previous question!)
for i,h in enumerate(h_vals):
    print("progress Q2:",(i/N)*100,"%")
    state = not state # this switches between h1 and h2
    rng = np.random.default_rng() # this needs to be a random state
    mode = rng.choice(2, 1)
    A_list = all_states(h,A,B,Kappa1,Kappa2)
    stability = convex_solve_NCS_CC(A_list,prob_matrix)
    if stability:
        stable_h_stoch.append(h)
#stable_h_e = [h for h, r in zip(h_vals, stable_h_stoch) if r < 1]
#print("Stable h values :[", min(stable_h_e),",",max(stable_h_e),"]")


""" Question 3 """
# new A and B!
A3 = np.array([[0.3+a-b,0],[1,0.5+c]])
B3 = np.array([[1],[0]])
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa3 = np.array([[k1], [k2]])
A_sizeT = np.shape(A3)
print(Kappa3)
# fill in u = -Kappa*state_space:
# This gives state_space' = (A-B*Kappa)* state_space
# With poles at -2+-j, meaning eigenvalues of A-B*Kappa must be equal to (s-(-2-j))(s-(-2+j))
# This in turn is equal to s^2+4s+5, now the result of 
temp = s*np.eye(A_sizeT[0])-(A3-(B3@Kappa3.T))
print(temp)

# this results in: (k1+s+2.7)*(s-1.5) - k2 = k1*s-1.5*k1+s^2+1.2*s-4.05-k2
# equal to s^2 + s(k1 +1.2) + (-1.5*k1-k2-4.05)
# this gives k1 = 4-1.2 = 2.8
# and -1.5*2.8-k2-4.05 = 5 gives k2 = -(5+1.5*2.7+1.5*2.8) = -13.25
k1 = 2.8
k2 = -13.25
Kappa3 = np.array([[k1], [k2]]).T
stable_h_delay = []
#h_vals_small = np.linspace(0.001, 0.3, N)
tau_vals = np.linspace(0.02, 1, N)
max_eig_mags_total = []
staticKappa3 = np.hstack((Kappa3, np.zeros((1,2))))
# test!
staticKappa = np.hstack((Kappa1, np.zeros((1,2))))
#
for i,h in enumerate(h_vals):
    print("progress Q3.1:",(i/N)*100,"%")
    max_eig_mags = []
    for tau in tau_vals:
        _,_,_,F,G = calculate_FG(A3,B3,h,tau,False)
        A_cl = F-G@staticKappa3
        eigenvalues = np.linalg.eigvals(A_cl)
        temp = np.max(np.abs(eigenvalues))
        max_eig_mags.append(temp)
    max_eig_mags_total.append(max_eig_mags)

stable_h_tau_delay = []
for i,h in enumerate(h_vals):
    print("progress Q3.2:",(i/N)*100,"%")
    F_list,tau_range = compute_polytopic_F(A3, B3, staticKappa3, h)
    stability = convex_solve_NCS_Poly(F_list)
    print(stability)
    if stability:
        for tau in tau_range:
            stable_h_tau_delay.append((h, tau))

""" Question 4 """


""" Plotting """

# Q1
plt.figure(figsize=(8, 4))
plt.plot(h_vals, [1 if h in stable_h else 0 for h in h_vals], 'bo', label='Stability')
plt.xlabel("Sampling time h")
plt.ylabel("Stability Indicator")
plt.title("Stability via Common Quadratic Lyapunov Function")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Q2
plt.figure(figsize=(8, 4))
plt.plot(h_vals, [1 if h in stable_h_stoch else 0 for h in h_vals], 'bo', label='Stability')
plt.xlabel("Sampling time h")
plt.ylabel("Stability Indicator")
plt.title("Stable values of h for ω-sequence switching")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
# Q3.1
# Convert to NumPy array for plotting
Z1 = np.array(max_eig_mags_total)

# Create meshgrid for contour plot
H, T = np.meshgrid(tau_vals, h_vals)  # note the order: row = h, column = tau

# Plot contour where max eigenvalue magnitude = 1
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, H, Z1, levels=50, cmap='viridis')
plt.colorbar(contour, label='Max |eig(F)|')

# Add stability boundary (e.g., contour where max eig = 1)
cs = plt.contour(T, H, Z1, levels=[1.0], colors='red', linewidths=2)
plt.clabel(cs, inline=True, fmt='Stability boundary', fontsize=10)

# Add diagonal line τ = h
min_val = max(np.min(T), np.min(H))
max_val = min(np.max(T), np.max(H))
plt.plot([min_val, max_val], [min_val, max_val], color='white', linestyle='--', linewidth=2, label='τ = h')

# Labels
plt.xlabel('τ (Delay)')
plt.ylabel('h (Sampling Time)')
plt.title('Max Eigenvalue Magnitude of F(h, τ)')
plt.grid(True)
plt.tight_layout()
plt.show()
# Q3.2
# Convert to NumPy array for plotting
stable_h_tau_delay = np.array(stable_h_tau_delay)

plt.figure(figsize=(8, 6))
if stable_h_tau_delay.size > 0:
    plt.scatter(stable_h_tau_delay[:, 0], stable_h_tau_delay[:, 1],
                s=10, c='green', label='Stable (h, τ)')
else:
    print("⚠️ No stable points found!")

plt.xlabel("h (Sampling Interval)")
plt.ylabel("τ (Delay)")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Stable (h, τ) Region via Polytopic Approximation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Q4

