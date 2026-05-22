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
from math import isclose
import cvxpy as cp
from collections import Counter
import mosek
import os
np.random.seed(5885221)


""" Variables """
tolerance = 1e-2
eta = 1e-6
a = 5 # first digit student number
b = 8 # third digit student number
c = 1 # last digid digit student number
# iterations
N = 250
K_Samples = 51
h_max = 0.5

path = os.path.dirname(os.path.abspath(__file__))
"""pre-calculations Kappa"""
A = np.array([[0,(0.5+c)],[(0.5+abs(a-b)),1]])
B = np.array([[1],[0]])
print("A=",A)
print("B=",B)
# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
K = np.array([[k1], [k2]])
A_sizeT = np.shape(A)
temp = s*np.eye(A_sizeT[0])-(A-(B@K.T))
temp_det = temp[0][0]*temp[1][1]-temp[0][1]*temp[1][0]
print(temp)
print(temp_det)
k1 = 5
k2 = 61/14
K = np.array([[k1], [k2]]).T # pre-transpose


# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa = np.array([[k1], [k2]])
A_sizeT = np.shape(A)
temp = s*np.eye(A_sizeT[0])-(A-(B@Kappa.T))
k1 = 2.5
k2 = -2/2.8
Kappa = np.array([[k1], [k2]]).T
# with extended state an extended Kappa is needed aswell using the static controller:
staticKappa = np.hstack((Kappa, np.zeros((1,1))))

""" Functions """
def calculate_FG_delay_extended(A,B,h,tau,K):
    # F values:
    n = A.shape[0]
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    # write as x_k+1 = x_k*(F_h-G_h*Kappa)
    if singular == n:
        Fh = expm(A*h)
        F_temp = expm(A*(h-tau))
        F_tau = expm(A*(tau))
        if tau<(h/2):
            Fh_tau = ((F_temp-np.eye(n))@(np.linalg.inv(A)))@B
            Fprev = np.zeros((m,m))
            G1 = ((F_tau-np.eye(n))@(np.linalg.inv(A)))@B
            Gcur = np.eye(m)
        elif tau>(h/2) and tau<h:
            Fh_tau = np.zeros((n,m))
            Fprev = np.eye(m)
            G1 = ((Fh-np.eye(n))@(np.linalg.inv(A)))@B
            Gcur = np.eye(m)
        else:
            # edge case tau == h/2 or tau == h
            Fh_tau = np.zeros((n,m))
            G1 = ((Fh - np.eye(n)) @ np.linalg.inv(A)) @ B
            Fprev = np.eye(m)
            Gcur = np.eye(m)
        F = np.block([[Fh, Fh_tau], [np.zeros((m,n)), Fprev]])
        G = np.vstack((G1, Gcur))
        A_cl = np.block([[Fh-K@G1, Fh_tau], [-K, Fprev]])
        return F,G,A_cl
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan
    
def calculate_FG_delay(A,B,h,tau):
    # F values:
    n = A.shape[0]
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    # write as x_k+1 = x_k*(F_h-G_h*Kappa)
    if singular == n:
        Fx = expm(A*h)
        temp = expm(A*(h-tau))
        Fu = ((Fx-temp)@(np.linalg.inv(A)))@B
        G1 = ((temp-np.eye(n))@(np.linalg.inv(A)))@B
        F = np.block([[Fx, Fu], [np.zeros((m,n)), np.zeros((m,m))]])
        G = np.vstack((G1, np.eye(m)))
        return F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def build_polytopic_model(A,B,K,h):
    tau_vals = [0.0, h/2.0, h]
    vertices = []
    n = K.shape[1]     # state dimension
    for tau in tau_vals:
        F, G = calculate_FG_delay(A,B,h,tau)
        n_aug = F.shape[0] # augmented dimension
        Fcl = F-G@K@(np.hstack([np.eye(n), np.zeros((n, n_aug - n))]))
        vertices.append(Fcl)
    return vertices

def convex_solve(A,B,Kappa,h):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable(((n+m),(n+m)), symmetric = True)
    stable_points = []
    constraints = [P >> eta * np.eye(n+m,n+m)]
    vertices = build_polytopic_model(A,B,Kappa,h)
    for Acl in vertices:
        constraints.append(Acl.T @ P @ Acl - P << -eta*np.eye(n+m))
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CVXOPT)
    if prob.status == 'optimal':
        return True
    else:
        return False
    
def convex_solve_NCS_CC(A_vals, prob_matrix):
    n = A_vals[0].shape[0]
    M = len(A_vals)
    #print(A_vals[1])
    P_var_list = [cp.Variable((n, n), symmetric=True) for _ in range(n)]
    constraints = [P >> 1e-4 * np.eye(n) for P in P_var_list]
    #use MSS, p_ij needs to be set up as a matrix with the transition probabilities
    k = -1
    for i in range(n):
        expected = 0
        for j in range(n):
            k+=1
            #print(k)
            #print(prob_matrix[i, j])
            expected += prob_matrix[i, j] * (A_vals[k].T @ P_var_list[j] @ A_vals[k])
        constraints.append(P_var_list[i] - expected >> eta * np.eye(n))
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.CVXOPT, verbose=False, max_iters=5000, eps=1e-5)
    if prob.status == 'optimal':
        return True
    else:
        return False

def all_states(h, A, B, K):
    F1,G1,A_cl1 = calculate_FG_delay_extended(A, B, h, h/3,K)
    F2,G2,A_cl2 = calculate_FG_delay_extended(A, B, h, 3*h/4,K)
    F3,G3,A_cl3 = calculate_FG_delay_extended(A, B, h, h/4,K)

    out = [
        A_cl1,
        A_cl1 @ A_cl2,
        A_cl1 @ A_cl3,
        A_cl2 @ A_cl1,
        A_cl2,
        A_cl2 @ A_cl3,
        A_cl3 @ A_cl1,
        A_cl3 @ A_cl2,
        A_cl3
    ]
    return out

""" Main Code """
""" Question 4 """

h_vals = np.linspace(0.01, 0.5, N)
spectral_radius_switch = []
stable_h = []


for i,h in enumerate(h_vals):
    print("progress Q4:",(i/N)*100,"%")
    feasibility = convex_solve(A,B,Kappa,h)
    stable_h.append(feasibility)



# This is more strict then the eigenvalue approach but less strict then the common lyapunov over all values of tau
# Mention this in report!

""" Question 5 """
n = A.shape[0]
r1 = 0.2
r2 = 0.2
# [ 0.1, 0.8, 0.1 ]
# [ 0.5, 0.3, 0.2 ]
# [ 0.2, 0.2, 0.6 ]
prob_matrix = np.vstack((np.array([0.1, 0.8, 0.1]),np.array([0.5, 0.3, 0.2]),np.array([r1, r2, 1-r1-r2])))
print(prob_matrix)
stable_h_stoch = []
for i,h in enumerate(h_vals):
    print("progress Q5:",(i/N)*100,"%")
    Acl_list = all_states(h,A,B,Kappa)
    stability = convex_solve_NCS_CC(Acl_list,prob_matrix)
    stable_h_stoch.append(stability)

#stable_h_e = [h for h, r in zip(h_vals, stable_h_stoch) if r < 1]
#print("Stable h values :[", min(stable_h_e),",",max(stable_h_e),"]")
M = 50
r_vals = np.linspace(0.01, 0.5, M)
stability = np.zeros((len(r_vals), len(h_vals)))
for i,r in enumerate(r_vals):
    print("progress Q5:",(i/M)*100,"%")
    prob_matrix = np.vstack((np.array([0.1, 0.8, 0.1]),np.array([0.5, 0.3, 0.2]),np.array([r, r, 1-2*r])))
    for j,h in enumerate(h_vals):
        Acl_list = all_states(h,A,B,Kappa)
        stability[i,j] = convex_solve_NCS_CC(Acl_list,prob_matrix)


""" Plotting """

# Q4

plt.plot(h_vals, stable_h, 'o-')
plt.xlabel("h")
plt.ylabel("Feasible (1=True, 0=False)")
plt.title("Polytopic stability region")
plt.grid(True)
plt.tight_layout()
plt.savefig("Q4_polytopicStability.png")
plt.show()

# Q5
# A
plt.plot(h_vals, stable_h_stoch, 'o-')
plt.xlabel("h")
plt.ylabel("Feasible (1=True, 0=False)")
plt.title("stochastic stability region")
plt.grid(True)
plt.tight_layout()
plt.savefig("Q5_fixedRStochStability.png")
plt.show()

# B
plt.imshow(stability, extent=[h_vals[0], h_vals[-1], r_vals[0], r_vals[-1]],
           origin='lower', aspect='auto', cmap='Greens')
plt.colorbar(label="Feasible (1=True, 0=False)")
plt.xlabel("h")
plt.ylabel("r")
plt.title("Stochastic stability region (heatmap)")
plt.tight_layout()
plt.savefig("Q5_heatmapStochStability.png")
plt.show()