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
np.random.seed(19680806)


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
N = 100
# state
x0 = np.array([[0],[0]])
x = np.zeros([2,N])
x[:,[0]] = x0


# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa = np.array([[k1], [k2]])
A_size = np.shape(A)
# fill in u = -Kappa*state_space:
# This gives state_space' = (A-B*Kappa)* state_space
# With poles at -2+-j, meaning eigenvalues of A-B*Kappa must be equal to (s-(-2-j))(s-(-2+j))
# This in turn is equal to s^2+4s+5, now the result of 
temp = s*np.eye(A_size[0])-(A-(B*Kappa))
print(temp)
# this results in: (k1+s)*(s+1) - 2.8*(k2+0.5) = k1*s + s^2 + s + k1 - 2,8*k2 - 0.5*2.8
# equal to s^2 + s(k1+1) + (k1-2.8*k2-1.4)
# this gives k1 = 3
# and 3-2.8*k2-1.4 = 5 gives k2 = -3.4/2.8
k1 = 3
k2 = -3.4/2.8
Kappa = np.array([[k1], [k2]]).T
print("Kappa", Kappa.shape)

# input
u0 = -Kappa@x0
u = np.zeros([N,1])
u[[0],:] = u0


""" Functions """
def calculate_FG(A,B,h,tau):
    # F values:
    A_size = np.shape(A)
    B_size = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    if singular == A_size[0]:
        #print("A is a not a singular matrix, thus invertable")
        # Since it is not singular and thus can be inverted the solutions is as follows:
        Fx = expm(A*h)
        #Fu = integral(h),(h-tau),(np.exp(A*s)*B)
        Fu = (expm(A*h)-expm(A*(h-tau)))@np.linalg.inv(A)@B
        #G1 = integral(h-tau),(0),(np.exp(A*s)*B)
        G1 = (expm(A*(h-tau))-np.eye(A_size[0]))@np.linalg.inv(A)@B
        # Construct matrices
        top = np.hstack((Fx, Fu))                                       # (n, n+m)
        bottom = np.hstack((np.zeros((B_size, A_size[0] + B_size))))    # (m, n+m)
        F = np.vstack((top, bottom))                                    # (n+m, n+m)
        G = np.vstack((G1, np.eye(B_size)))                             # (n+m, m)
        return Fx,Fu,G1,F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def convex_solve(F,G,Kappa):
    n = F.shape[0]
    m = G.shape[1]
    P = cp.Variable((n,n), symmetric = True)
    Q = cp.Variable((n,n), symmetric = True)
    
    #zero_return = [[0 for col in range(n)] for row in range(m)]
    constraints = [P >> 1e-4 * np.eye(3), Q >> 1e-4 * np.eye(3)]
    h_vals = np.linspace(0.01, 1, 200)
    for h in h_vals:
        tau_vals = np.linspace(0.01, h, h*200)
        for tau in tau_vals:
            Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau)
            temp = F-G@Kappa
            constraints.append(temp.T @ P @ temp - P + Q << 0)
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)
    prob.solve(solver=cp.SCS)
    if prob.status in ["infeasible"]:
        return np.nan,np.nan
    outP = P.value
    outQ = Q.value
    print(outP)
    print(outQ)
    return outP,outQ

""" Main Code """
""" Question 1 """
#for k in range(N):
    #u[:,k] = -Kappa*x[k,:]
    #print(u[:,k])
    #x[k+1,:] = Fx*x[k,:]+Fu*u[:,k-1]+G1*u[:,k]
    #print(x[k+1,:])

h_vals = np.linspace(0.01, 1, 100)
max_eig_mags = []
best_h = []
staticKappa = np.hstack((Kappa, np.zeros((1,1))))
for h in h_vals:
    Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau)
    A_stability = F-G@staticKappa
    eigenvalues = np.linalg.eigvals(A_stability)
    max_eig_mags.append(np.max(np.abs(eigenvalues)))
    if isclose(np.max(np.abs(eigenvalues)), 1, abs_tol=tolerance) == True:
        best_h.append(h)

print("Highest limit of sampling time given by h=",np.mean(best_h),"with tolerance equal to +-:",tolerance)
print("Lower limit will always approach 0")

""" Question 2 """

# Checking stability is different for NCS and max(eig(A))<1 is no longer sufficient!
# (F-G*Kappa_stability).T @ P*(F-G@Kappa_stability)-P ⪯ −Q
# Stable if for any h there exist a P and Q that satisfy above conditions
# this means we need a convex optimisation to find a P and Q to satisfy?
tau_vals = np.linspace(0.01, 1, 100)
max_eig_mags_withTau = []
best_h_withTau = []
best_tau = []
valid_h = []
valid_tau = []

for h in h_vals:
    for tau in tau_vals:
        if (tau < h):
            Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau)
            A_stability = F-G@staticKappa
            eigenvalues = np.linalg.eigvals(A_stability)
            temp = np.max(np.abs(eigenvalues))
            max_eig_mags_withTau.append(temp)
            if (np.max(np.abs(eigenvalues)) < 1):
                valid_h.append(h)
                valid_tau.append(tau)
        else:
            print("invalid tau")
            max_eig_mags_withTau.append(5)
        
print("Lower limit will always approach 0 for h and tau > h")

#possibleP,possibleQ = convex_solve(F,G,staticKappa) # h not needed since its already looping!
#for k in range(N):
    #u[:,k] = -Kappa*x[k,:]
    #print(u[:,k])
    #x[k+1,:] = Fx*x[k,:]+Fu*u[:,k-1]+G1*u[:,k]
    #print(x[k+1,:])


""" Question 3 """
# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa2 = np.array([[k1], [k2]])
# fill in u = -Kappa*state_space:
# This gives state_space' = (A-B*Kappa)* state_space
# With poles at -1 & -3, meaning eigenvalues of A-B*Kappa must be equal to (s-(-1))(s-(-3))
# This in turn is equal to s^2+4s+3, now the result of 
temp = s*np.eye(A_size[0])-(A-(B*Kappa))
print(temp)
# this results in: (k1+s)*(s+1) - 2.8*(k2+0.5) = k1*s + s^2 + s + k1 - 2,8*k2 - 0.5*2.8
# equal to s^2 + s(k1+1) + (k1-2.8*k2-1.4)
# this gives k1 = 3
# and 3-2.8*k2-1.4 = 3 gives k2 = -1.4/2.8
k1 = 3
k2 = -1.4/2.8
Kappa2 = np.array([[k1], [k2]]).T

# new input
u0 = -Kappa@x0-Kappa2@x0
u = np.zeros([N,1])
u[[0],:] = u0

h_vals = np.linspace(0.01, 1, 100)
max_eig_mags = []
best_h = []
staticKappa2 = np.hstack((Kappa2, np.zeros((1,1))))
for h in h_vals:
    Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau)
    A_stability = F-G@staticKappa-(F-G@staticKappa2)
    eigenvalues = np.linalg.eigvals(A_stability)
    max_eig_mags.append(np.max(np.abs(eigenvalues)))
    if isclose(np.max(np.abs(eigenvalues)), 1, abs_tol=tolerance) == True:
        best_h.append(h)

""" Plotting """
# For stability the maximum eigenvalues needs to be smaller then 1 of the A_stability matrix
# According to slides lecture 2 slide 18, using Ku is 0. Stability is valid for eigenvalues of closed loop A smaller then 1
plt.figure(figsize=(8, 4))
plt.plot(h_vals, max_eig_mags, label='Max |Eigenvalue|')
plt.axhline(1.0, color='red', linestyle='--', label='Stability Limit')
plt.xlabel("Sampling Time h")
plt.ylabel("Max Magnitude of Eigenvalues")
plt.title("Stability of Sampled-Data Closed-Loop System")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.scatter(valid_h, valid_tau, alpha=0.9)
plt.xlabel("Sampling Time h")
plt.ylabel("Delay Time Tau")
plt.title("Stability of Sampled-Data Closed-Loop System")
plt.legend()
plt.show()



