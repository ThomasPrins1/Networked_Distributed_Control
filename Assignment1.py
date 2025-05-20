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
kappaSamples = 9
# state
x0 = np.array([[0],[0]])
x = np.zeros([2,N])
x[:,[0]] = x0


# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa = np.array([[k1], [k2]])
A_sizeT = np.shape(A)
# fill in u = -Kappa*state_space:
# This gives state_space' = (A-B*Kappa)* state_space
# With poles at -2+-j, meaning eigenvalues of A-B*Kappa must be equal to (s-(-2-j))(s-(-2+j))
# This in turn is equal to s^2+4s+5, now the result of 
temp = s*np.eye(A_sizeT[0])-(A-(B*Kappa))
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
def calculate_FG(A,B,h,tau,adj):
    # F values:
    n = A.shape[0]
    A_full_size = np.shape(A)
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    if singular == A_full_size[0]:
        Fx = expm(A*h)
        if adj == True:
            Fu1 = np.linalg.solve(A, expm(A * (h - tau)) - np.eye(n)) @ B
            Fu2 = np.zeros((n,m))
            Fu = np.hstack((Fu1, Fu2))
            G1 = (expm(A*(h-tau))-np.eye(n))@np.linalg.inv(A)@B
            LL = np.eye(n)
            LR = np.zeros((n, n))
            G_added = np.zeros((n,m))
            F = np.block([[Fx, Fu], [LL, LR]])
        else:
            Fu = (expm(A*h)-expm(A*(h-tau)))@np.linalg.inv(A)@B
            G1 = (expm(A*(h-tau))-np.eye(n))@np.linalg.inv(A)@B
            LL = np.zeros((m, n))
            LR = np.zeros((m, m))
            G_added = np.eye(m)
            F = np.block([[Fx, Fu], [LL, LR]])
        G = np.vstack((G1, G_added))  

    if singular == A_full_size[0]:
        #print("A is a not a singular matrix, thus invertable")
        # Since it is not singular and thus can be inverted the solutions is as follows:
        # Fu = integral(h),(h-tau),(np.exp(A*s)*B)
        # G1 = integral(h-tau),(0),(np.exp(A*s)*B)
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
            Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau,False)
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

def checkElements(F,G,staticKappa_vals):
    out = []
    i=0
    for i, K in enumerate(staticKappa_vals):
        A_cl = F-G@(K.reshape(1, -1))
        eigenvalues = np.linalg.eigvals(A_cl)
        if (np.max(np.abs(eigenvalues)) < 1):
            out.append(i) # index of Kappa
    #print(out)
    return out

""" Main Code """
""" Question 1 """

h_vals = np.linspace(0.01, 1, 50)
max_eig_mags = []
best_h = []
staticKappa = np.hstack((Kappa, np.zeros((1,1))))
for h in h_vals:
    Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau,False)
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
tau_vals = np.linspace(0.01, 1, 50)
max_eig_mags_withTau = []
best_h_withTau = []
best_tau = []
valid_h = []
valid_tau = []
bestKappaIndex = []
Kappa_test1 = np.linspace(-10, 10, kappaSamples)
Kappa_test2 = np.linspace(-10, 10, kappaSamples)
Kappa_test3 = np.zeros((1,kappaSamples))
P,Q,R = np.meshgrid(Kappa_test1, Kappa_test2, Kappa_test3, indexing='ij')
staticKappa_vals = np.stack([P,Q,R], axis=-1).reshape(-1, 3)
print(staticKappa_vals)
for h in h_vals:
    for tau in tau_vals:
        if (tau < h):
            Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau,False)
            bestKappaIndex.extend(checkElements(F,G,staticKappa_vals))
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
test = Counter(bestKappaIndex).most_common(1)
print("The most valid values for K are:", staticKappa_vals[test[0][0], :])
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
temp = s*np.eye(A_sizeT[0])-(A-(B*Kappa))
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

max_eig_mags2_total = []
best_h2 = []
staticKappa2 = np.hstack((Kappa,Kappa2))
bestKappaIndex2 = []
Kappa_test1 = np.linspace(-10, 10, kappaSamples)
Kappa_test2 = np.linspace(-10, 10, kappaSamples)
Kappa_test3 = np.linspace(-10, 10, kappaSamples)
Kappa_test4 = np.linspace(-10, 10, kappaSamples)
P,Q,R,T = np.meshgrid(Kappa_test1, Kappa_test2, Kappa_test3, Kappa_test4, indexing='ij')
staticKappa_vals = np.stack([P,Q,R,T], axis=-1).reshape(-1, 4)

for h in h_vals:
    print("progress:",h)
    max_eig_mags2 = []
    for tau in tau_vals:
        Fx,Fu,G1,F,G = calculate_FG(A,B,h,tau,True)
        bestKappaIndex2.extend(checkElements(F,G,staticKappa_vals))
        A_stability2 = F-G@staticKappa2
        eigenvalues = np.linalg.eigvals(A_stability2)
        max_eig_mags2.append(np.max(np.abs(eigenvalues)))
    max_eig_mags2_total.append(max_eig_mags2)
test = Counter(bestKappaIndex2).most_common(3)
print("The most valid values for K are:", staticKappa_vals[87, :])
# Fx*xk + Fu*uk-1 + G1*uk
# Fx*xk + Fu*(-Kappa*xk-1-Kappa2*xk-2)+ G1*(-Kappa*xk)
# xke = [xk,xk-1,xk-2], uk = uk = -Kappa*xk
# [Fx, -Kappa*Fu, -Kappa2*Fu]*xke + [G1;I]*uk
""" Plotting """
# For stability the maximum eigenvalues needs to be smaller then 1 of the A_stability matrix
# According to slides lecture 2 slide 18, using Ku is 0. Stability is valid for eigenvalues of closed loop A smaller then 1
plt.figure(figsize=(8, 4))
plt.plot(h_vals, max_eig_mags, label='Max |Eigenvalue|')
plt.axhline(1.0, color='red', linestyle='--', label='Stability Limit')
plt.xlabel("Sampling Time h")
plt.ylabel("Max Magnitude of Eigenvalues")
plt.title("Stability of Sampled-Data Closed-Loop System Exercise 1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.scatter(valid_h, valid_tau, alpha=0.9)
plt.xlabel("Sampling Time h")
plt.ylabel("Delay Time Tau")
plt.title("Stability of Sampled-Data Closed-Loop System Exercise 2")
plt.legend()
plt.show()

# Convert to NumPy array for plotting
Z = np.array(max_eig_mags2_total)

# Create meshgrid for contour plot
H, T = np.meshgrid(tau_vals, h_vals)  # note the order: row = h, column = tau

# Plot contour where max eigenvalue magnitude = 1
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, H, Z, levels=50, cmap='viridis')
plt.colorbar(contour, label='Max |eig(F)|')

# Add stability boundary (e.g., contour where max eig = 1)
cs = plt.contour(T, H, Z, levels=[1.0], colors='red', linewidths=2)
plt.clabel(cs, inline=True, fmt='Stability boundary', fontsize=10)

# Labels
plt.xlabel('τ (Delay)')
plt.ylabel('h (Sampling Time)')
plt.title('Max Eigenvalue Magnitude of F(h, τ)')
plt.grid(True)
plt.tight_layout()
plt.show()


