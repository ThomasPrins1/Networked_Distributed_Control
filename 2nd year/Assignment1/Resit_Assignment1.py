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
tolerance = 1e-1
eta = 1e-4
a = 5 # first digit student number
b = 8 # third digit student number
c = 1 # last digid digit student number
A = np.array([[(0.5-c),0],[(0.2+a-b),-1]])
B = np.array([[1],[0]])
print("A=",A)
print("B=",B)
# iterations
N = 100
kappaSamples = 5*N+1


# eigenvalues K:
k1 = symbols('k1')
k2 = symbols('k2')
s = symbols('s')
Kappa = np.array([[k1], [k2]])
A_sizeT = np.shape(A)
print(Kappa)

# fill in u = -Kappa*state_space:
# This gives state_space' = (A-B*Kappa)* state_space
# With poles at -2+-j, meaning eigenvalues of A-B*Kappa must be equal to (s-(-2-j))(s-(-2+j))
# This in turn is equal to s^2+4s+5, now the result of 
temp = s*np.eye(A_sizeT[0])-(A-(B@Kappa.T))
print(temp)
# (k1+s+0.5)(s+1) - 2.8*k2 -> k1*s + k1 + s^2 + s + 0.5*s + 0.5 - 2.8*k2 -> s^2 + s(k1+1.5) + (k1+0.5-2.8*k2)
# k1 = 2.5
# 3-2.8*k2 = 5 -> k2 = (5-3)/-2.8 -> k2 = -2/2.8
# this results in: (k1+s)*(s+1) - 2.8*(k2+0.5) = k1*s + s^2 + s + k1 - 2,8*k2 - 0.5*2.8
# equal to s^2 + s(k1+1) + (k1-2.8*k2-1.4)
# this gives k1 = 3
# and 3-2.8*k2-1.4 = 5 gives k2 = -3.4/2.8
k1 = 2.5
k2 = -2/2.8
Kappa = np.array([[k1], [k2]]).T


""" Functions """
def calculate_FG(A,B,h):
    # F values:
    n = A.shape[0]
    # m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    # write as x_k+1 = x_k*(F_h-G_h*Kappa), where:
    ## F_h = e^A*h and G_h = integral(e^A*h)*B
    if singular == n:
        F = expm(A*h)
        print(F)
        G = ((F-np.eye(n))@(np.linalg.inv(A)))@B
        print(G)
        #G = np.linalg.solve(A, expm(A * h) - expm(A * (h - tau))) @ B
        return F,G
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

def calculate_FG_delay_extended(A,B,h,tau):
    # F values:
    n = A.shape[0]
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    # write as x_k+1 = x_k*(F_h-G_h*Kappa)
    if singular == n:
        Fx = expm(A*h)
        temp = expm(A*(h-tau))
        if tau>(h/2) and tau<h:
            Fu = np.zeros((n,m))
            G1 = ((Fx-np.eye(n))@(np.linalg.inv(A)))@B
        else:
            Fu = ((Fx-temp)@(np.linalg.inv(A)))@B
            G1 = ((temp-np.eye(n))@(np.linalg.inv(A)))@B
        F = np.block([[Fx, Fu], [np.zeros((m,n)), np.zeros((m,m))]])
        G = np.vstack((G1, np.eye(m)))
        return F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def convex_solve(A,B,Kappa,tau_vals,h):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((2*(n+m),2*(n+m)), symmetric = True)
    stable_points = []
    constraints = [P >> 1e-4 * np.zeros_like(P)]
    for tau in tau_vals:
        if tau<h/2:
            F,G = calculate_FG_delay_extended(A,B,h,tau)
            Acl = F-G@Kappa
            A_aug = np.block([[Acl, np.zeros_like(Acl)], [np.eye(Acl.shape[0]), np.zeros_like(Acl)]])
            constraints += [A_aug.T @ P @ A_aug - P << -eta * np.eye(A_aug.shape[0])]
        elif tau>h/2 and tau<h:
            F,G = calculate_FG_delay_extended(A,B,h,tau)
            Acl = F-G@Kappa
            A_aug = np.block([[Acl, np.zeros_like(Acl)], [np.eye(Acl.shape[0]), np.zeros_like(Acl)]])
            constraints += [A_aug.T @ P @ A_aug - P << -eta * np.eye(A_aug.shape[0])]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-5)
    if prob.status == 'optimal':
        return True
    else:
        return False
    #return stable_points

def convex_solve_fixed(A,B,Kappa,h):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((2*(n+m),2*(n+m)), symmetric = True)
    stable_points = []
    constraints = [P >> 1e-4 * np.zeros_like(P)]

    F1,G1 = calculate_FG_delay_extended(A,B,h,h/3)
    F2,G2 = calculate_FG_delay_extended(A,B,h,3*h/4)
    F3,G3 = calculate_FG_delay_extended(A,B,h,h/4)
    Acl1 = F1-G1@Kappa
    Acl2 = F2-G2@Kappa
    Acl3 = F3-G3@Kappa
    Acl = Acl1@Acl2@Acl3
    A_aug = np.block([[Acl, np.zeros_like(Acl)], [np.eye(Acl.shape[0]), np.zeros_like(Acl)]])
    constraints += [A_aug.T @ P @ A_aug - P << -eta * np.eye(A_aug.shape[0])]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-5)
    if prob.status == 'optimal':
        return True
    else:
        return False
    
def checkElements(F,G,Kappa_vals):
    out = []
    i=0
    for i, K in enumerate(Kappa_vals.T):
        A_cl = F-G@(K.reshape(1,-1))
        eigenvalues = np.linalg.eigvals(A_cl)
        if (np.max(np.abs(eigenvalues)) < 1):
            out.append(i) # index of Kappa
    return out

""" Main Code """
""" Question 1 """

h_vals = np.linspace(0.01, 1, N)
max_eig_mags = []
best_h = []

for h in h_vals:
    F,G = calculate_FG(A,B,h)
    A_cl = F-G@Kappa
    eigenvalues = np.linalg.eigvals(A_cl)
    max_eig_mags.append(np.max(np.abs(eigenvalues)))
    if isclose(np.max(np.abs(eigenvalues)), 1, abs_tol=tolerance) == True:
        best_h.append(h)

print("Highest limit of sampling time given by h=",np.mean(best_h),"with tolerance equal to +-:",tolerance)
print("Lower limit will always approach 0")



""" Question 2 """
tau_vals = np.linspace(0, 1, N)
max_eig_mags2_total = []
bestKappaIndex = []
# with extended state an extended Kappa is needed aswell using the static controller:
staticKappa = np.hstack((Kappa, np.zeros((1,1))))

Kappa_u_vals = np.linspace(-10, 10, kappaSamples)
dynamicKappa_vals = np.stack([np.ones_like(Kappa_u_vals)*Kappa[0,0],np.ones_like(Kappa_u_vals)*Kappa[0,1], Kappa_u_vals])
for i,h in enumerate(h_vals):
    print("progress Q2:",(i/N)*100,"%")
    max_eig_mags2 = []
    for tau in tau_vals:
        if tau<h:
            F,G = calculate_FG_delay(A,B,h,tau)
            A_cl = F-G@staticKappa
            eigenvalues = np.linalg.eigvals(A_cl)
            temp = np.max(np.abs(eigenvalues))
            max_eig_mags2.append(temp)
        else:
            max_eig_mags2.append(np.nan)
    max_eig_mags2_total.append(max_eig_mags2)

# h= 0.6 is arbitrarily chosen while being stable at tau=0
# we use the dynamic Kappa for this with variable dynamic control.
h = 0.6
for tau in tau_vals:
    if tau<h:
        F,G = calculate_FG_delay(A,B,h,tau)
        bestKappaIndex.extend(checkElements(F,G,dynamicKappa_vals))

commonKappa = Counter(bestKappaIndex).most_common(10)
print(commonKappa)
print("The most valid values for K are:", dynamicKappa_vals[:,commonKappa[0][0]])

""" Question 3 """
feasible_list = []
feasible_list_fixed = []
h_vals = np.linspace(0, 1, N)
tau_vals = np.linspace(0, 1, N)
for i,h in enumerate(h_vals):
    print("progress Q3:",(i/N)*100,"%")
    feasibility = convex_solve(A,B,staticKappa,tau_vals,h)
    feasibility_fixed = convex_solve_fixed(A,B,staticKappa,h)
    feasible_list.append(feasibility)
    feasible_list_fixed.append(feasibility_fixed)


#Q1
min_index = max_eig_mags.index(min(max_eig_mags))
min_h = h_vals[min_index]
min_val = max_eig_mags[min_index]

plt.figure(figsize=(8, 4))
plt.plot(h_vals, max_eig_mags, label='Max |Eigenvalue|')
plt.axhline(1.0, color='red', linestyle='--', label='Stability Limit')

# Highlight the minimum point
plt.plot(min_h, min_val, 'o', color='green', label='Minimum Value')
plt.annotate(f'Min: ({min_h:.2f}, {min_val:.2f})',
             xy=(min_h, min_val), xytext=(min_h, min_val + 0.05),
             arrowprops=dict(arrowstyle='->', color='green'),
             color='green')

plt.xlabel("Sampling Time h")
plt.ylabel("Max Magnitude of Eigenvalues")
plt.title((f"Stability of Sampled-Data Closed-Loop System between: [0, {np.mean(best_h):.2f}]"))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q1_stabilityOfStaticController.png")
plt.show()

#Q2
# Convert to NumPy array for plotting
Z1 = np.array(max_eig_mags2_total)

# Create meshgrid for contour plot
H, T = np.meshgrid(h_vals,tau_vals)  # note the order: row = h, column = tau

# Plot contour where max eigenvalue magnitude = 1
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, H, Z1, levels=50, cmap='viridis')
plt.colorbar(contour, label='Max |eig(F)|')

# Add stability boundary (e.g., contour where max eig = 1)
cs = plt.contour(T, H, Z1, levels=[1.0], colors='red', linewidths=2)
plt.clabel(cs, inline=True, fmt='Stability boundary', fontsize=10)

# Labels
plt.ylabel('τ (Delay)')
plt.xlabel('h (Sampling Time)')
plt.title('Max Eigenvalue Magnitude of F(h, τ)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q2_stabilityOfNCS.png")
plt.show()

# Unpack the data
kappa_values, counts = zip(*commonKappa)

# Create bar chart
plt.figure(figsize=(6, 4))
plt.bar(dynamicKappa_vals[2,kappa_values], counts, color='skyblue')
plt.xlabel('dynamic Kappa value')
plt.ylabel('Frequency')
plt.title('Top 10 Most Common Kappa Index Values')
plt.grid(axis='y', linestyle='--', alpha=0.9)
plt.tight_layout()
plt.savefig("Q2_mostCommonKappa.png")
plt.show()


#Q3
# Convert booleans to array
feasible_arr = np.array(feasible_list)
feasible_arr_fixed = np.array(feasible_list_fixed)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# First plot
axes[0].scatter(h_vals[feasible_arr], np.ones(sum(feasible_arr)),
                color="green", marker="o", label="Feasible")
axes[0].scatter(h_vals[~feasible_arr], np.ones(sum(~feasible_arr)),
                color="red", marker="x", label="Infeasible")
axes[0].set_xlabel("h (Sampling Time)")
axes[0].set_title("Original Feasibility")
axes[0].grid(True)
axes[0].legend()
axes[0].set_yticks([])

# Second plot
axes[1].scatter(h_vals[feasible_arr_fixed], np.ones(sum(feasible_arr_fixed)),
                color="green", marker="o", label="Feasible")
axes[1].scatter(h_vals[~feasible_arr_fixed], np.ones(sum(~feasible_arr_fixed)),
                color="red", marker="x", label="Infeasible")
axes[1].set_xlabel("h (Sampling Time)")
axes[1].set_title("Fixed Feasibility")
axes[1].grid(True)
axes[1].legend()
axes[1].set_yticks([])

plt.suptitle("Feasible h Values (LMI Stability Test)")
plt.tight_layout()
plt.savefig("Q3_rangeOfStability_ZOH.png")
plt.show()
