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
# iterations
N = 20
K_Samples = 51
h_max = 0.5
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


""" Functions """
def FG_model_noDelay(A,B,h):
    # This function uses the A and B matrix to form the F and G matrix using the sampling time h
    n = A.shape[0]
    singular = np.linalg.matrix_rank(A)
    if singular == n:
        F = expm(A*h)
        G = ((F-np.eye(n))@(np.linalg.inv(A)))@B
        return F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan
    

def FG_model_delay(A,B,h,tau):
    # This function uses the A and B matrices to form the extended F and G matrices. It uses the sampling time h and the delay tau.
    # Take care this is an extended system, which means the sizes of the F and G matrices are not equal to the noDelay model!
    # the delay goes back to k-2
    n = A.shape[0]
    m = B.shape[1]
    singular = np.linalg.matrix_rank(A)
    # write as x_k+1 = x_k*(F_h-G_h*Kappa)
    if singular == n:
        Fx = expm(A*h) # 2x2
        temp = expm(A*(h-tau)) # 2x2
        if tau<=h:
            Fu = ((Fx-temp)@(np.linalg.inv(A)))@B # 2x1
        elif tau>h:
            Fu = np.zeros((n,m)) # 2x1
        G1 = ((temp-np.eye(n))@(np.linalg.inv(A)))@B # 2x1
        F = np.block([[Fx, Fu], [np.zeros((m,n+m))]]) # 3x3
        G = np.block([[G1],[np.eye(m)]])
        return F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def convex_solve(A,B,K,tau_vals,h):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((2*(n+m),2*(n+m)), symmetric = True)
    stable_points = []
    constraints = [P >> 1e-4 * np.zeros_like(P)]
    for tau in tau_vals:
        if tau < 2*h:
            F,G = FG_model_delay(A,B,h,tau)
            Acl = F-G@K
            A_aug = np.block([[Acl, np.zeros_like(Acl)], [np.eye(Acl.shape[0]), np.zeros_like(Acl)]])
            constraints += [A_aug.T @ P @ A_aug - P << -eta * np.eye(A_aug.shape[0])]
    prob = cp.Problem(cp.Minimize(0), constraints)
    prob.solve(solver=cp.SCS, max_iters=5000, eps=1e-5)
    if prob.status == 'optimal':
        return True
    else:
        return False
    #return stable_points

def convex_solve_fixed(A,B,K,tau1_vals,tau2_vals,h):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable((2*(n+m),2*(n+m)), symmetric = True)
    stable_points = []
    constraints = [P >> 1e-4 * np.zeros_like(P)]
    # first check (tau1,tau1,tau2)
    for tau1 in tau1_vals:
        if tau1 < 2*h:
            F1,G1 = FG_model_delay(A,B,h,tau1)
            Acl1 = F1-G1@K
            for tau2 in tau2_vals:
                F2,G2 = FG_model_delay(A,B,h,tau2)
                Acl2 = F2-G2@K
                Acomb = Acl1@Acl1@Acl2
                A_aug1 = np.block([[Acomb, np.zeros_like(Acomb)], [np.eye(Acomb.shape[0]), np.zeros_like(Acomb)]])
                constraints += [A_aug1.T @ P @ A_aug1 - P << -eta * np.eye(A_aug1.shape[0])]
                # second check (tau2)
                
    # this should potentially give less constraints since 2nd check is limited over h<tau<2h
    for tau2 in tau2_vals:
        if tau2>h and tau2<2*h:
                A_aug2 = np.block([[Acl2, np.zeros_like(Acl2)], [np.eye(Acl2.shape[0]), np.zeros_like(Acl2)]])
                constraints += [A_aug2.T @ P @ A_aug2 - P << -eta * np.eye(A_aug2.shape[0])]
    
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
        Kex = np.hstack((K.reshape(1, -1),np.zeros((1,1))))
        A_cl = F-G@(Kex.reshape(1,-1))
        eigenvalues = np.linalg.eigvals(A_cl)
        if (np.max(np.abs(eigenvalues)) < 1):
            out.append(i) # index of Kappa
    return out

""" Main Code """

""" Question 1 """

h_vals = np.linspace(0, h_max, N)
max_eig_noDelay = []
best_h = []

for h in h_vals:
    F,G = FG_model_noDelay(A,B,h)
    A_cl = F-G@K
    eigenvalues = np.linalg.eigvals(A_cl)
    max_eig_noDelay.append(np.max(np.abs(eigenvalues)))
    if isclose(np.max(np.abs(eigenvalues)), 1, abs_tol=tolerance) == True:
        best_h.append(h)

print("Highest limit of sampling time given by h=",np.mean(best_h),"with tolerance equal to +-:",tolerance)
print("Lower limit will always approach 0")

## Plotting

min_index = max_eig_noDelay.index(min(max_eig_noDelay))
min_h = h_vals[min_index]
min_val = max_eig_noDelay[min_index]

plt.figure(figsize=(8, 4))
plt.plot(h_vals, max_eig_noDelay, label='Max |Eigenvalue|')
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



""" Question 2 """

tau_vals = np.linspace(0, 2*h_max, N)
max_eig_delay_total = []
best_K_index = []
# with extended state an extended Kappa is needed aswell using the static controller:
Ke = np.hstack((K, np.zeros((1,1)))) # 1x3

for i,h in enumerate(h_vals):
    print("progress Q2:",(i/N)*100,"%")
    max_eig_delay = []
    for tau in tau_vals:
        if tau<2*h: # case where tau < 2h
            F,G = FG_model_delay(A,B,h,tau)
            A_cl = F-G@Ke
            eigenvalues = np.linalg.eigvals(A_cl)
            temp = np.max(np.abs(eigenvalues))
            max_eig_delay.append(temp)
        else: # case where exceeds boundary of 2h
            max_eig_delay.append(np.nan)
    max_eig_delay_total.append(max_eig_delay)

# h= 0.3 is arbitrarily chosen while being stable at tau=0
# we use the dynamic Kappa for this with variable dynamic control.
k1_range = np.linspace(0, 30, K_Samples)
k2_range = np.linspace(0, 30, K_Samples)

K1, K2 = np.meshgrid(k1_range, k2_range)

dynamic_K_vals = np.vstack([
    K1.ravel(),
    K2.ravel()
])
best_K_Index = []
h = 0.3
for i,tau in enumerate(tau_vals):
    print("progress Q2, K optimisation:",(i/N)*100,"%")
    if tau<2*h:
        F,G = FG_model_delay(A,B,h,tau)
        best_K_Index.extend(checkElements(F,G,dynamic_K_vals))

commonKappa = Counter(best_K_Index)

## Plotting
Z1 = np.where(np.array((max_eig_delay_total))>=1,0,1) # Convert to NumPy array for plotting & make 1 for stable, 0 for unstable

# Create meshgrid for contour plot
H, T = np.meshgrid(h_vals,tau_vals)  # note the order: row = h, column = tau

# Plot contour where max eigenvalue magnitude = 1
plt.figure(figsize=(8, 6))
contour = plt.contourf(T, H, Z1, levels=2, cmap='viridis')
plt.colorbar(contour, label='Max |eig(F)|')

# Add stability boundary (e.g., contour where max eig = 1)
#cs = plt.contour(T, H, Z1, levels=[1.0], colors='red', linewidths=2)
#plt.clabel(cs, inline=True, fmt='Stability boundary', fontsize=10)

# Labels
plt.ylabel('τ (Delay)')
plt.xlabel('h (Sampling Time)')
plt.title('Max Eigenvalue Magnitude of F(h, τ)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Q2_stabilityOfNCS.png")
plt.show()

# Create empty heatmap matrix
heatmap = np.zeros((len(k2_range), len(k1_range)))

# Fill heatmap with occurrence counts
for idx, count in commonKappa.items():

    k1 = dynamic_K_vals[0, idx]
    k2 = dynamic_K_vals[1, idx]

    # Find indices in grid
    i = np.where(k2_range == k2)[0][0]
    j = np.where(k1_range == k1)[0][0]

    heatmap[i, j] = count

# Find highest frequency
max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)

best_k2 = k2_range[max_pos[0]]
best_k1 = k1_range[max_pos[1]]
best_count = heatmap[max_pos]

# Plot heatmap
plt.figure(figsize=(8,6))

im = plt.imshow(
    heatmap,
    origin='lower',
    extent=[
        k1_range.min(),
        k1_range.max(),
        k2_range.min(),
        k2_range.max()
    ],
    aspect='auto',
    cmap='hot'
)

# Colorbar
cbar = plt.colorbar(im)
cbar.set_label('Frequency')

# Highlight maximum
plt.scatter(
    best_k1,
    best_k2,
    s=250,
    facecolors='none',
    edgecolors='cyan',
    linewidths=3,
    label=f'Max Frequency = {int(best_count)}'
)

# Labels
plt.xlabel('k1')
plt.ylabel('k2')
plt.title('K1-K2 Frequency Heatmap')

plt.legend()
plt.grid(False)
plt.tight_layout()

plt.savefig("Q2_mostCommonKappa_heatmap.png")
plt.show()

print("Highest frequency at:")
print(f"k1 = {best_k1}")
print(f"k2 = {best_k2}")
print(f"count = {int(best_count)}")





""" Question 3 """
feasible_list = []
feasible_list_fixed = []
h_vals = np.linspace(0, h_max, N)
tau_vals = np.linspace(0, 2*h_max, N)
tau_vals1 = np.linspace(0, h_max, int(N/2))
tau_vals2 = np.linspace(h_max, 2*h_max, int(N/2))
for i,h in enumerate(h_vals):
    print("progress Q3:",(i/N)*100,"%")
    feasibility = convex_solve(A,B,Ke,tau_vals,h)
    feasibility_fixed = convex_solve_fixed(A,B,Ke,tau_vals1,tau_vals2,h)
    feasible_list.append(feasibility)
    feasible_list_fixed.append(feasibility_fixed)
    



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
