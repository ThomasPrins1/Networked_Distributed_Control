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
N = 1000
K_Samples = 5*N+1
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
        Fu = ((Fx-temp)@(np.linalg.inv(A)))@B # 2x1
        G1 = ((temp-np.eye(n))@(np.linalg.inv(A)))@B # 2x1
        F = np.block([[Fx, Fu, np.zeros((n,m))], [np.zeros((m,n)), np.zeros((m,m)), np.zeros((m,m))], [np.zeros((m,n)), np.eye(m),np.zeros((m,m))]]) # 4x4
        G = np.block([[G1],[np.eye(m)],[np.zeros((m,m))]])
        ##np.vstack((G1, np.eye(m)), np.zeros((m,m))) # 4x1
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
Ke = np.hstack((K, np.eye(1), np.zeros((1,1)))) # 1x4 since [K,1,0], not sure about 1 though?

#Ke_vals = np.linspace(-10, 10, K_Samples) #?
#dynamicKappa_vals = np.stack([np.ones_like(Ke_vals)*K[0,0],np.ones_like(Ke_vals)*K[0,1], Ke_vals]) #?
for i,h in enumerate(h_vals):
    print("progress Q2:",(i/N)*100,"%")
    max_eig_delay = []
    for tau in tau_vals:
        if tau<h: # case where tau < h
            F,G = FG_model_delay(A,B,h,tau)
            A_cl = F-G@Ke
            eigenvalues = np.linalg.eigvals(A_cl)
            temp = np.max(np.abs(eigenvalues))
            max_eig_delay.append(temp)
        else: # case where tau > h
            F,_ = FG_model_delay(A,B,h,tau)
            eigenvalues = np.linalg.eigvals(F)
            temp = np.max(np.abs(eigenvalues))
            max_eig_delay.append(temp)
    max_eig_delay_total.append(max_eig_delay)

# h= 0.6 is arbitrarily chosen while being stable at tau=0
# we use the dynamic Kappa for this with variable dynamic control.
#h = 0.6
#for tau in tau_vals:
#    if tau<h:
#        F,G = calculate_FG_delay(A,B,h,tau)
#        bestKappaIndex.extend(checkElements(F,G,dynamicKappa_vals))

#commonKappa = Counter(bestKappaIndex).most_common(10)
#print(commonKappa)
#print("The most valid values for K are:", dynamicKappa_vals[:,commonKappa[0][0]])

## Plotting

Z1 = np.array(max_eig_delay_total) # Convert to NumPy array for plotting

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

# temp exit()
exit()

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





#Q2 extra
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
