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
import scipy.io

np.random.seed(19680806)


""" Variables """
tolerance = 1e-2
eta = 1e-4
a = 5 # first digit student number
b = 8 # third digit student number
c = 1 # last digid digit student number
max_iter = 50
agents = 4

# load in the agents.mat dataset:
agent_dynamic_data = scipy.io.loadmat('agents.mat')
A1 = np.array((agent_dynamic_data['A1']))
A2 = np.array((agent_dynamic_data['A2']))
A3 = np.array((agent_dynamic_data['A3']))
A4 = np.array((agent_dynamic_data['A4']))
B1 = np.array((agent_dynamic_data['B1']))
B2 = np.array((agent_dynamic_data['B2']))
B3 = np.array((agent_dynamic_data['B3']))
B4 = np.array((agent_dynamic_data['B4']))
x0_1 = (agent_dynamic_data['x01'])[0][0]
x0_2 = (agent_dynamic_data['x02'])[0][0]
x0_3 = (agent_dynamic_data['x03'])[0][0]
x0_4 = (agent_dynamic_data['x04'])[0][0]
N_p = (agent_dynamic_data['Tfinal'])[0][0]
u_max = (agent_dynamic_data['umax'])[0][0]

x = np.zeros((agents,N_p))
x0 = np.array((x0_1, x0_2, x0_3, x0_4))
""" Functions """
def solveSystem(A,B,N_p,u_max,x0,lapda):
    # this solves the system using its cost function
    # important is that this is just for one of the nodes not for both!
    # returns value for x
    n = A.shape[0]
    m = B.shape[1]
    x = cp.Variable((n, N_p+1))
    u = cp.Variable((m, N_p))
    
    constraints = [x[:, 0] == x0]
    J = 0
    for t in range(N_p):
        constraints.append(A@x[:,t]+B@u[:,t] == x[:,t+1])
        constraints.append(cp.abs(u[:,t]) <= u_max)

        J += (cp.quad_form(x[:, t], np.eye(n)) + cp.quad_form(u[:, t], np.eye(m))) 
    J += lapda@x[:,-1]
    prob = cp.Problem(cp.Minimize(J), constraints)
    prob.solve()
    return x[:,-1].value,u.value

def updateLagrange(lapda,subgrad,alpha):
    lapda = lapda + alpha*subgrad
    return lapda
"""""
def convexSolve(A,B,x0,k,N_p,u_max):
    n = x0.shape[0] # number of agents
    J = cp.Variable()
    lapda = cp.Variable((n,1), nonnegative = True)
    x = cp.Variable((n,N_p))
    x[:,0] = x0
    u = cp.Variable((n,1))
    g = cp.Variable((n,1))
    constraints = []
    lapda = np.zeros((N_p,1))
    x_f = 5 # not sure what value this is yet! Think this gets updated?
    # need lagrangian! x(end) = x_final gives x(T_final) - x_f = 0
    # Lagrangian = sum_i(sum_t(x^2+u^2)) + sum_i(lapda(i)*(x(T_final) - x_f))
    # then J(x) = max(Lagrangian) for lapda>=0
    # dual = min(lagrangian) = min(sum_i(sum_t(x^2+u^2) + lapda(i)*(x(T_final) - x_f)))
    # then the optimization problem becomes:
    # max(dual) for lapda >= 0
    # but assignment asks for a dual decomposition lagrangian, which means:
    # min f_i + f_j, this gives:
    # Lagrangian = sum_i(sum_t(x^2+u^2)) + sum_i(lapda(i)*(x(T_final) - x_f))
    for i in range(n):
        constraints.append(cp.abs(u[i]) <= u_max)
    
    for t in range(1,N_p): # start at 1 since x0 is known already?
        for i in range(n):
            #dual_function = (x[i].T*x[i] + u[i].T*u[i])
            x_star_i,u = solveSystem(A[j],B[j],N_p,u_max,x0,lapda)
            lapda[t+1] = updateLagrange(lapda[t],0.1,x_star_i,x_star_j)
            for j in range(n):
                x_star_j,u = solveSystem(A[j],B[j],N_p,u_max,x0,lapda)
                sub_j[t] = x_star_j-x_f
                # dual function states x_i=x_j
            
            #for j in range(n):
            #    x_star = updateStep()
            #subgradient[i] = lapda[i]*(x(i,-1)-x_f)
            #x(i,t),sub_bounds = dynamics(x(i,t),x(i,t-1),g,) #static alpha for now
            constraints.append(A(i)@x(i,t)+B(i)@u(i,t) == 0) # primal constraints
            #constraints.append(sub_bounds) # constraints for subgradient
        
        summed_dual_function += dual_function #+ lapda(i)(x(i,-1)-x_f)
    J = cp.max(summed_dual_function)
    prob = cp.Problem(cp.Minimize(J), constraints)
    prob.solve(solver=cp.SCS)
    return u.value
"""""
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
A_list = [A1,A2,A3,A4]
B_list = [B1,B2,B3,B4]
n = A1.shape[0]
m = B1.shape[1]
lapda = np.zeros((m,n))
subgrad = np.zeros((m,n))
x_star_i = np.zeros((m,n))
x_star_j = np.zeros((m,n))
subgrad_norms = []
target_errors = []
x_f = 10
for time,k in enumerate(range(max_iter)):
    print(time)
    for i in range(agents):
        x_star_i,u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda)
        subgrad = x_star_i - x_f
        lapda = updateLagrange(lapda,subgrad,alpha=0.1)
        target_errors.append(np.linalg.norm(x_star_i - x_f))

        #for j in range(agents):
        #    if i!=j:
        #        x_star_i,u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda)
        #        x_star_j,u_test = solveSystem(A_list[j],B_list[j],N_p,u_max,x0[j],-lapda)
        #        subgrad = x_star_i - x_star_j
        #        lapda = updateLagrange(lapda,subgrad,alpha=0.1)
        #        subgrad_norms.append(np.linalg.norm(subgrad))

""" Plotting """
plt.figure(figsize=(8, 5))
plt.plot(target_errors)
plt.title("Convergence to Target State x_f")
plt.ylabel("‖x_i(T) - x_f‖")
plt.xlabel("Iteration")
plt.title("Consensus Convergence via Dual Decomposition")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()