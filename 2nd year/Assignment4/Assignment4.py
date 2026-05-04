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
max_iter = 200
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
x0_1 = (agent_dynamic_data['x01'])
x0_2 = (agent_dynamic_data['x02'])
x0_3 = (agent_dynamic_data['x03'])
x0_4 = (agent_dynamic_data['x04'])
N_p = (agent_dynamic_data['Tfinal'])[0][0]
u_max = (agent_dynamic_data['umax'])[0][0]

x = np.zeros((agents,N_p))
x0 = np.vstack((x0_1.T, x0_2.T, x0_3.T, x0_4.T))
print(x0)
""" Functions """
def solveCentralisedSystem(A_list, B_list, x0_list, N_p, u_max):
    agents = len(A_list)
    n = A_list[0].shape[0]
    m = B_list[0].shape[1]

    # Variables for all agents
    X = [cp.Variable((n, N_p + 1)) for _ in range(agents)]
    U = [cp.Variable((m, N_p)) for _ in range(agents)]
    cost = 0
    constraints = []

    for i in range(agents):
        A = A_list[i]
        B = B_list[i]
        x0 = x0_list[i]

        constraints.append(X[i][:, 0] == x0)
        for t in range(N_p):
            constraints += [
                X[i][:, t+1] == A @ X[i][:, t] + B @ U[i][:, t],
                cp.abs(U[i][:, t]) <= u_max
            ]
            cost += cp.quad_form(X[i][:, t], np.eye(n)) + cp.quad_form(U[i][:, t], np.eye(m))

        cost += cp.quad_form(X[i][:, N_p], np.eye(n))

    # Terminal consensus constraint
    for i in range(1, agents):
        constraints.append(X[i][:, N_p] == X[0][:, N_p])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    X_values = [Xi.value for Xi in X]
    U_values = [Ui.value for Ui in U]
    x_f = X[0][:, N_p].value

    return X_values, U_values, x_f

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

def exactLineSearch(x,subgrad):
    #alpha = sum(sum(np.gradient((x-alpha*subgrad)^2 + u^2))) == 0
    #alpha = sum(sum(np.gradient(x^2+(alpha*subgrad)^2)-2*x*alpha*subgrad + u^2)) == 0
    #alpha = sum(sum(0 + 2*alpha*subgrad^2 -2*subgrad + 0)) == 0
    #alpha = 2*alpha*subgrad^2-2*x*subgrad == 0
    #alpha = alpha(subgrad^2) == x*subgrad
    alpha = sum((x.T@subgrad)/(subgrad.T@subgrad))
    #alpha = sum(x/subgrad)
    return alpha

def updateLagrange(lapda,subgrad,alpha):
    lapda = lapda + alpha*subgrad
    return lapda

def nesterovAccelaration(lapda,eta,subgrad,alpha, mu = 0.5):
    oldlapda = lapda
    lapda = eta + alpha*subgrad
    eta = lapda + mu*(lapda-oldlapda)
    return lapda,eta

""" Main Code """
""" Question 1 """

# initialize
A_list = [A1,A2,A3,A4]
B_list = [B1,B2,B3,B4]
n = A1.shape[0]
m = B1.shape[1]
lapda = np.zeros((agents,n))
subgrad = np.zeros((agents,n))
x_star_i = np.zeros((agents,n))
x_f = np.zeros((m,n))
#x_f = np.mean(x0, axis=0)
print(np.mean(x0, axis=0))
x_f_tiled = np.tile((np.mean(x0, axis=0)), (agents, 1))
print(x_f_tiled)
initial_error = np.linalg.norm(x0 - x_f_tiled) 
print(initial_error)
target_errors_CSS = [initial_error] # constant step size

# plotting things:
x_traj_CSS = np.zeros((max_iter, agents, n))
x_traj_ELC = np.zeros((max_iter, agents, n))
x_traj_ANM = np.zeros((max_iter, agents, n))
x_traj_CC  = np.zeros((max_iter, agents, n))
x_traj_ADMM  = np.zeros((max_iter, agents, n))
# centralized comparison:
X_central, U_central, x_f_central = solveCentralisedSystem(A_list, B_list, x0, N_p, u_max)

# static alpha decentralized:
for time,k in enumerate(range(max_iter)):
    print(time)
    for i in range(agents):
        x_star_i[i],u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda[i])
    x_traj_CSS[time] = x_star_i
    x_f = np.mean(x_star_i, axis=0)
    subgrad = x_star_i - x_f # (agents,2) in size
    for i in range(agents):
        lapda[i] = updateLagrange(lapda[i],subgrad[i],alpha=0.1)
    target_errors_CSS.append(np.linalg.norm(subgrad))

# exact line search:
# re-initialize:
lapda = np.zeros((agents,n))
subgrad = np.zeros((agents,n))
x_star_i = np.zeros((agents,n))
x_f = np.zeros((m,n))
target_errors_ELC = [initial_error] # exact line search


for time,k in enumerate(range(max_iter)):
    print(time)
    for i in range(agents):
        x_star_i[i],u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda[i])
    x_traj_ELC[time] = x_star_i
    x_f = np.mean(x_star_i, axis=0)
    subgrad = x_star_i - x_f # (agents,2) in size
    alpha = exactLineSearch(x_star_i,subgrad)
    #alpha = argmin(f(x-alpha*subgrad)), where f is the cost function
    for i in range(agents):
        lapda[i] = updateLagrange(lapda[i],subgrad[i],alpha)
    target_errors_ELC.append(np.linalg.norm(subgrad))

## accelerated nesterov method:
# re-initialize:
lapda = np.zeros((agents,n))
subgrad = np.zeros((agents,n))
x_star_i = np.zeros((agents,n))
x_f = np.zeros((m,n))
eta = np.zeros((agents,n))
target_errors_ANM = [initial_error] # nesterov and exact line search

for time,k in enumerate(range(max_iter)):
    print(time)
    for i in range(agents):
        x_star_i[i],u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda[i])
    x_traj_ANM[time] = x_star_i
    x_f = np.mean(x_star_i, axis=0)
    subgrad = x_star_i - x_f # (agents,2) in size
    alpha = exactLineSearch(x_star_i,subgrad)
    for i in range(agents):
        lapda[i],eta[i] = nesterovAccelaration(lapda[i],eta[i],subgrad[i],alpha)
    target_errors_ANM.append(np.linalg.norm(subgrad))

# combined consensus:
W = np.array(([0.75,0.25,0,0],
              [0.25,0.5,0.25,0],
              [0,0.25,0.5,0.25],
              [0,0,0.25,0.75]))
cons_iter_iter = 4
# re-initialize:

x_star_i = np.zeros((agents,n))
target_errors_CC = [] # combined consensus
cons_iter_max = np.linspace(1,37,cons_iter_iter)
summed_norm = np.zeros((cons_iter_iter,1))
for l,cons_iter in enumerate(cons_iter_max):
    target_errors_CC_tryout = [initial_error]
    lapda = np.zeros((agents,n))
    subgrad = np.zeros((agents,n))
    x_f = np.zeros((agents,int(cons_iter)+1,n))
    for time,k in enumerate(range(max_iter)):
        print(time)
        for i in range(agents):
            x_star_i[i],u_test = solveSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda[i])
        x_traj_CC[time] = x_star_i
        x_f[:,0,:] = x_star_i
        for r in range(int(cons_iter)):
            for i in range(agents):
                consensus_sum = np.zeros(n)
                for j in range(agents):
                    consensus_sum += W[i, j] * (x_f[j, r, :] - x_f[i, r, :])
                x_f[i, r + 1, :] = x_f[i, r, :] + consensus_sum
        subgrad = x_star_i - x_f[:, -1, :] # substract final consensus value
        alpha = exactLineSearch(x_star_i,subgrad)
        for i in range(agents):
            lapda[i] = updateLagrange(lapda[i],subgrad[i],alpha)
        target_errors_CC_tryout.append(np.linalg.norm(subgrad))
        summed_norm[l] += np.linalg.norm(subgrad)
    target_errors_CC.append(target_errors_CC_tryout)
best_index = np.argmin(summed_norm)
target_errors_CC_best = target_errors_CC[best_index]

#TEST
## ADMM

def solveADMMSystem(A,B,N_p,u_max,x0,lapda,z,rho):
    # basically z = xf
    n = A.shape[0]
    m = B.shape[1]
    x = cp.Variable((n, N_p+1))
    u = cp.Variable((m, N_p))
    lapda = cp.Constant(lapda.reshape(-1, 1))
    z = cp.Constant(z.reshape(-1, 1))
    constraints = [x[:, 0] == x0]
    J = 0
    for t in range(N_p):
        constraints.append(A@x[:,t]+B@u[:,t] == x[:,t+1])
        constraints.append(cp.abs(u[:,t]) <= u_max)

        J += (cp.quad_form(x[:, t], np.eye(n)) + cp.quad_form(u[:, t], np.eye(m))) 
    J += cp.matmul(lapda.T,(x[:,-1:]-z)) + (rho/2)*cp.sum_squares(x[:,-1:]-z)
    prob = cp.Problem(cp.Minimize(J), constraints)
    prob.solve(solver=cp.OSQP)
    return x[:,-1].value,u.value

def averageZ(x,lapda,rho):
    z = np.mean((x+(1/rho)*lapda),axis=0)
    return z

max_iter_rho = 5
x_star_i = np.zeros((agents,n))
target_errors_ADMM = [] # ADMM
rho_iter = np.linspace(0.1,10,max_iter_rho)
summed_norm = np.zeros((max_iter_rho,1))
print("ADMM")
for l,rho in enumerate(rho_iter):
    target_errors_ADMM_tryout = [initial_error]
    lapda = np.zeros((agents, n))
    subgrad = np.zeros((agents, n))
    z = np.zeros((n, 1))
    for time,k in enumerate(range(max_iter)):
        print(time)
        for i in range(agents):
            x_star_i[i,:],u_test = solveADMMSystem(A_list[i],B_list[i],N_p,u_max,x0[i],lapda[i],z,rho)
        x_traj_ADMM[time] = x_star_i
        z = averageZ(x_star_i,lapda,rho)
        for i in range(agents):
            subgrad[i,:] = x_star_i[i] - z
            lapda[i,:] = updateLagrange(lapda[i],subgrad[i],rho)
        target_errors_ADMM_tryout.append(np.linalg.norm(subgrad))
        summed_norm[l] += np.linalg.norm(subgrad)
    target_errors_ADMM.append(target_errors_ADMM_tryout)
best_index = np.argmin(summed_norm)
target_errors_ADMM_best = target_errors_ADMM[best_index]

""" Plotting """
# Compute final error (‖x_i(T) - x_f‖) for all agents
final_errors_centralized = [
    np.linalg.norm(X_central[i][:, -1] - x_f_central)
    for i in range(len(X_central))
]

# Average or max error as a benchmark
centralized_error = np.max(final_errors_centralized)

plt.figure(figsize=(8, 5))

plt.plot(target_errors_CSS, label="CSS")
plt.plot(target_errors_ELC, label="ELC")
plt.plot(target_errors_ANM, label="ANM")
plt.plot(target_errors_CC_best, label="CC")
plt.plot(target_errors_ADMM_best, label="ADMM")
plt.axhline(y=centralized_error, color='k', linestyle='--', label='Centralized')

plt.xlabel("Iteration")
plt.ylabel("‖x_i(T) - x_f‖")
plt.yscale('log')
plt.title("Convergence to Target State x_f")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# state plot
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
labels = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4']

method_data = {
    'CSS': x_traj_CSS,
    'ELC': x_traj_ELC,
    'ANM': x_traj_ANM,
    'CC':  x_traj_CC,
    'ADMM':  x_traj_ADMM,
}

for method, data in method_data.items():
    fig, axs = plt.subplots(n, 1, figsize=(10, 6), sharex=True)
    
    for dim in range(n):
        for agent in range(agents):
            axs[dim].plot(
                range(max_iter),
                data[:, agent, dim],
                label=labels[agent],
                color=colors[agent]
            )
        axs[dim].axhline(y=x_f_central[dim], color='k', linestyle='--', linewidth=1, label='Centralized')
        axs[dim].set_ylabel(f'$x_{{{dim+1}}}$')
        axs[dim].grid(True)

    axs[-1].set_xlabel("Iteration")
    handles, labls = axs[0].get_legend_handles_labels()
    fig.legend(handles, labls, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Final State Evolution – {method}")
    plt.tight_layout()
    plt.show()

## EXTRA FOR ADMM TRYOUT FOR RHO:
# Plotting convergence for each rho
plt.figure(figsize=(10, 6))
for i, rho in enumerate(rho_iter):
    plt.plot(target_errors_ADMM[i], label=f"ρ = {rho:.2f}", linewidth=2)

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("‖Subgradient‖", fontsize=12)
plt.yscale('log')
plt.title("ADMM Convergence vs. ρ", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting convergence for each rho
plt.figure(figsize=(10, 6))
for i, cons in enumerate(cons_iter_max):
    plt.plot(target_errors_CC[i], label=f"ρ = {cons:.2f}", linewidth=2)

plt.xlabel("Iteration", fontsize=12)
plt.ylabel("‖Subgradient‖", fontsize=12)
plt.yscale('log')
plt.title("Consensus Convergence vs. ρ", fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()