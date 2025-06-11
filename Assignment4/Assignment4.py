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
A = np.array([[0,(0.5-c)],[(0.2+a-b),-1]])
B = np.array([[1],[0]])


# load in the agents.mat dataset:
print()
agent_dynamic_data = scipy.io.loadmat('agents.mat')
A1 = agent_dynamic_data['A1']
A2 = agent_dynamic_data['A2']
A3 = agent_dynamic_data['A3']
A4 = agent_dynamic_data['A4']
B1 = agent_dynamic_data['B1']
B2 = agent_dynamic_data['B2']
B3 = agent_dynamic_data['B3']
B4 = agent_dynamic_data['B4']
x0_1 = agent_dynamic_data['x01']
x0_2 = agent_dynamic_data['x02']
x0_3 = agent_dynamic_data['x03']
x0_4 = agent_dynamic_data['x04']
max_iter = agent_dynamic_data['Tfinal']
u_max = agent_dynamic_data['umax']
#print(agent_dynamic_data)

""" Functions """

def convex_solve(x,N_p,u_max):
    n = x.shape[0]
    #m = B.shape[1]
    J = cp.Variable()
    x = cp.Variable((n,1))
    u = cp.Variable((n,1))
    constraints = []
    for i in range(u_max):
        constraints.append(cp.abs(u[i]) <= u_max)
        
    prob = cp.Problem(cp.Minimize(J), constraints)
    prob.solve(solver=cp.SCS)
    return u.value

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
for k in max_iter:
    convex_solve()

""" Plotting """
