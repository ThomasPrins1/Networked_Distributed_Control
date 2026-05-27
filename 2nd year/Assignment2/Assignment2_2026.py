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
from scipy.integrate import solve_ivp
np.random.seed(5885221)


""" Variables """
tolerance = 1e-2
eta = 1e-6
a = 5 # first digit student number
b = 8 # third digit student number
c = 1 # last digid digit student number
# iterations
N = 100
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

""" Functions """
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
        if tau<h:
            temp = expm(A*(h-tau)) # 2x2
            Fu = ((np.linalg.inv(A))@(Fx-temp))@B # 2x1
        elif tau>=h:
            temp = expm(A*(2*h-tau)) # 2x2
            Fu = np.zeros((n,m)) # 2x1
        G1 = ((np.linalg.inv(A))@(temp-np.eye(n)))@B # 2x1
        F = np.block([[Fx, Fu], [np.zeros((m,n+m))]]) # 3x3
        G = np.block([[G1],[np.eye(m)]]) # 3x1
        return F,G
    else:
        # if not invertable:
        print("A is a singular matrix")
        return np.nan,np.nan,np.nan,np.nan,np.nan

def build_polytopic_model(A, B, h, tau_vals):

    vertices = []

    for tau in tau_vals:
        F, G = FG_model_delay(A, B, h, tau)

        if np.any(np.isnan(F)):
            raise ValueError(
                f"Failed to build F,G at tau={tau}"
            )

        vertices.append((F,G))

    return vertices

def convex_solve_poly(A,B,K,h,tau_vals):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable(((n+m),(n+m)), symmetric = True) #3x3
    C = 0
    constraints = [P >> eta * np.eye(P.shape[0])]
    vertices = build_polytopic_model(A, B, h, tau_vals)
    for F,G in vertices:
        Acl = F - G @ K
        constraints += [Acl.T @ P @ Acl - P << -eta * np.eye(Acl.shape[0])]
    prob = cp.Problem(cp.Minimize(0), constraints)

    try:
        prob.solve(solver=cp.MOSEK)
    except Exception as e:
        print("Solver failed:")
        print(e)
        return False
    return prob.status in [cp.OPTIMAL,cp.OPTIMAL_INACCURATE]

def convex_solve(A,B,K,h,tau_vals):
    n = A.shape[0]
    m = B.shape[1]
    P = cp.Variable(((n+m),(n+m)), symmetric = True) #3x3
    C = 0
    constraints = [P >> eta * np.eye(P.shape[0])]
    Acl = []
    for tau in tau_vals:
        F,G = FG_model_delay(A,B,h,tau)
        Acl.append(F-G@K)
    # add additional constraints from automaton
    A_automaton = [
        Acl[2]@Acl[1]@Acl[1],
        Acl[0]@Acl[1]@Acl[1],
        Acl[1]@Acl[0],
        Acl[2]@Acl[0],
        Acl[1]@Acl[2],
        ]
    for j in range(len(A_automaton)):
        constraints += [A_automaton[j].T @ P @ A_automaton[j] - P << -eta * np.eye(A_automaton[j].shape[0])]
    prob = cp.Problem(cp.Minimize(0), constraints)

    try:
        prob.solve(solver=cp.MOSEK)
    except Exception as e:
        print("Solver failed:")
        print(e)
        return False
    return prob.status in [cp.OPTIMAL,cp.OPTIMAL_INACCURATE]
    
def convex_solve_NCS_CC(A,B,Ke,h,prob_matrix):
    n = len(Ke.T)
    m = B.shape[1]
    P_list = [cp.Variable((n, n), symmetric=True) for _ in range(len(prob_matrix))]
    constraints = [P >> eta * np.eye(n) for P in P_list]

    #use MSS, p_ij needs to be set up as a matrix with the transition probabilities
    Acl_vals = all_states(A,B,h,Ke)
    for i in range(len(prob_matrix)):
        sum_term = 0
        for j in range(len(prob_matrix)):
            sum_term += prob_matrix[i, j]*(Acl_vals[j].T@P_list[j]@Acl_vals[j])
        constraints += [P_list[i] - sum_term >> eta * np.eye(n)]
    prob = cp.Problem(cp.Minimize(0), constraints)
    try:
        prob.solve(solver=cp.MOSEK)
    except Exception as e:
        print("Solver failed:")
        print(e)
        return False
    return prob.status in [cp.OPTIMAL,cp.OPTIMAL_INACCURATE]

def all_states(A,B,h,K):
    n = 3
    F1,G1 = FG_model_delay(A, B, h, 0)
    F2,G2 = FG_model_delay(A, B, h, 0.5*h)
    F3,G3 = FG_model_delay(A, B, h, 1.5*h)
    A_cl1 = F1-G1@K
    A_cl2 = F2-G2@K
    A_cl3 = F3-G3@K
    #out = np.array([[A_cl1, A_cl2@A_cl1, A_cl3@A_cl1],
    #                [A_cl1@A_cl2, A_cl2, A_cl3@A_cl2],
    #                [A_cl1@A_cl3, A_cl2@A_cl3, A_cl3]])
    out = [A_cl1,A_cl2,A_cl3]
    return out

""" Main Code """
""" Question 1 """

h_vals = np.linspace(0, h_max, N)
stable_h = []
# with extended state an extended Kappa is needed aswell using the static controller:
Ke = np.hstack((K, np.zeros((1,1)))) # 1x3

for i,h in enumerate(h_vals):
    print("progress Q1:",(i/N)*100,"%")
    tau_vals = [0,0.5*h,1.5*h]
    feasibility = convex_solve(A,B,Ke,h,tau_vals)
    stable_h.append(feasibility)

# plotting:
# Convert True/False → 1/0
stable_h = np.array(stable_h).astype(int)

# Make 2D strip for imshow
stability_strip = stable_h.reshape(1,-1)

fig, ax = plt.subplots(
    figsize=(10,2),
    constrained_layout=True
)

im = ax.imshow(
    stability_strip,
    extent=[h_vals[0], h_vals[-1], 0, 1],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    vmin=0,
    vmax=1
)

ax.set_xlabel('Sampling time h')
ax.set_yticks([])
ax.set_title(
    r'Stability over sampling time '
    r'($\tau=\{0,0.5h,1.5h\}$)'
)

cbar = fig.colorbar(im)
cbar.set_label('Stable (1=True, 0=False)')

fig.savefig(
    os.path.join(
        path,
        "A2Q1_stability_over_h.png"
    ),
    dpi=300
)

plt.show()


# This is more strict then the eigenvalue approach but less strict then the common lyapunov over all values of tau

""" Question 2 """
p_max = 0.3
q_max = 0.2
p_vals = np.linspace(0, p_max, int(N/4))
q_vals = np.linspace(0, q_max, int(N/4))
temp_index = np.where(stable_h == 1)
temp_value = h_vals[temp_index]
h_constant_vals = [max(temp_value),
                   min(temp_value),
                   sum(temp_value)/len(temp_value),
                   max(temp_value)*1.5] 


# Plotting
# Create 2x2 figure
fig, axes = plt.subplots(
    2, 2,
    figsize=(12,10),
    constrained_layout=True
)

axes = axes.flatten()

extent = [
    q_vals[0], q_vals[-1],   # x-axis
    p_vals[0], p_vals[-1]    # y-axis
]

for k,h in enumerate(h_constant_vals):
    stable_h_stoch_grid = np.zeros((len(p_vals),len(q_vals)))
    for i,p in enumerate(p_vals):
        print("progress Q2:",(i/len(p_vals))*100,"%", "for h=",h)
        for j,q in enumerate(q_vals):
            prob_matrix = np.vstack((np.array([0, 0.5, 0.5]),np.array([q, p, 1-q-p]),np.array([q, 1-q-p, p])))
            stability = convex_solve_NCS_CC(A,B,Ke,h,prob_matrix)
            stable_h_stoch_grid[i,j] = int(stability)

    im = axes[k].imshow(
        stable_h_stoch_grid,
        origin='lower',
        extent=extent,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        vmin=0,
        vmax=1
    )

    axes[k].set_title(
        rf'$h={h:.3f}$'
    )

    axes[k].set_xlabel('q')
    axes[k].set_ylabel('p')

    axes[k].set_xlim(
        q_vals[0],
        q_vals[-1]
    )

    axes[k].set_ylim(
        p_vals[0],
        p_vals[-1]
    )

cbar = fig.colorbar(
    im,
    ax=axes,
    location='right',
    pad=0.02
)

cbar.set_label(
    'Stable (1=True, 0=False)'
)

fig.suptitle(
    'Stochastic Stability Regions'
)

fig.savefig(
    os.path.join(
        path,
        "A2Q2_stochastic_stability.png"
    ),
    dpi=300
)
plt.show()

""" Question 3 """
stable_h_poly = []
for i,h in enumerate(h_vals):
    print("progress Q3:",(i/N)*100,"%")
    tau_vals = np.linspace(0, h, N)
    feasibility = convex_solve_poly(A,B,Ke,h,tau_vals)
    stable_h_poly.append(feasibility)




# Plotting
stable_numeric = np.array(
    stable_h_poly,
    dtype=int
)

plt.figure()

plt.fill_between(
    h_vals,
    0,
    stable_numeric,
    step='mid',
    alpha=0.4
)

plt.plot(
    h_vals,
    stable_numeric
)

plt.xlabel("Sampling time h")
plt.ylabel("Stability")

plt.yticks(
    [0,1],
    ["Not feasible","Feasible"]
)

plt.grid()
plt.savefig(
    os.path.join(
        path,
        "A2Q3_stability_over_h.png"
    ),
    dpi=300
)
plt.show()


def closed_loop_dynamics(t, x, A, B, K, xk):

    u = K @ xk

    dx = A @ x + B @ u

    return dx
def triggering_event(t, x, A, B, K, xk, sigma, eps):

    e = xk - x

    return (
        np.linalg.norm(e)**2
        - sigma*np.linalg.norm(x)**2
        - eps
    )
triggering_event.terminal = True
triggering_event.direction = 1
def simulate_ETC(
    A,
    B,
    K,
    x0,
    sigma,
    eps,
    r,
    tmax=20
):

    t0 = 0

    x = x0.copy()

    xk = x.copy()

    T = [t0]
    X = [x.copy()]

    event_times = [t0]

    while np.linalg.norm(x) > r and t0 < tmax:

        event_fun = lambda t, xx: triggering_event(
            t,
            xx,
            A,
            B,
            K,
            xk,
            sigma,
            eps
        )

        event_fun.terminal = True
        event_fun.direction = 1

        sol = solve_ivp(
            lambda t, xx:
                closed_loop_dynamics(
                    t,
                    xx,
                    A,
                    B,
                    K,
                    xk
                ),
            [t0, tmax],
            x,
            events=event_fun,
            max_step=0.01
        )

        # store trajectory
        T.extend(sol.t[1:])
        X.extend(sol.y.T[1:])

        # final state
        x = sol.y[:, -1]

        t0 = sol.t[-1]

        # update transmitted state
        xk = x.copy()

        event_times.append(t0)

    return (
        np.array(T),
        np.array(X),
        np.array(event_times)
    )
sigmas = [10, 50, 90]

x0 = np.array([2, 0])

r = 0.05

eps = 1e-4

for sigma in sigmas:

    T, X, events = simulate_ETC(
        A,
        B,
        K,
        x0,
        sigma,
        eps,
        r
    )

    avg_sampling = (
        events[-1] / len(events)
    )

    print(
        f"sigma={sigma}"
    )

    print(
        f"communications={len(events)}"
    )

    print(
        f"average inter-event time={avg_sampling}"
    )

    plt.plot(
        T,
        np.linalg.norm(X, axis=1),
        label=f"sigma={sigma}"
    )

    plt.axhline(
    r,
    linestyle='--',
    label='target radius'
)

plt.xlabel("time")
plt.ylabel("||x(t)||")

plt.grid()

plt.legend()

plt.show()