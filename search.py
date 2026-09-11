#Code for the WT-triple search procedure outlined in the paper.

import math
import numpy as np
from scipy.fft import fft, ifft, fftshift
import cvxpy as cp
import dccp
import matplotlib.pyplot as plt

#Parameters
N = 100
upsilon = 10
target_k0 = float('inf')
Kmin = 2
Kmax = float('inf')
#Folder to save results to
prefix = 'asymptotic'

sigma = 2*upsilon
delta = upsilon / N

def init(Kmin):
    widths = [int(t * N) for t in np.linspace(1/10, 1, Kmin)]

    Gp = cp.Parameter((Kmin, N), pos=True)
    #each row is an indicator of an interval
    Gp.value = np.vstack([np.hstack([np.zeros((1, (N - width) // 2)),
                                     np.ones((1, width)),
                                     np.zeros((1, N - (width + (N - width) // 2)))])
                          / (width * delta)
                          for width in widths])
    return Gp

verbose = False
prec = 1e-5

def solve(problem, cc=False):
    if cc:
        problem.solve(method='dccp', seed=cc,
                      solver='CLARABEL', verbose=verbose, ignore_dpp=True,
                      tol_gap_abs=prec, tol_gap_rel=prec, tol_feas=prec,
                      tol_infeas_abs=prec, tol_infeas_rel=prec,
                      tol_ktratio=prec)
    else:
        try:
            problem.solve(solver='CLARABEL', verbose=verbose, ignore_dpp=True,
                          tol_gap_abs=prec, tol_gap_rel=prec, tol_feas=prec,
                          tol_infeas_abs=prec, tol_infeas_rel=prec,
                          tol_ktratio=prec)
        except cp.error.SolverError:
            pass
        c = prec
        while problem.status != 'optimal':
            c *= 2
            try:
                problem.solve(solver='CLARABEL', verbose=verbose, ignore_dpp=True,
                          tol_gap_abs=c, tol_gap_rel=c, tol_feas=c,
                          tol_infeas_abs=c, tol_infeas_rel=c,
                          tol_ktratio=c)
            except cp.error.SolverError:
                pass

def stack(v, length):
    return cp.hstack([v[:,None] for _ in range(length)])

def quad_over_lin_vec(X, y):
    '''Canonizer for sum_{ij} X^2_{ij} / y_i.'''
    T = cp.Variable(X.shape[0])
    constraints = []
    for i in range(X.shape[0]):
        constraints.append(cp.SOC(t=y[i] + T[i],
                                  X=cp.hstack([y[i] - T[i],
                                               2*X[i]])))
    return T, constraints

def opt_H(Gp):
    K, N = Gp.shape
    pv = cp.Variable(K, pos=True)
    Hv1 = cp.Variable((K, N))
    Hv2 = cp.Variable((K, N))
    Hv = cp.hstack([Hv1, stack(pv, N), Hv2])

    Th, Thconstraints = quad_over_lin_vec(Hv, pv)
    alphah = -sigma + cp.sum(Th) * delta
    betah = cp.sum([cp.sum_squares(Gp[i]) * pv[i]
                    for i in range(K)]) * delta

    convsh = []
    for j in range(0, 2*N + 1):
        if K != 1: #workaround for a bug in cvxpy
            convh = cp.trace(Gp @ Hv[:, j:j+N].T) * delta
        else:
            convh = (Gp[0] @ Hv[0, j:j+N]) * delta
        convsh.append(convh)
        
    probh = cp.Problem(cp.Minimize(alphah + betah),
                       [conv >= 1 for conv in convsh] + Thconstraints
                       + [cp.sum(pv) == 1])
    solve(probh)

    Hp1 = cp.Parameter((K, N))
    Hp2 = cp.Parameter((K, N))
    Hp = cp.hstack([Hp1, np.ones((K, N)), Hp2])
    pp = cp.Parameter(K, pos=True)
    
    p = np.array(pv.value)
    p[p == 0] = 1
    
    Hp1.value = Hv1.value / p[:, None]
    Hp2.value = Hv2.value / p[:, None]
    pp.value = pv.value
    return Gp, Hp, pp

def opt_G(Hp):
    K, N3 = Hp.shape
    N = N3 // 3
    pv = cp.Variable(K, pos=True)
    Gv = cp.Variable((K, N), pos=True)

    Tg, Tgconstraints = quad_over_lin_vec(Gv, pv)
    alphag = -sigma + cp.sum([cp.sum_squares(Hp[i]) * pv[i]
                              for i in range(K)]) * delta
    betag = cp.sum(Tg) * delta

    convsg = []
    for j in range(0, 2*N + 1):
        if K != 1: #workaround for a bug in cvxpy
            convg = cp.trace(Gv @ Hp[:, j:j+N].T) * delta
        else:
            convg = (Gv[0] @ Hp[0, j:j+N]) * delta
        convsg.append(convg)
    
    probg = cp.Problem(cp.Minimize(alphag + betag),
                       [conv >= 1 for conv in convsg] + Tgconstraints
                       + [cp.sum(Gv, 1) * delta == pv, cp.sum(pv) == 1])
    solve(probg)

    Gp = cp.Parameter((K, N), pos=True)
    pp = cp.Parameter(K, pos=True)

    G = Gv.value
    for i in range(len(G)):
        if np.all(G[i] == 0):
            G[i] = 1
    Gp.value = G / (np.sum(G, 1)[:, None] * delta)
    pp.value = pv.value
    return Gp, Hp, pp

resolution = 10000*N
def get_f(G, H, p):
    K, N = G.shape
    f = sum(p[i] * fftshift(ifft(fft(H[i] * delta, resolution) / fft(G[i] * delta, resolution)))
            for i in range(K)).real
    midpoint = len(f) // 2
    f = f[midpoint:midpoint + 2*N + 1] / delta
    f[f < 0] = 0
    return f

seed = 2026

def new_G_row(G, H, p):
    seed = 2026
    val, k0 = evaluate(G, H, p)
    old_G = np.array(G)
    f = get_f(G, H, p)
    mu = sum(f) * delta - sigma
    
    gv = cp.Variable(N, pos=True)
    lamb = cp.Variable(pos=True)
    nu = cp.sum_squares(cp.convolve(f, gv) * delta) * delta - lamb - sigma
    
    prob = cp.Problem(cp.Maximize(nu),
                      [cp.sum_squares(gv) * delta <= lamb,
                       cp.sum(gv) * delta == 1])
    solve(prob, cc=seed)

    bound = 2*mu - nu.value
    attempts = 1
    while bound > val and attempts < 10:
        seed += 1
        attempts += 1
        gv.value = None
        lamb.value = None
        solve(prob, cc=seed)
        bound = 2*mu - nu.value

    Gp = cp.Parameter((K+1, N), pos=True)
    g = gv.value
    newline = g / (np.sum(g) * delta)
    Gp.value = np.vstack([old_G, newline])
    return Gp

def evaluate(G, H, p):
    K, N = G.shape
    G = G / (np.sum(G, 1)[:, None] * delta) #normalize each row of G
    #rescale H so all the relevant convolutions are >= 1
    min_conv = float('inf')
    for j in range(0, 2*N + 1):
        conv = sum(p[i] * G[i] @ H[i, j : j + N] for i in range(K)) * delta
        if conv < min_conv:
            min_conv = conv
    H = H / min_conv
    beta = sum(G[i] @ G[i] * p[i] for i in range(K)) * delta
    alpha = sum(H[i] @ H[i] * p[i] for i in range(K)) * delta - sigma
    return (2*math.sqrt(alpha * beta),
            4*(1 + upsilon * math.sqrt(beta / alpha))**2)

def write(G, H, p):
    K, N = G.shape
    val, k0 = evaluate(G, H, p)
    np.savetxt(f'{prefix}/G.txt', G)
    np.savetxt(f'{prefix}/H.txt', H)
    np.savetxt(f'{prefix}/p.txt', p)
    with open(f'{prefix}/upsilon.txt', 'w') as f:
        f.write(str(upsilon))

Gp = init(Kmin)
best = 1.9
Hp = None
pp = None
K = Kmin
while K <= Kmax:
    print(f'{K=}')
    improved = True
    best_inner = float('inf')
    best_inner_at = None
    best_inner_k0 = float('inf')
    while improved:
        improved = False
        Gp, Hp, pp = opt_H(Gp)
        G, H, p = Gp.value, Hp.value, pp.value
        val1, k0 = evaluate(G, H, p)
        print('H optimized', f'b_inf={val1}', f'k_0={k0}')
        if val1 < best and k0 < target_k0:
            best = val1
            write(G, H, p)
        if val1 < best_inner and (k0 < target_k0 or k0 < best_inner_k0):
            if val1 < best_inner:
                improved = True
            best_inner = val1
            if k0 < best_inner_k0:
                best_inner_k0 = k0
            best_inner_at = G, H, p
        Gp, Hp, pp = opt_G(Hp)
        G, H, p = Gp.value, Hp.value, pp.value
        val2, k0 = evaluate(G, H, p)
        print('G optimized', f'b_inf={val2}', f'k_0={k0}')
        if val2 < best and k0 < target_k0:
            best = val2
            write(G, H, p)
        if val2 < best_inner and (k0 < target_k0 or k0 < best_inner_k0):
            if val2 < best_inner:
                improved = True
            best_inner = val2
            if k0 < best_inner_k0:
                best_inner_k0 = k0
            best_inner_at = G, H, p
    if best_inner_at is None:
        G, H, p = Gp.value, Hp.value, pp.value
    else:
        G, H, p = best_inner_at
    
    Gp = new_G_row(G, H, p)
    K += 1
    while p.min() < prec:
        K -= 1
        min_at = p.argmin()
        G = np.delete(Gp.value, [min_at], 0)
        Gp = cp.Parameter((K, N))
        Gp.value = G
        print('Deleted a pair with probability', p.min())
        p = np.delete(p, [min_at], 0)
    Hp = None
    pp = None

