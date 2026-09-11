#Code for verifying a WT-triple.
# Usage:
#    `python.exe verify.py [folder]`
# where [folder] is one of `asymptotic` or `explicit`

import sys
import sympy

folder_name = sys.argv[1]

def read_matrix(filename):
    out = []
    with open(filename, 'r') as f:
        for line in f:
            row = line.strip().split(',')
            out.append([sympy.Rational(x) for x in row])
    return sympy.Matrix(out)

G = read_matrix(f'{folder_name}/G.txt')
H = read_matrix(f'{folder_name}/H.txt')
p = read_matrix(f'{folder_name}/p.txt').row(0)
upsilon = read_matrix(f'{folder_name}/upsilon.txt')[0]

K, N = G.shape
sigma = upsilon * 2
delta = sympy.Rational(upsilon, N)

def verify_alpha_beta(G, H, p):
    '''Verify the alpha + beta upper bound.'''
    K, N = G.shape
    min_conv = float('inf')
    for j in range(0, 2*N + 1):
        conv = sum(p[k] * (G.row(k) @ H[k, j:j+N].T)[0] for k in range(K)) * delta
        if conv < min_conv:
            min_conv = conv
    assert all(sum(G.row(k)) * delta == 1 for k in range(K))
    print('checked integral g == 1 for all g')
    assert sum(p) == 1
    print('checked sum p == 1')
    assert min_conv >= 1
    print('checked every convolution >= 1')
    beta = sum(sum(x**2 for x in G.row(k)) * p[k] for k in range(K)) * delta
    alpha = sum(sum(x**2 for x in H.row(k)) * p[k] for k in range(K)) * delta - sigma
    return alpha, beta

alpha, beta = verify_alpha_beta(G, H, p)
print(f'alpha: {float(alpha)}')
print(f'beta: {float(beta)}')
print(f'2*sqrt(alpha*beta): {float(2*sympy.sqrt(alpha * beta))}')
print(f'k_0: {float(4*(1+upsilon*sympy.sqrt(beta/alpha))**2)}')
