import math
import sys
import numpy as np
import sympy

G = np.loadtxt(sys.argv[1])
try:
    upsilon = float(np.loadtxt(sys.argv[2]))
except FileNotFoundError:
    upsilon = float(sys.argv[2])
sdp_result_filename = sys.argv[3]

K, N = G.shape
sigma = upsilon * 2
delta = sympy.Rational(upsilon, N)
den = 2**20

#round G
M = sympy.Matrix([[sympy.Rational(math.floor(x*den), den) for x in y]
                  for y in G])
for i in range(M.rows):
    correction = 1/delta - sum(M.row(i))
    j = 0
    while M[i,j] + correction < 0:
        j += 1
    M[i,j] += correction
G = M

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

with open(sdp_result_filename, 'r') as f:
    for line in f:
        if line == 'xVec = \n':
            break
    line = f.readline()
    line = line[1:-2].split(',')
    X = [sympy.Rational(v) for v in line]

H = sympy.Matrix.zeros(K, 3*N)
h_width = 2*N
h_count = K*h_width
for c in range(K):
    for r in range(3*N):
        if r < N:
            val = X[c*h_width + r]
        elif N <= r < 2*N:
            val = X[h_count + c]
        else:
            val = X[c*h_width + r - N]
        H[c,r] = val / X[h_count + c]

p = X[h_count:h_count + K]
p = sympy.Matrix(p)
p /= sum(p)
alpha, beta = verify_alpha_beta(G, H, p)
print(f'alpha: {float(alpha)}')
print(f'beta: {float(beta)}')
print(f'2*sqrt(alpha*beta): {float(2*sympy.sqrt(alpha * beta))}')
print(f'k_0: {float(4*(1+upsilon*sympy.sqrt(beta/alpha))**2)}')

with open('G_exact.txt', 'w') as f:
    for i in range(K):
        f.write(','.join(str(x) for x in G.row(i)) + '\n')

with open('H_exact.txt', 'w') as f:
    for i in range(K):
        f.write(','.join(str(x) for x in H.row(i)) + '\n')

with open('p_exact.txt', 'w') as f:
    f.write(','.join(str(x) for x in p) + '\n')

print('exact solution written to file')
