import math
import sys
import numpy as np
import sympy

def write(g, f):
    '''
    Write SDP to file.
    See https://sdpa-python.github.io/docs/formats/sdpa.html for the format specification.
    '''
    K, N = g.shape
    h_width = 2*N #the middle N values are all the same, given by p
    h_count = K*h_width
    p_count = K
    T_count = K
    a = h_count
    b = a + p_count
    
    m = h_count + p_count + T_count
    f.write(f'{m}\n')
    lines = {}
    block_struct = []
    #sum p_i >= 1
    block = 1
    block_struct.append(1)
    lines[(0, block, 1, 1)] = 1.0
    for i in range(a, b):
        lines[(i+1, block, 1, 1)] = 1.0
    #sum p_i <= 1
    block += 1
    block_struct.append(1)
    lines[(0, block, 1, 1)] = -1.0
    for i in range(a, b):
        lines[(i+1, block, 1, 1)] = -1.0
    #p_i >= 0
    for c in range(K):
        block += 1
        block_struct.append(1)
        lines[(h_count + c + 1, block, 1, 1)] = 1.0
    #convolution >= 1 * fudge_factor
    for j in range(0, 2*N + 1):
        block += 1
        block_struct.append(1)
        lines[(0, block, 1, 1)] = fudge_factor / delta
        for c in range(K):
            for r in range(N):
                h_ind = r + j
                if h_ind < N:
                    ind = c*h_width + h_ind + 1
                elif N <= h_ind < 2*N:
                    ind = h_count + c + 1 #this is an index into p, not h
                else:
                    ind = c*h_width + h_ind - N + 1
                if g[c, r]:
                    key = (ind, block, 1, 1)
                    if key in lines:
                        lines[key] += g[c, r]
                    else:
                        lines[key] = g[c, r]
    #quadratic-over-linear SDP encoding (similar to MOSEK cookbook 6.2.5)
    #We have to do a bit of finagling to move the p_i terms out of the vector portion;
    # the obvious formulation is singular.
    for c in range(K):
        block += 1
        block_struct.append(h_width + 1)
        lines[(b + c + 1, block, 1, 1)] = 1.0
        lines[(h_count + c + 1, block, 1, 1)] = -float(N)
        for raw in range(h_width):
            ind = c*h_width + raw + 1
            lines[(ind, block, 1, 2+raw)] = 1.0
            lines[(h_count + c + 1, block, 2+raw, 2+raw)] = 1.0

    f.write(f'{block}\n')
    f.write(' '.join(str(x) for x in block_struct) + '\n')
    f.write(' '.join('0.0' if i < a
                     else str(sympy.N(sum(x**2 for x in g.row(i - a)), precision)) if i < b
                     else '1.0' for i in range(m))  + '\n')
    for (a,b,c,d),e in sorted(lines.items(), key=lambda x:[x[0][1], x[0][0], x[0][2:]]):
        if isinstance(e, sympy.Rational):
            E = sympy.N(e, precision)
        else:
            E = e
        f.write(f'{a} {b} {c} {d} {E}\n')

#feel free to edit these parameters:
#upsilon = 1.688390480540569838e+00#10
#upsilon = 1.755010483406061139e+00

#see instructions.txt for these two
den = 2**20
fudge_factor = 1 + sympy.Integer(10)**(-30)

#number of digits to write to file
precision = 50

if __name__ == "__main__":
    G = np.loadtxt(sys.argv[1])
    try:
        upsilon = float(np.loadtxt(sys.argv[2]))
    except FileNotFoundError:
        upsilon = float(sys.argv[2])
    sigma = 2*upsilon
    K, N = G.shape
    delta = sympy.Rational(upsilon, N)
    
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
    with open('sidon.dat-s', 'w') as f:
        write(G, f)
    print('written to file')
