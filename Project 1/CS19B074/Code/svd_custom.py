import numpy as np

'''
Singular Value Decomposition
(strictly for square matrices)
----------------------------
parameters - A (m * n matrix) (this is a real-valued matrix)
returns    - U (matrix of left eigen vectors)
           - D (array of min(m,n) eigen values in ascending order)
           - Vt(tranpose of right eigen vectors)
'''
def svd(A):

    # step 1: form At @ A
    At = np.transpose(A)
    S = At @ A

    # step 2: form Vt . Matrix of eigen vectors of At @ A
    evals, V = np.linalg.eig(S) # note, here eigenvalues will be in ascending order
    ind = np.argsort(evals) # sorting order
    ind = ind[::-1] # decreasing order
    evals = np.abs(evals[ind])
    V = np.array([row[ind] for row in V])
    for i in range(len(V)):
        norm = np.linalg.norm(V[i])
        V[i] = V[i] / norm
    Vt = np.transpose(V)
    D = np.sqrt(evals) # eigen values, sqrt applied over the eigen values of At @ A

    # # step 3, form cols of U, one at a time
    m = np.shape(A)[0]
    U = [[0 for _ in range(m)] for _ in range(m)] # m * m initialization
    for i in range (m):
        col = (A @ Vt[i]) / D[i]
        for j in range(m):
            U[j][i] = col[j]

    return np.array(U), D, Vt


