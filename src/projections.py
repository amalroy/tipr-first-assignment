# Implement code for random projections here!
import numpy as np
from scipy import sparse
def random_proj(X,k):
    d=X.shape[1]
    R=np.zeros((d,k))
    s=3
    for i in range(0,d):
        for j in range(0,k):
            r=np.random.rand()
            if (r <= 1/(2*s)):
                R[i,j]=np.sqrt(s)/np.sqrt(k)
            elif (r > 1/(2*s)  and r<=1/s):
                R[i,j]=-np.sqrt(s)/np.sqrt(k)
    print(X.shape,R.shape)
    X_red=sparse.csr_matrix(X)*(sparse.csr_matrix(R)).todense()
    return X_red

