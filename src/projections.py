# Implement code for random projections here!
import numpy as np
def random_proj(X,k,s):
    d=X.shape[0]
    R=np.zeros((k,d))
    for i in range(0,k):
        for j in range(0,d):
            r=np.random.rand()
            if (r <= 1/(2*s)):
                R[i,j]=np.sqrt(s)
            elif (r > 1/(2*s)  and r<=1/s):
                R[i,j]=-np.sqrt(s)
    return np.matmul(R,X)

