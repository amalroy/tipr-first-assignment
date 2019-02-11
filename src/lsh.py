# Implement code for Locality Sensitive Hashing here!
import numpy as np
def reduce_dim(X,k,w=10):
    n=X.shape[0]
    d=X.shape[1]
    random_hash=np.zeros((n,k))
    for i in range(n):
        x=X[i,:]
        for j in range(k):
            r=np.random.normal(0,1,d)
            b=np.random.uniform(0,w)
            random_hash[i,j]=np.floor((np.dot(x,r)+b)/w)
    return random_hash
#def classify(X,w):
#    X_hash=np.

