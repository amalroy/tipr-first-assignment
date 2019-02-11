import numpy as np
from numpy.linalg import norm
# Implement Nearest Neighbour classifier here!
def neighbors(X_train,test,k):
    n=X_train.shape[0]
    dist=np.zeros((n,))
    for i in range(n):
        dist[i]=norm(X_train[i]-test,axis=0)
    idx=np.argpartition(dist,k)
    return idx[:k]
def get_label(neighbors,y_train):
    n_labels=len(np.unique(y_train))
    count={}
    for i in neighbors:
        neigh_label=y_train[i]
        if (neigh_label in count):
            count[neigh_label]+=1
        else:
            count[neigh_label]=1

    return max(count,key=lambda k:count[k])


def cluster_knn(X_test,X_train,y_train,k):
    n_test=X_test.shape[0]
    n_labels=len(np.unique(y_train))
    pred_label=np.zeros(n_test)
    for i in range(n_test):
        pred_label[i]=get_label(neighbors(X_train,X_test[i],k),y_train)
    return pred_label

        
    

