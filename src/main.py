import argparse
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import lsh
import bayes
import projections
import nn
def reduce_dim(X,loc):
    k=X.shape[1]
    for d in range(2,int(np.ceil(k/2)),2):
        fname=loc+'/X_dim_'+str(d)+'.csv'
        X_red=projections.random_proj(X,d)
        np.savetxt(fname,X_red,delimiter=' ')
count_vect = CountVectorizer()
parser = argparse.ArgumentParser()
parser.add_argument('--test-data')
parser.add_argument('--test-label')
parser.add_argument('--dataset')
parser.add_argument('--mode')
args=parser.parse_args()
discrete=False
if (args.dataset=='twitter'):
    discrete=True
    with open(args.test_data) as file:
        tweets=file.readlines()
    X=count_vect.fit_transform(tweets)
    y=genfromtxt(args.test_label,delimiter=' ')
else:
    X=genfromtxt(args.test_data,delimiter=' ')
    y=genfromtxt(args.test_label,delimiter=' ')
X=normalize(X)
def bayes_train_test(X,y):
    n_splits=10
    kf = KFold(n_splits,shuffle=True)
    kf.get_n_splits(X)
    accuracy=[]
    f1_macro=[]
    f1_micro=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred=bayes.predict(X_test,X_train,y_train)
        accuracy.append(accuracy_score(y_test,y_pred))
        f1_macro.append(f1_score(y_test,y_pred,average='macro'))
        f1_micro.append(f1_score(y_test,y_pred,average='micro'))
    acc=np.mean(np.asarray(accuracy))
    f1_ma=np.mean(np.asarray(f1_macro))
    f1_mi=np.mean(np.asarray(f1_micro))
    print("Test accuracy ::",acc)
    print("Test macro F1 Score ::",f1_ma)
    print("Test micro F1 Score ::",f1_mi)
    return acc,f1_ma,f1_mi
def knn_train_test(X,y,k):
    n_splits=10
    kf = KFold(n_splits,shuffle=True)
    kf.get_n_splits(X)
    accuracy=[]
    f1_macro=[]
    f1_micro=[]
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_pred=nn.cluster_knn(X_test,X_train,y_train,k)
        accuracy.append(accuracy_score(y_test,y_pred))
        f1_macro.append(f1_score(y_test,y_pred,average='macro'))
        f1_micro.append(f1_score(y_test,y_pred,average='micro'))
    acc=np.mean(np.asarray(accuracy))
    f1_ma=np.mean(np.asarray(f1_macro))
    f1_mi=np.mean(np.asarray(f1_micro))
    print("Test accuracy ::",acc)
    print("Test macro F1 Score ::",f1_ma)
    print("Test micro F1 Score ::",f1_mi)
    return acc,f1_ma,f1_mi
def bayes_sklearn(X,y):
    n_splits=10
    kf = KFold(n_splits,shuffle=True)
    kf.get_n_splits(X)
    accuracy=[]
    f1_macro=[]
    f1_micro=[]
    clf = GaussianNB()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        accuracy.append(accuracy_score(y_test,y_pred))
        f1_macro.append(f1_score(y_test,y_pred,average='macro'))
        f1_micro.append(f1_score(y_test,y_pred,average='micro'))
    acc=np.mean(np.asarray(accuracy))
    f1_ma=np.mean(np.asarray(f1_macro))
    f1_mi=np.mean(np.asarray(f1_micro))
    print("Test accuracy ::",acc)
    print("Test macro F1 Score ::",f1_ma)
    print("Test micro F1 Score ::",f1_mi)
    return acc,f1_ma,f1_mi
def knn(X,y,k):
    n_splits=10
    kf = KFold(n_splits,shuffle=True)
    kf.get_n_splits(X)
    accuracy=[]
    f1_macro=[]
    f1_micro=[]
    clf = KNeighborsClassifier(n_neighbors=k)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        accuracy.append(accuracy_score(y_test,y_pred))
        f1_macro.append(f1_score(y_test,y_pred,average='macro'))
        f1_micro.append(f1_score(y_test,y_pred,average='micro'))
    acc=np.mean(np.asarray(accuracy))
    f1_ma=np.mean(np.asarray(f1_macro))
    f1_mi=np.mean(np.asarray(f1_micro))
    print("Test accuracy ::",acc)
    print("Test macro F1 Score ::",f1_ma)
    print("Test micro F1 Score ::",f1_mi)
    return acc,f1_ma,f1_mi
if __name__ == '__main__':
    # The entire code should be able to run from this file!
    #bayes_train_test(X,y)
    #print(X)
    #reduce_dim(X,'../data/'+args.dataset)
    #bayes_sklearn(X,y)
    #knn(X,y,5)
    k=X.shape[1]
    dims=range(2,int(np.ceil(k/2)),2)
    n_run=len(dims)
    accuracy=np.zeros(n_run)
    f1_macro=np.zeros(n_run)
    f1_micro=np.zeros(n_run)
    
    loc='../data/'+args.dataset
    for d in dims:
        fname=loc+'/X_dim_'+str(d)+'.csv'
        X_red=genfromtxt(fname,delimiter=' ')
        y=genfromtxt(args.test_label,delimiter=' ')
        X=normalize(X)
        #acc,f1_ma,f1_mi=knn_train_test(X_red,y,5)
        #acc,f1_ma,f1_mi=bayes_train_test(X,y)
        #acc,f1_ma,f1_mi=bayes_sklearn(X,y)
        acc,f1_ma,f1_mi=knn(X,y,5)
        accuracy[int((d-1)/2)]=acc
        f1_macro[int((d-1)/2)]=f1_ma
        f1_micro[int((d-1)/2)]=f1_mi
    plt.plot(dims,accuracy)
    plt.xlabel('Dimension')
    plt.ylabel('Accuracy')
    plt.show()
    #    X_red=projections.random_proj(X,d)
    #    acc,f1_ma,f1_mi=knn_train_test(X_red,y,5)
    #    f1

