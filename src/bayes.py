# Implement Bayes Classifier here!
import numpy as np
def sep_class(X,y):
    train={}
    for i in range(len(y)):
        x=X[i]
        label=y[i]
        if(label not in train):
            train[label]=[]
        train[label].append(x)
    return train
def class_estimates(train):
    mean={}
    var={}
    for label in train:
        #calculate sample mean
        mean[label]=np.mean(np.asarray(train[label]),axis=0)
        #calculate sample variance. ddof set(unless only one example is there for label)
        #because we have to divide by n-1 to get sample variance
        dof=1
        if(len(train[label])==1):
            dof=0
        var[label]=np.var(np.asarray(train[label]),axis=0,ddof=dof)
    return mean,var
def prior_prob(train):
    p_class={}
    n=sum([len(v) for v in train.values()])
    for label in train:
        p_class[label]=len(train[label])/n
    return p_class
def likelihood(x,mean,var):
    p=0.0
    for i in range(len(x)):
        if(var[i]==0):
            if(x[i]==mean[i]):
                return 1.0
            else:
                return 0.0
        else:
            p=p-0.5*np.log(2*np.pi*var[i])-((x[i]-mean[i])**2)/(2*var[i])
    return (p)
def posterior(x,train):
    post={}
    mean,var=class_estimates(train)
    p_class=prior_prob(train)
    for label in train:
        post[label]=likelihood(x,mean[label],var[label])+np.log(p_class[label])
    return post
def predict(X_test,X_train,y_train):
    n_test=X_test.shape[0]
    train=sep_class(X_train,y_train)
    pred_label=np.zeros(n_test)
    for i in range(n_test):
        post=posterior(X_test[i,:],train)
        pred_label[i]=max(post,key=lambda k:post[k])
    return pred_label
    
