# Implement Bayes Classifier here!
def sep_class(X,y):
    sep={}
    for i in range(len(y)):
        x=X[i]
        label=y[i]
        if(label not in sep):
            sep[label]=[]
        sep[label].append(x)
    return sep
