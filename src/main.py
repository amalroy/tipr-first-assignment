import argparse
from numpy import genfromtxt
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
parser = argparse.ArgumentParser()
parser.add_argument('--test-data')
parser.add_argument('--test-label')
parser.add_argument('--dataset')
args=parser.parse_args()
print(args.test_data)
print(args.test_label)
print(args.dataset)
if (args.dataset=='twitter'):
    with open(args.test_data) as file:
        tweets=file.readlines()
    X=count_vect.fit_transform(tweets)
    y=genfromtxt(args.test_label,delimiter=' ')
else:
    X=genfromtxt(args.test_data,delimiter=' ')
    y=genfromtxt(args.test_label,delimiter=' ')
print(X.shape)
print(y)
if __name__ == '__main__':
    print('Welcome to the world of high and low dimensions!')
    # The entire code should be able to run from this file!
