#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
#from sklearn.naive_bayes import GaussianNB  
from sklearn.svm import SVC
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
#clf = GaussianNB()
clf = SVC(kernel="linear")
t0 = time()
clf.fit(features_train,labels_train)
print ("training time:", round(time()-t0, 3), "s")
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=1.0, kernel='linear',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
t0 = time()
pred = clf.predict(features_test)
print ("predictinging time:", round(time()-t0, 3), "s")

#### store your predictions in a list named pred



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(acc)