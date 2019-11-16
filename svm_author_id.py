#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
from sklearn.svm import SVC    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

count = 0
t0 = time()
#features_train = features_train[:(int)(len(features_train)/100)]
#labels_train = labels_train[:(int)(len(labels_train)/100)]
clf = SVC(kernel='rbf', C=10000.0, decision_function_shape='ovr').fit(features_train,labels_train)
#clf.fit(features_train,labels_train)
#SVC(C=10000.0, cache_size=200, class_weight=None, coef0=0.0,
#    decision_function_shape='ovr', degree=3, gamma=1.0, kernel='rbf',
#    max_iter=-1, probability=False, random_state=None, shrinking=True,
#    tol=0.001, verbose=False)
print ("training time:", round(time()-t0, 3), "s")
t0 = time()
pred = clf.predict(features_test)
for i in range(len(pred)):
	if(pred[i]== 1):
		count = count + 1
print ("predictinging time:", round(time()-t0, 3), "s")

#### store your predictions in a list named pred



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print(acc)
print(count)
#########################################################
### your code goes here ###

#########################################################


