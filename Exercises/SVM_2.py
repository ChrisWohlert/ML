# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:50:22 2018

@author: Sila
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
 
# import some data to play with
iris = datasets.load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris['data'], iris['target'], random_state=0)

y = []
for i in Y_train:
   if i == 2:
      y = np.append(y, [1])
   else:
      y = np.append(y, [0])

y_test = []
for i in Y_test:
   if i == 2:
      y_test = np.append(y_test, [1])
   else:
      y_test = np.append(y_test, [0])


h = .02  # step size in the mesh
 
# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 100000.0  # SVM regularization parameter
#svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=0.5).fit(X_train, y)
poly_svc = svm.SVC(kernel='poly', degree=5, C=10000000).fit(X_train, y)
#lin_svc = svm.LinearSVC(C=C).fit(X, y)
 

print("Training RBF scores: {:.2f}".format(rbf_svc.score(X_train, y)))
print("Test RBF scores: {:.2f}".format(rbf_svc.score(X_test,y_test)))

print("Training poly scores: {:.2f}".format(poly_svc.score(X_train, y)))
print("Test poly scores: {:.2f}".format(poly_svc.score(X_test,y_test)))