# -*- coding: utf-8 -*-
"""
Created on Mon September 19 10:02:23 2022

@author: sila
"""

#Import scikit-learn dataset library
from audioop import avg
from sklearn import datasets

#Load dataset
cancer = datasets.load_breast_cancer()

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print data(feature)shape
print(cancer.data.shape)

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test

#Import svm model
from sklearn import svm
from sklearn.linear_model import LogisticRegression

#Create a svm Classifier
clf = LogisticRegression(C=1000, max_iter=10000) #svm.SVC(kernel='rbf', C=1000) # Also test Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#For further evaluation, you can also check precision and recall of model.
precision = metrics.precision_score(y_test, y_pred);

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:", precision)

recall = metrics.recall_score(y_test, y_pred)
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:", recall)

print((precision + recall) / 2)

