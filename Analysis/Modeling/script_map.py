from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier

import numpy as np
# dividing X, y into train and test data

import functions.get_svm as get_svm
import os

Y_CAT='nImg'
GROUP_CAT='nVpn'
SPLIT = 20
from functions.load_dataset import load_map


X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map(split=SPLIT, y_cat=Y_CAT, group_cat=GROUP_CAT,load=1)


#X_p, y_p, X_i, y_i, vpn_p, vpn_i
X_train, X_test, y_train, y_test = get_svm.get_train_test(
    X_i, y_i, test_train_ratio=0.7,random_state=None)


# training a linear SVM classifier

#svm_model_linear = SVC(kernel='rbf').fit(X_train, y_train)
svm_linear =LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.00001,
     verbose=0).fit(X_train, y_train)


#svm_linear= svm_model_linear.fit(X_p, y_p)
svm_predictions = svm_linear.predict(X_test)
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
# model accuracy for X_test
accuracy = svm_linear.score(X_test, y_test)
print('SVM acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=20,n_jobs=-1)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
rf_predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)
# model accuracy for X_test
accuracy = clf.score(X_test, y_test)
print('RF acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)

logistic = linear_model.LogisticRegression(C=1e5)
logistic_linear= logistic.fit(X_train, y_train)
logistic_predictions = logistic_linear.predict(X_test)
cm = confusion_matrix(y_test, logistic_predictions)
# model accuracy for X_test
accuracy = logistic_linear.score(X_test, y_test)
print('\nLinear acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)

