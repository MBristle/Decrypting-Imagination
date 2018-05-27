
import functions.get_svm as get_svm
import os

from functions.load_dataset import load_summary
# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_summary()
# importing necessary libraries
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
# dividing X, y into train and test data


X_train, X_test, y_train, y_test = get_svm.get_train_test(
    X_i, y_i, test_train_ratio=0.7,random_state=None)

# training a linear SVM classifier

#svm_model_linear = SVC(kernel='rbf').fit(X_train, y_train)
svm_model_linear =LinearSVC(C=300, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0).fit(X_train, y_train)


svm_linear= svm_model_linear.fit(X_p, y_p)
svm_predictions = svm_linear.predict(X_test)
# creating a confusion matrix
cm = confusion_matrix(y_test, svm_predictions)
# model accuracy for X_test
accuracy = svm_model_linear.score(X_test, y_test)
print('SVM acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)


logistic = linear_model.LogisticRegression(C=1e5)
logistic_linear= logistic.fit(X_p, y_p)
logistic_predictions = logistic_linear.predict(X_test)
cm = confusion_matrix(y_test, logistic_predictions)
# model accuracy for X_test
accuracy = logistic_linear.score(X_test, y_test)
print('\nLinear acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)




#print('logloss SVM ', log_loss(y_test,svm_model_linear.predict_proba(X_test)))

#print('logloss RF ',log_loss(y_test, clf_probs))

# Build a forest and compute the feature importances

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=20,
                              random_state=0)
X=X_train
y=y_train
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
#plt.show()



# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
clf = RandomForestClassifier(n_estimators=20)
clf.fit(X_train, y_train)
clf_probs = clf.predict_proba(X_test)
rf_predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)
# model accuracy for X_test
accuracy = clf.score(X_test, y_test)
print('RF acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)


from sklearn.tree import export_graphviz
import pandas as pd
df=pd.DataFrame(X, columns=('xr','yr','pur','pupil','numFix','numBlink','xl','yl'))
clf=DecisionTreeClassifier()
clf.fit(df,y)
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
print(clf)
os.system('dot -Tpng tree.dot -o tree.png')

# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data

clf_probs = clf.predict_proba(X_test)
rf_predictions = clf.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)
# model accuracy for X_test
accuracy = clf.score(X_test, y_test)
print('RF acc: %0.2f \n\n'%accuracy, 'CM: \n', cm)
