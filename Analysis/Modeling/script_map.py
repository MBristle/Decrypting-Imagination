from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
import pandas as pd
import numpy as np
# dividing X, y into train and test data
from functions.load_dataset import load_map
import functions.get_svm as get_svm
import os

Y_CAT = 'nCat'
GROUP_CAT = 'nVpn'
COMP = ["p->p", "i->i", "p->i", "i->p"]
DS_SPLIT = np.arange(8)
CV_SPLIT = 5

skf_cv = LeaveOneGroupOut()

sz= len(DS_SPLIT)*CV_SPLIT*len(COMP)
track = dict()
track['DS_Split'] = np.empty(sz)
track['Comparison'] = ["" for x in range(sz)]
track['CV_split'] = np.empty(sz)
track['SVMacc'] = np.empty(sz)
track['RFacc'] = np.empty(sz)
track['LRacc'] = np.empty(sz)

LEN_COMP = len(COMP)

it=0
for k in DS_SPLIT:
    split = 2 ** k

    X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map(split=split, y_cat=Y_CAT, group_cat=GROUP_CAT, load=1)

    X_train = [X_p, X_i, X_p, X_i]
    y_train = [y_p, y_i, y_p, y_i]
    X_test = [X_p, X_i, X_i, X_p]
    y_test = [y_p, y_i, y_i, y_p]

    Xlen = len(X_train)

    for i in range(LEN_COMP):
        ii=1
        for train, test in skf_cv.split(X=X_train[i % Xlen], y=y_train[i % Xlen],groups=vpn_p):

            c_X_train = X_train[i % Xlen][train]
            c_X_test = X_test[i % Xlen][test]
            c_y_train = y_train[i % Xlen][train]
            c_y_test = y_test[i % Xlen][test]

            track['DS_Split'][it] = 2 ** k
            track['Comparison'][it] = COMP[i]
            track['CV_split'][it] = ii
            ii+=1
            print('start DS_SPlit: ',split,', Comp: ',COMP[i],', CVsplit:', ii)
            # training a linear SVM classifier
            # svm_model_linear = SVC(kernel='rbf').fit(X_train, y_train)
            svm_linear = LinearSVC(C=1, class_weight=None, dual=False, fit_intercept=True,
                                   intercept_scaling=1, loss='squared_hinge', max_iter=10000,
                                   multi_class='ovr', penalty='l2', random_state=None, tol=0.00001,
                                   verbose=0).fit(c_X_train,   c_y_train)
            svm_predictions = svm_linear.predict(c_X_test)
            cm = confusion_matrix(c_y_test, svm_predictions)
            accuracy = svm_linear.score(c_X_test, c_y_test)
            track['SVMacc'][it] = accuracy
            print('SVM acc: %0.2f' % accuracy)
            #print('SVM acc: %0.2f \n\n' % accuracy, 'CM: \n', cm)
            if False:
                # Train uncalibrated random forest classifier on whole train and validation
                # data and evaluate on test data
                rf = RandomForestClassifier(n_estimators=20, n_jobs=-1).fit(c_X_train, c_y_train)
                rf_predictions = rf.predict(c_X_test)
                cm = confusion_matrix(c_y_test, rf_predictions)
                accuracy = rf.score(c_X_test, c_y_test)
                track['RFacc'][it] = accuracy
                print('RF acc: %0.2f' % accuracy)
                #print('RF acc: %0.2f \n\n' % accuracy, 'CM: \n', cm)

                logistic = linear_model.LogisticRegression(C=1e5).fit(c_X_train, c_y_train)
                logistic_predictions = logistic.predict(c_X_test)
                cm = confusion_matrix(c_y_test, logistic_predictions)
                accuracy = logistic.score(c_X_test, c_y_test)
                track['LRacc'][it] = accuracy
                print('Linear acc: %0.2f'% accuracy)# \n\n' % accuracy, 'CM: \n', cm)
            it+=1

track=pd.DataFrame(track)
track.to_csv('DS_SPLIT_comp_%d.csv' %len(DS_SPLIT), index=False)