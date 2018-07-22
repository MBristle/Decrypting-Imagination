# Perform Permutation tests
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
import random as rnd
import functions.get_svm as get_svm
from functions.load_dataset import load_map,load_summary
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import multiprocessing
from sklearn.decomposition import PCA

# import Data in features X and targets y
X, y, X_i, y_i, vpn_p, vpn_i = load_map()

TEST_TRAIN_RATIO = 0.7
n_jobs = multiprocessing.cpu_count()
print(f'cpu cores: {n_jobs:d}')
rnd.seed = 0
RANDOM_STATE = rnd.randint(0, 2**32 - 1)
COMPONENT_NUM = 8
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_TRAIN_RATIO, random_state=RANDOM_STATE)

pca = PCA(n_components=COMPONENT_NUM, whiten=True)
pca.fit(X_train)
pca_2d = pca.transform(X_train)

import pylab as pl
for i in range(0, pca_2d.shape[0]):
    if y_train[i] == 1:
        c1 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='r',    marker='+')
    elif y_train[i] == 2:
        c2 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='g',    marker='o')
    elif y_train[i] == 3:
        c3 = pl.scatter(pca_2d[i,0],pca_2d[i,1],c='b',    marker='*')
pl.legend([c1, c2, c3], ['Face', 'Art', 'Landscape'])
pl.title('Fixation training dataset with 3 classes and  known outcomes')
pl.show()

