import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import  StratifiedKFold,LeaveOneGroupOut
from functions.load_dataset import load_map
from sklearn import preprocessing
from functions.load_dataset import load_summary
import sklearn.metrics as mt
import numpy as np

# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map()
X = X_i
y = y_i

scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear", max_iter= 10000,cache_size=3000)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(n_splits=4),
               n_jobs=-1)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)
print('print plot')
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

if False:
    ranking = rfecv.ranking_.reshape(X_p[0].shape)

    # Plot pixel ranking
    plt.bar(['xr','yr','dur','pupil','numOfFix','numOfBlink','xl','yl'], height= ranking)
    plt.title("Ranking of features with RFE")
    plt.show()
else:
    a = rfecv.ranking_.reshape(7, 7, 8)
    for i in range(8):
        b = a[:6, :6, i]==1
        plt.imshow(b)
    for i in range(3):
        b = np.mean(a[:6, :6, :], axis=i)
        plt.imshow(b)
