import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import  StratifiedKFold
from functions.load_dataset import load_map
from functions.load_dataset import load_summary
import sklearn.metrics as mt

# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map()

# Create the RFE object and compute a cross-validated score.
svc = SVC(C=1, class_weight=None, dual=False, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=10000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.00001,
     verbose=0)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(n_splits=4),
               n_jobs=1)
rfecv.fit(X_p, y_p)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

ranking = rfecv.ranking_.reshape(X_p[0].shape)

# Plot pixel ranking
plt.bar(['xr','yr','dur','pupil','numOfFix','numOfBlink','xl','yl'], height= ranking)
plt.title("Ranking of features with RFE")
plt.show()