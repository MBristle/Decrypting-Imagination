import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from sklearn.model_selection import  StratifiedKFold,LeaveOneGroupOut
from functions.load_dataset import load_map
from functions.load_dataset import load_summary
import sklearn.metrics as mt

# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map()

# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(n_splits=4),
               n_jobs=-1)
rfecv.fit(X_i, y_i)

print("Optimal number of features : %d" % rfecv.n_features_)
print('print plot')
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