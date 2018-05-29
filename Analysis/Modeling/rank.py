from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

from functions.load_dataset import load_summary

# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_summary()


# Create the RFE object and rank each pixel
svc = SVC(kernel="linear")
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X_p, y_p)
ranking = rfe.ranking_.reshape(X_p[0].shape)

# Plot pixel ranking
plt.bar(['xr','yr','dur','pupil','numOfFix','numOfBlink','xl','yl'],height= ranking)
plt.title("Ranking of features with RFE")
plt.show()