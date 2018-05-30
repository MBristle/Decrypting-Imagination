from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from functions.load_dataset import load_summary,load_map
import numpy as np
from functions.stats import bootstrap
# import Data in features X and targets y
X_p, y_p, X_i, y_i, vpn_p, vpn_i = load_map()
X = X_p
y = y_p

scaler = preprocessing.MinMaxScaler().fit(X)
X = scaler.transform(X)

# Create the RFE object and rank each pixel
svc = SVC(kernel="linear", max_iter= 10000)
rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(X, y)


if False:
    ranking = rfe.ranking_.reshape(X_p[0].shape)

    # Plot pixel ranking
    plt.bar(['xr', 'yr', 'dur', 'pupil', 'numOfFix', 'numOfBlink', 'xl', 'yl'], height=ranking)
    plt.title("Ranking of features with RFE")
    plt.show()
else:
    a = rfe.ranking_.__invert__().reshape(7, 7, 8)
    for i in range(8):
        b = a[:6, :6, i]
        print(str(i)+ '  mean: '+ str(np.mean(b.__invert__().ravel())))
        bootstrap(b.__invert__().ravel(), eval='median')
        #plt.imshow(b - b.min(), vmin=0, vmax=int((b.min() * -1) + (b.max() * -1)))
    for i in range(3):
        b = np.mean(a[:6, :6, :], axis=i)
        plt.imshow(b - b.min(), vmin=0, vmax=int((b.min() * -1) + (b.max() * -1)))
    for i in range(3):
        b = np.std(a[:6, :6, :], axis=i)
        plt.imshow(b - b.min(), vmin=0, vmax=int((b.min()) + (b.max())))