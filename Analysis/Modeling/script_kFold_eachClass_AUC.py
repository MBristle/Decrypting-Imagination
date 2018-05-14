# Perform Permutation tests
from functions.load_dataset import load_summary
import functions.get_svm as get_svm
import functions.plot as plot
from sklearn.model_selection import StratifiedKFold,LeaveOneGroupOut
from sklearn.metrics import roc_curve, auc
from scipy import interp
import numpy as np

## import Data in features X and targets y
X_p,y_p,X_i,y_i,vpn_p,vpn_i = load_summary()

TEST_TRAIN_RATIO=0.7
RANDOM_STATE=0


COMPARISON= ["IID: p->p","IID: i->i","IID: p->i","IID: i->p",
             "Vpn: p->p","Vpn: i->i","Vpn: p->i","Vpn: i->p"]
LEN_COMP = len(COMPARISON)
GROUP = [None, None,None, None, vpn_p, vpn_i, vpn_p, vpn_i]
skf_cv = StratifiedKFold(n_splits=6,random_state=RANDOM_STATE)
logo_cv = LeaveOneGroupOut()
CROSSVAL=[skf_cv, skf_cv,skf_cv, skf_cv, logo_cv, logo_cv, logo_cv, logo_cv]

X_train=[X_p,X_i,X_p,X_i]
y_train=[y_p,y_i,y_p,y_i]
X_test=[X_p,X_i,X_i,X_p]
y_test=[y_p,y_i,y_i,y_p]
Xlen = len(X_train)

classifier=list()
class_tested=list()
y_score=list()
y_tested=list()
n_classes=list()
tprs_all = list()
aucs_all = list()
fpr_all = list()
tpr_all = list()
roc_auc_all = list()
mean_fpr_all = list()


for i in range(1):#LEN_COMP
    print("start CV testing: ", COMPARISON[i] )

    classifier_tmp, y_score_tmp,y_test_tmp,n_classes_tmp = get_svm.each_class_CV(
        X_train[i % Xlen],y_train[i%Xlen],test_train_ratio=TEST_TRAIN_RATIO,random_state=RANDOM_STATE,
        X_t=X_train[i % Xlen],y_t=y_test[i%Xlen])

    y_train_b,_=get_svm.binarize(y_train[i%Xlen])
    y_test_b,_=get_svm.binarize(y_test[i%Xlen])

    for k in range(n_classes_tmp):
        y_train_bk = y_train_b[:,k]
        y_test_bk= y_test_b[:,k]
        tprs = []
        aucs = []
        fpr_c = []
        tpr_c = []
        roc_auc = []
        mean_fpr = np.linspace(0, 1, 100)

        ii = 0
        for train, test in CROSSVAL[i].split(X_train[i%Xlen], y_train_bk, groups=GROUP[i]):
            probas_ = classifier_tmp.fit(X_train[i%Xlen][train], y_train_bk[train]).predict_proba(X_test[i%Xlen][test])

            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test_bk[test], probas_[:, 1])

            fpr_c.append(fpr)
            tpr_c.append(tpr)

            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            # plt.plot(fpr, tpr, lw=1, alpha=0.3,
            #         label='ROC fold %d (AUC = %0.2f)' % (ii, roc_auc))
            ii += 1

        #store variables
        classifier.append(classifier_tmp)
        class_tested.append(k)
        y_score.append(y_score_tmp)
        y_tested.append(y_test_tmp)
        n_classes.append(n_classes_tmp)
        tprs_all.append(tprs)
        aucs_all.append(aucs)
        fpr_all.append(fpr_c)
        tpr_all.append(tpr_c)
        mean_fpr_all.append(mean_fpr)

np.savez('tmp_each_class',
         comparison=COMPARISON[i],
         class_tested=k,
         classifier=classifier,
         y_score=y_score,
         y_tested=y_tested,
         n_classes=n_classes,
         tprs_all=tprs_all,
         aucs_all=aucs_all,
         fpr_all=fpr_all,
         tpr_all=tpr_all,
         mean_fpr_all=mean_fpr_all
         )





