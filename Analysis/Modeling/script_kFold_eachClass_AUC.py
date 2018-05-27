# Perform Permutation tests
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
import random as rnd
import functions.get_svm as get_svm
import functions.load_dataset as ld
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import multiprocessing

# import Data in features X and targets y
Xs_p, ys_p, Xs_i, ys_i, group_s_p, group_s_i = ld.load_summary()
Xm_p, ym_p, Xm_i, ym_i, group_m_p, group_m_i = ld.load_map()

TEST_TRAIN_RATIO = 0.7
n_jobs = multiprocessing.cpu_count()
print(f'cpu cores: {n_jobs:d}')
rnd.seed = 0
RANDOM_STATE = rnd.randint(0, 2 ** 32 - 1)
print(f'Random state: {RANDOM_STATE:d}')
COMPARISON = ["IID: p->p", "IID: i->i", "IID: p->i", "IID: i->p",
              "Vpn: p->p", "Vpn: i->i", "Vpn: p->i", "Vpn: i->p"]
LEN_COMP = len(COMPARISON)
GROUP = [None, None, None, None,
         group_s_p, group_s_i, group_s_p, group_s_i,
         None, None, None, None,
         group_m_p, group_m_i, group_m_p, group_m_i]

skf_cv = StratifiedKFold(n_splits=5, random_state=RANDOM_STATE, shuffle=True)
gkf_cv = LeaveOneGroupOut()  # GroupKFold(n_splits=5)
CROSSVAL = [skf_cv, skf_cv, skf_cv, skf_cv, gkf_cv, gkf_cv, gkf_cv, gkf_cv,
            skf_cv, skf_cv, skf_cv, skf_cv, gkf_cv, gkf_cv, gkf_cv, gkf_cv, ]

X_train = [Xs_p, Xs_i, Xs_p, Xs_i, Xm_p, Xm_i, Xm_p, Xm_i]
y_train = [ys_p, ys_i, ys_p, ys_i, ym_p, ym_i, ym_p, ym_i]
X_test = [Xs_p, Xs_i, Xs_i, Xs_p, Xm_p, Xm_i, Xm_i, Xm_p]
y_test = [ys_p, ys_i, ys_i, ys_p, ym_p, ym_i, ym_i, ym_p]

Xlen = len(X_train)

svm_model = list()
class_tested = list()
test_selection = list()

y_score = list()
probas = list()
y_tested = list()
X_tested = list()

tprs_all = list()
aucs_all = list()
fpr_all = list()
tpr_all = list()
roc_auc_all = list()
mean_fpr_all = list()

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                         random_state=RANDOM_STATE))
for i in range(LEN_COMP):  #
    print("start CV testing: ", COMPARISON[i])

    y_train_b, d = get_svm.binarize(y_train[i % Xlen])
    y_test_b, _ = get_svm.binarize(y_test[i % Xlen])

    for k in range(d):
        print(' class: ', k)

        y_train_bk = y_train_b[:, k]
        y_test_bk = y_test_b[:, k]

        svm_model_c = []
        test_selection_c = []
        X_tested_c = []
        y_tested_c = []

        y_score_c = []
        probas_c = []

        tprs = []
        aucs = []
        fpr_c = []
        tpr_c = []
        roc_auc = []
        mean_fpr = np.linspace(0, 1, 100)

        ii = 0
        for train, test in CROSSVAL[i].split(X=X_train[i % Xlen], y=y_train_bk, groups=GROUP[i]):
            print('     it: ', ii)

            test_selection_c.append(test)
            X_tested_c.append(X_test[i % Xlen][test])
            y_tested_c.append(y_test_bk[test])

            # train
            svm_model_tmp = classifier.fit(X_train[i % Xlen][train], y_train_bk[train])
            svm_model_c.append(svm_model_tmp)

            # test
            probas_ = svm_model_tmp.predict_proba(X_test[i % Xlen][test])
            probas_c.append(probas_)
            y_score_tmp = svm_model_tmp.predict(X_test[i % Xlen][test])
            y_score_c.append(y_score_tmp)

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

        # store variables

        svm_model.append(svm_model_c)
        class_tested.append(k)
        test_selection.append(test_selection_c)

        X_tested.append(X_tested_c)
        y_tested.append(y_tested_c)

        y_score.append(y_score_c)

        tprs_all.append(tprs)
        aucs_all.append(aucs)
        fpr_all.append(fpr_c)
        tpr_all.append(tpr_c)
        mean_fpr_all.append(mean_fpr)

np.savez('each_class_r%d' % RANDOM_STATE,
         comparison=COMPARISON[i],
         class_tested=np.asarray(class_tested),
         svm_model=svm_model,
         test_selection=test_selection,
         y_score=y_score,
         probas=probas,
         y_tested=y_tested,
         X_tested=X_tested,
         tprs_all=tprs_all,
         aucs_all=aucs_all,
         fpr_all=fpr_all,
         tpr_all=tpr_all,
         mean_fpr_all=mean_fpr_all
         )