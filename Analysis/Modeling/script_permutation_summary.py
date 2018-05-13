# Perform Permutation tests
from functions.load_dataset import load_summary
import functions.get_svm as get_svm
import numpy as np
## import Data in features X and targets y
X_p,y_p,X_i,y_i,vpn_p,vpn_i = load_summary()

#get Classes
_,n_classes = get_svm.binarize(y_p)


PERMUTATIONS=2
RANDOM_STATE=0

COMPARISON= ["IID: p->p","IID: i->i","Vpn: p->p","Vpn: i->i"]
LEN_COMP = len(COMPARISON)
GROUP = [None, None, vpn_p, vpn_i]
X=[X_p,X_i,X_p,X_i]
y=[y_p,y_i,y_p,y_i]


score=np.empty(LEN_COMP)
permutation_scores=None
pvalue=np.empty(LEN_COMP)

for i in range(LEN_COMP):
    print("start Permutationtesting: ", COMPARISON[i] )
    score[i], permutation_scores_tmp, pvalue[i] = get_svm.permutationTesting(
        X[i], y[i],group=GROUP[i], n_permutations= PERMUTATIONS, random_state = RANDOM_STATE)

    if permutation_scores is None:
        permutation_scores = permutation_scores_tmp
    else:
        permutation_scores = np.append([permutation_scores], [permutation_scores_tmp], axis=0)


import numpy as np
np.savez('summary_permutation',comparison=COMPARISON, score=score, permutation_scores=permutation_scores,pvalue=pvalue)

