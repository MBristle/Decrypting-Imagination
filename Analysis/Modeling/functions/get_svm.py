# Several functions to train specific SVM models
# return parameters may vary depending on the need of further processing



def first_view_SVM(X: float, y: int, test_train_ratio: float=0.7, random_state: int=None, X_t: float = None, y_t: int = None):
    "Function splitting the DS, calc linear SVM, predict data, calc accuracy and confusion matrix"

    # importing necessary libraries
    from sklearn.svm import SVC
    from sklearn.metrics import confusion_matrix

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = get_train_test(
        X, y, test_train_ratio, random_state, X_t, y_t)

    # training a linear SVM classifier
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    # creating a confusion matrix
    cm = confusion_matrix(y_test, svm_predictions)

    # model accuracy for X_test
    accuracy = svm_model_linear.score(X_test, y_test)

    return accuracy, cm


def each_class_CV(X: float, y: int, test_train_ratio: float, random_state: int,
                  X_t: float = None, y_t: int = None,n_jobs:int=4,fit: bool=True):
    "get SVM for the class specified"

    from sklearn import svm
    from sklearn.multiclass import OneVsRestClassifier

    y, n_classes = binarize(y)
    if y_t is not None:
        y_t,_ = binarize(y_t)

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = get_train_test(
        X, y, test_train_ratio, random_state, X_t, y_t)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=0), n_jobs)
    if fit:
        y_score = classifier.fit(X_train, y_train).decision_function(X_test)
    else:
        y_score = None

    return classifier, y_score, y_test, n_classes


def permutationTesting(X: float, y: int,group=None, cv=None,n_permutations: int = 100, random_state: int = 0,n_jobs:int=4):
    "get SVM for the class specified"
    from sklearn import svm
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import permutation_test_score

    random_state = 0
    classifier = svm.SVC(kernel='linear', random_state=random_state)
    if cv is None:
        cv = StratifiedKFold(n_splits=3)

    score, permutation_scores, pvalue = permutation_test_score(
        classifier, X, y,groups=group, scoring="accuracy", cv=cv, n_permutations=n_permutations, n_jobs=n_jobs)

    return score, permutation_scores, pvalue



def get_train_test(X: float, y: int, test_train_ratio: float, random_state: int, X_t: float = None, y_t: int = None):
    "split data to train and testset depending on input"
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_train_ratio, random_state=random_state)

    if X_t is not None and y_t is not None:
        X_test = X_t
        y_test = y_t
    return X_train, X_test, y_train, y_test


def binarize(y):
    import numpy as np
    if y is None:
        return None

    from sklearn import preprocessing
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    # print(lb.classes_.shape[0])
    n_classes = lb.classes_.shape[0]
    y = lb.transform(y)
    return y, n_classes


