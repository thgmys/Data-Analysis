import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve


def PartA(df, id):
    X1 = df.iloc[:, 0]
    X2 = df.iloc[:, 1]
    X = np.column_stack((X1, X2))
    y = df.iloc[:, 2]
    # plotRawData(X, y, id)
    kf = KFold(n_splits=5)
    q_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    aLogRegChoosePolynomial(kf, X, y, q_range)
    # Ci_range = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50]
    # aLogRegChooseC(kf, X, y, Ci_range)

    if id == 'id:24-24-24-1':
        q = 1
        c = 1
        k = 25
        # aLogRegPlotPred(kf, X, y, q, c)
        k_range = range(5, 200, 10)
        # bKnnChoosek(kf, X, y, k_range)
        # bKnnPlotPred(kf, X, y, k)
        # cBaseline(kf, X, y)
        # dPlotROC(kf, X, y, q, c, k)

    elif id == 'id:24-48-24-1':
        q = 2
        c = 5
        k = 9
        # aLogRegPlotPred(kf, X, y, q, c)
        # k_range = range(5, 50, 2)
        # bKnnChoosek(kf, X, y, k_range)
        bKnnPlotPred(kf, X, y, c)
        # cBaseline(kf, X, y)
        # dPlotROC(kf, X, y, q, c, k)


def plotRawData(X, y, id):
    plt.title("Visualisation of dataset #" + id)
    plt.scatter(*X[y == 1].T, s=50, marker='+',
                label="train data target value = +1", c='lime')
    plt.scatter(*X[y == -1].T, s=10, marker='o',
                label="train data target value = -1", c='b')
    plt.legend(bbox_to_anchor=(0.3, 1.2))
    plt.xlabel("first feature x1")
    plt.ylabel("second feature x2")
    plt.show()


def aLogRegChoosePolynomial(kf, X, y, q_range):
    f1 = []
    std_error = []
    model = LogisticRegression(penalty='l2', C=1)
    for q in q_range:
        Xpoly = PolynomialFeatures(q).fit_transform(X)
        temp = []
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(f1_score(y[test], ypred))
        f1.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(q_range, f1, yerr=std_error, linewidth=1)
    plt.ylim(0.7, 1)
    plt.xlabel('q')
    plt.ylabel('f1 score')
    plt.title('f1 score with respect to degree of polynomial for model')
    plt.show()


def aLogRegChooseC(kf, X, y, Ci_range):
    f1 = []
    std_error = []
    Xpoly = PolynomialFeatures(2).fit_transform(X)

    for Ci in Ci_range:
        model = LogisticRegression(penalty='l2', C=Ci)
        temp = []
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(f1_score(y[test], ypred))
        f1.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())

    plt.errorbar(Ci_range, f1, yerr=std_error, linewidth=1)
    plt.xlabel('Ci')
    plt.ylabel('f1 score')
    plt.ylim(0.5, 1)
    plt.title('f1 score with respect to differing values for C')
    plt.show()


def aLogRegPlotPred(kf, X, y, q, c):

    model = LogisticRegression(penalty='l2', C=c)
    Xpoly = PolynomialFeatures(q).fit_transform(X)
    for train, test in kf.split(Xpoly):
        model.fit(Xpoly[train], y[train])
        ypred = model.predict(Xpoly[test])
    print(model.intercept_)
    print(model.coef_)
    plt.scatter(*X[test][ypred == 1].T, color='black', s=10,
                label='predictions target value = +1')
    plt.scatter(*X[test][ypred == -1].T, color='red', s=10,
                label='predictions target value = -1')
    plt.legend(bbox_to_anchor=(0.3, 1.2))
    plt.title('Predictions')
    plt.show()
    plt.scatter(*X[y == 1].T, s=50, marker='+',
                label="train data target value = +1", c='lime')
    plt.scatter(*X[y == -1].T, s=10, marker='o',
                label="train data target value = -1", c='b')
    plt.scatter(*X[test][ypred == 1].T, color='black', s=10)
    plt.scatter(*X[test][ypred == -1].T, color='red', s=10)
    plt.legend(bbox_to_anchor=(0.3, 1.2))
    plt.title('Predictions alongside raw data')
    plt.show()
    cConfusionMatrix(y[test], ypred)


def bKnnChoosek(kf, X, y, k_range):
    mean_error = []
    std_error = []
    for k in k_range:
        temp = []
        model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(pow(mean_squared_error(y[test], ypred), 0.5))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.errorbar(k_range, mean_error, yerr=std_error, linewidth=1)
    plt.xlabel('k')
    plt.ylabel('Root Mean Square Error')
    plt.title('RMSE with respect to k')
    plt.show()


def bKnnPlotPred(kf, X, y, k):
    model = KNeighborsClassifier(n_neighbors=k, weights='uniform')
    for train, test in kf.split(X):
        model.fit(X[train], y[train])
        ypred = model.predict(X[test])

    plt.scatter(*X[test][ypred == 1].T, color='black',
                s=10, label='predictions target value = +1')
    plt.scatter(*X[test][ypred == -1].T, color='red', s=10,
                label='predictions target value = -1')
    plt.legend(bbox_to_anchor=(0.3, 1.2))
    plt.title('Predictions')
    plt.show()
    plt.scatter(*X[y == 1].T, s=50, marker='+',
                c='lime')
    plt.scatter(*X[y == -1].T, s=10, marker='o',
                c='b')
    plt.scatter(*X[test][ypred == 1].T, color='black',
                s=10, label='predictions target value = +1')
    plt.scatter(*X[test][ypred == -1].T, color='red', s=10,
                label='predictions target value = -1')
    plt.legend(bbox_to_anchor=(0.3, 1.2))
    plt.title('Predictions alongside raw data')
    plt.show()
    cConfusionMatrix(y[test], ypred)


def cConfusionMatrix(ytest, preds):
    print(confusion_matrix(ytest, preds))
    pass


def cBaseline(kf, X, y):
    mf = DummyClassifier(strategy="most_frequent")
    st = DummyClassifier(strategy="stratified")
    for train, test in kf.split(X):
        mf.fit(X[train], y[train])
        ypredmf = mf.predict(X[test])
        st.fit(X[train], y[train])
        ypredst = st.predict(X[test])
    cConfusionMatrix(y[test], ypredmf)
    cConfusionMatrix(y[test], ypredst)


def dPlotROC(kf, X, y, q, c, k):

    lr = LogisticRegression(penalty='l2', C=c)
    Xpoly = PolynomialFeatures(q).fit_transform(X)

    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform')

    mf = DummyClassifier(strategy="most_frequent")
    st = DummyClassifier(strategy="stratified")

    for train, test in kf.split(X):
        lr.fit(Xpoly[train], y[train])
        knn.fit(X[train], y[train])
        mf.fit(X[train], y[train])
        st.fit(X[train], y[train])

    fprlr, tprlr, _ = roc_curve(
        y[test], lr.decision_function(Xpoly[test]))
    fprknn, tprknn, _ = roc_curve(y[test], knn.predict_proba(X[test])[:, 1])
    fprmf, tprmf, _ = roc_curve(y[test], mf.predict_proba(X[test])[:, 1])
    fprst, tprst, _ = roc_curve(y[test], st.predict_proba(X[test])[:, 1])
    plt.plot(fprlr, tprlr, color='red',
             label='logistic regression')
    plt.plot(fprknn, tprknn, color='blue', label='kNN')
    plt.plot(fprmf, tprmf, color='yellow',
             linestyle='dashed', label='most frequent TV')
    plt.plot(fprst, tprst, c='g', label='stratified sample')
    plt.legend()
    plt.ylabel('True Positive rate')
    plt.xlabel('False Positive rate')
    plt.title('ROC curve for different classifiers')
    plt.show()


# id:24-24-24-1
df = pd.read_csv("week4_1.csv", header=None)
PartA(df, 'id:24-24-24-1')

# id:24-48-24-1
df = pd.read_csv("week4_2.csv", header=None)
PartA(df, 'id:24-48-24-1')
