import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# dataset #id:25-25-25

df = pd.read_csv("week3.csv")
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

Xtest = []
grid = np.linspace(-1.5, 1.5)
for i in grid:
    for j in grid:
        Xtest.append([i, j])
Xtest = np.array(Xtest)


def partIa():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], y, c='b', label='data')
    ax.set_xlabel("X1 features")
    ax.set_ylabel("X2 features")
    ax.set_zlabel("y target")
    ax.set_title("3D Visualisation of Data")
    ax.legend()
    ax.view_init(20, 60)
    plt.show()


def partIb():
    Xtrain, _, ytrain, _ = train_test_split(X, y, test_size=0.2)
    Xtrain_poly = PolynomialFeatures(5).fit_transform(Xtrain)
    c = [1, 10, 100, 1000]
    for i in c:
        model = Lasso(alpha=1/(2*i)).fit(Xtrain_poly, ytrain)
        print("C: " + str(i))
        print("intercept: " + str(model.intercept_))
        print("coefficients: " + str(model.coef_))


def partIc():
    Xtrain, _, ytrain, _ = train_test_split(X, y, test_size=0.2)
    Xtrain_poly = PolynomialFeatures(5).fit_transform(Xtrain)
    Xtest_poly = PolynomialFeatures(5).fit_transform(Xtest)

    dummy = DummyRegressor(strategy="mean").fit(Xtrain_poly, ytrain)
    ydummy = dummy.predict(Xtest_poly)

    c = [1, 10, 100, 1000]
    c = [100]
    for i in c:
        model = Lasso(alpha=1/(2*i)).fit(Xtrain_poly, ytrain)
        ypred = model.predict(Xtest_poly)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(Xtest)
        ax.plot_trisurf(Xtest[:, 0], Xtest[:, 1], ypred, color='g', alpha=.6)
        ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data')
        ax.set_xlabel('X1 features')
        ax.set_ylabel('X2 features')
        ax.set_zlabel('y target')
        ax.set_title('3D Data visualisation with Lasso model.\n C=' + str(i))
        handles, _ = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='g', label='Predictions', alpha=0.6)
        handles.append(patch)
        ax.legend(handles=handles)
        ax.view_init(20, 60)
        plt.show()
    # Dummy Baseline
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xtest[:, 0], Xtest[:, 1], ydummy, color='g', alpha=.6)
    ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data')
    ax.set_xlabel('X1 features')
    ax.set_ylabel('X2 features')
    ax.set_zlabel('y target')
    ax.set_title('Baseline model with dummy prediction')
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='g', label='Predictions', alpha=0.6)
    handles.append(patch)
    ax.legend(handles=handles)
    ax.view_init(20, 60)
    plt.show()


def partIeb():
    Xtrain, _, ytrain, _ = train_test_split(X, y, test_size=0.2)
    Xtrain_poly = PolynomialFeatures(5).fit_transform(Xtrain)
    c = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for i in c:
        model = Ridge(alpha=1/(2*i)).fit(Xtrain_poly, ytrain)
        print("C: " + str(i))
        print("intercept: " + str(model.intercept_))
        print("coefficients: " + str(np.round(model.coef_, decimals=3)))
    return


def partIec():
    Xtrain, _, ytrain, _ = train_test_split(X, y, test_size=0.2)
    Xtrain_poly = PolynomialFeatures(5).fit_transform(Xtrain)
    Xtest_poly = PolynomialFeatures(5).fit_transform(Xtest)

    dummy = DummyRegressor(strategy="mean").fit(Xtrain_poly, ytrain)
    ydummy = dummy.predict(Xtest_poly)
    print(len(ydummy))

    c = [0.001, 0.1, 1, 10, 100, 1000]
    for i in c:
        model = Ridge(alpha=1/(2*i)).fit(Xtrain_poly, ytrain)
        ypred = model.predict(Xtest_poly)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(Xtest[:, 0], Xtest[:, 1], ypred, color='g', alpha=.6)
        ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data')
        ax.set_xlabel('X1 features')
        ax.set_ylabel('X2 features')
        ax.set_zlabel('y target')
        ax.set_title('3D Data visualisation with Ridge model.\n C=' + str(i))
        handles, _ = ax.get_legend_handles_labels()
        patch = mpatches.Patch(color='g', label='Predictions', alpha=0.6)
        handles.append(patch)
        ax.legend(handles=handles)
        ax.view_init(20, 60)
        plt.show()
    # Dummy Baseline
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xtest[:, 0], Xtest[:, 1], ydummy, color='g', alpha=.6)
    ax.scatter(X[:, 0], X[:, 1], y, color='b', label='Data')
    ax.set_xlabel('X1 features')
    ax.set_ylabel('X2 features')
    ax.set_zlabel('y target')
    ax.set_title('Baseline model with dummy prediction')
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='g', label='Predictions', alpha=0.6)
    handles.append(patch)
    ax.legend(handles=handles)
    ax.view_init(20, 60)
    plt.show()
    return


def partIIa():
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []
    Ci_range = [1, 5, 10, 25, 50, 100, 1000]
    for Ci in Ci_range:
        model = Lasso(alpha=1/(2*Ci))
        temp = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.title("Lasso MSE vs C")
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('Mean square error')
    plt.xlim((0, 100))
    plt.show()


def partIIc():
    kf = KFold(n_splits=5)
    mean_error = []
    std_error = []
    Ci_range = [0.1, 0.5, 1, 5, 10, 25, 50, 100, 1000]
    for Ci in Ci_range:
        model = Ridge(alpha=1/(2*Ci))
        temp = []
        for train, test in kf.split(X):
            model.fit(X[train], y[train])
            ypred = model.predict(X[test])
            temp.append(mean_squared_error(y[test], ypred))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    plt.title("Ridge MSE vs C")
    plt.errorbar(Ci_range, mean_error, yerr=std_error)
    plt.xlabel('Ci')
    plt.ylabel('Mean square error')
    plt.xlim((0, 25))
    plt.ylim((0.1, 0.16))
    plt.show()


partIa()
# partIb()
# partIc()
# partIeb()
# partIec()
# partIIa()
# partIIc()
