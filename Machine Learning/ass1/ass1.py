from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# id:5-5-5
df = pd.read_csv("week2.csv")
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = df.iloc[:, 2]

# # part i) plot the data


def PartAi():
    fig, ax = plt.subplots()
    ax.set_title("Visualisation of data")
    ax.scatter(*X[y == 1].T, s=50, marker='+',
               label="points with target value = +1", c='lime')
    ax.scatter(*X[y == -1].T, s=10, marker='o',
               label="points with target value = -1", c='b')
    ax.legend(bbox_to_anchor=(0.3, 1.2))
    ax.set_xlabel("first feature x1")
    ax.set_ylabel("second feature x2")
    plt.show()

# part ii) logistic regression model with sklearn


def PartAii():

    model = LogisticRegression(penalty='none', solver='lbfgs')
    model.fit(X, y)
    w0 = model.intercept_[0]
    w1 = model.coef_[0][0]
    w2 = model.coef_[0][1]

    # # theta 0
    # print("model.intercept_ = " + str(w0))
    # # theta 1 and theta 2
    # print("model.coef_ = " + str(w1) + " " + str(w2))
    # # parameter values theta0 + theta1*x1 + theta2*x2

    # part iii) plot predictions
    # decision boundary
    x = np.array([-1, 1])
    decision_boundary = (-w1/w2)*x + (-w0/w2)
    pred = model.predict(X)
    df['pred'] = pred
    print(len(X[pred == 1]))
    print(len(X[pred == -1]))
    fig, ax = plt.subplots()
    ax.set_title(
        "Visualisation of data with decision boundary and predicted values")

    ax.scatter(*X[pred == 1].T, s=50, marker='o',
               label="predictions with target value = +1", c='black')
    ax.scatter(*X[pred == -1].T, s=50, marker='o',
               label="predictions with target value = -1", c='red')

    ax.scatter(*X[y == 1].T, s=30, marker='+',
               label="points with target value = +1", c='lime')
    ax.scatter(*X[y == -1].T, s=10, marker='o',
               label="points with target value = -1", c='b')
    ax.plot(x, decision_boundary, lw=2, ls='-',
            c='r', label='decision boundary')

    ax.legend(bbox_to_anchor=(0.2, 1.1))
    ax.set_xlabel("first feature x1")
    ax.set_ylabel("second feature x2")
    plt.show()


def PartBi():
    for c in (0.001, 0.01, 0.1, 1, 10, 100, 1000):
        model = LinearSVC(C=c).fit(X, y)
        pred = model.predict(X)
        w0 = model.intercept_
        w1 = model.coef_[0][0]
        w2 = model.coef_[0][1]
        x = np.array([-1, 1])
        decision_boundary = (-w1/w2)*x + (-w0/w2)
        print("parameter values for linear SVM classifier with C= " + str(c))
        print("theta0 = " + str(w0) + " theta1 = " +
              str(w1)+" theta2 = " + str(w2))

        # parameter values for linear SVM classifier with C= 0.001
        # theta0 = [0.24386245] theta1 = -0.008311814549474108 theta2 = 0.4574743015830204
        # parameter values for linear SVM classifier with C= 0.01
        # theta0 = [0.39731206] theta1 = 0.018888933715088506 theta2 = 1.1457246892035486
        # parameter values for linear SVM classifier with C= 0.1
        # theta0 = [0.55046008] theta1 = 0.039938825844197806 theta2 = 1.6152091673262439
        # parameter values for linear SVM classifier with C= 1
        # theta0 = [0.59013621] theta1 = 0.043193285958026906 theta2 = 1.7229554743132327
        # parameter values for linear SVM classifier with C= 10
        # theta0 = [0.59505326] theta1 = 0.04350745931674381 theta2 = 1.7361191735106842
        # parameter values for linear SVM classifier with C= 100
        # theta0 = [0.58911082] theta1 = 0.08624588336967831 theta2 = 1.7558885343696982
        # ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
        # parameter values for linear SVM classifier with C= 1000
        # theta0 = [-0.19714935] theta1 = 0.10387207212033917 theta2 = 2.0595671380951677

        # part ii)
        fig, ax = plt.subplots()
        ax.set_title(
            "SVM with C= "+str(c))

        ax.scatter(*X[pred == 1].T, s=50,  marker='o',
                   label="predictions with target value = +1", c='black')
        ax.scatter(*X[pred == -1].T, s=50,  marker='o',
                   label="predictions with target value = -1", c='red')

        ax.scatter(*X[y == 1].T, s=30,  marker='+',
                   label="points with target value = +1", c='lime')
        ax.scatter(*X[y == -1].T, s=10,  marker='o',
                   label="points with target value = -1", c='b')
        ax.plot(x, decision_boundary, lw=2, ls='-',
                c='r', label='decision boundary')

        ax.legend(bbox_to_anchor=(0.2, 1.1))
        ax.set_xlabel("first feature x1")
        ax.set_ylabel("second feature x2")
        plt.show()


def PartC():

    xpartc = np.column_stack((X1, X2, pow(X1, 2), pow(X2, 2)))

    model = LogisticRegression(penalty='none', solver='lbfgs')
    model.fit(xpartc, y)
    pred = model.predict(xpartc)

    w0 = model.intercept_[0]
    w1 = model.coef_[0][0]
    w2 = model.coef_[0][1]
    w3 = model.coef_[0][2]
    w4 = model.coef_[0][3]

    print(str(*model.intercept_) + str(*model.coef_))

    fig, ax = plt.subplots()
    ax.set_title(
        "Visualisation of data with decision boundary and predicted values")

    ax.scatter(*X[pred == 1].T, s=50,  marker='o',
               label="predictions with target value = +1", c='black')
    ax.scatter(*X[pred == -1].T, s=50,  marker='o',
               label="predictions with target value = -1", c='red')

    ax.scatter(*X[y == 1].T, s=30,  marker='+',
               label="points with target value = +1", c='lime')
    ax.scatter(*X[y == -1].T, s=10,  marker='o',
               label="points with target value = -1", c='b')
    # ax.plot(x, decision_boundary, lw=2, ls='-',c='r',label='decision boundary')

    ax.legend(bbox_to_anchor=(0.2, 1.1))
    ax.set_xlabel("first feature x1")
    ax.set_ylabel("second feature x2")
    plt.show()

    return


# PartAi()
# PartAii()
# PartBi()
PartC()
