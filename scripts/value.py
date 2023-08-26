import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

df=pd.read_csv('../data/listings.csv')
df=df.dropna(subset=['review_scores_value'])

amenities = df.loc[:,'amenities']
price = df.loc[:,'price']
n_amenities=[]
for i in amenities:
    i = i.split(',')
    n_amenities.append(len(i))

p=[]
for i in price:
    i = i.replace('$','')
    i = i.replace(',','')
    p.append(float(i))

df2=pd.DataFrame()
df2['n_amenities']=n_amenities
df2['price']=p
df2['value_score']=df['review_scores_value'].copy()
df2=df2.dropna()
df2.reset_index(drop=True, inplace=True)
X1 = df2['price']
X2 = df2['n_amenities']
y = df2.loc[:, 'value_score']
X = np.column_stack((X1, X2))

def graph2dprices():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X[:, 0], y, c='b', label='data',s=5)
    plt.xlabel("prices ($)")
    plt.xlim(0,1000)
    plt.ylabel("rating")
    plt.title("price vs rating")
    plt.show()

def graph2damenities():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(X[:, 1], y, c='b', label='data',s=5)
    plt.xlabel("number of amenities")
    # plt.xlim(0,100)
    plt.ylabel("rating")
    plt.title("amenities vs rating")
    plt.show()

def predict():
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = Lasso(alpha=10).fit(X[train], y[train])
        ypred = model.predict(X[test])
        dummy = DummyRegressor(strategy="mean").fit(X[train], y[train])
        ydummy = dummy.predict(X[test])
    print("intercept: " + str(model.intercept_))
    print("coef: "+ str(model.coef_))
    print("MSE: "+ str(mean_squared_error(y[test],ypred)))

    plt.plot(y[test],ypred,'.')
    # plt.xlim(4.2,5)
    # plt.ylim(4.2,5)
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Lasso Regression Predictions for Value ratings')
    plt.show()

    plt.plot(y[test],ydummy,'.')
    # plt.xlim(4.2,5)
    # plt.ylim(4.2,5)
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Dummy Baseline for Value ratings')
    plt.show()

def baseline():
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
        ypred = model.predict(X[test])
    print("intercept: " + str(model.intercept_))
    print("coef: "+ str(model.coef_))
    print("MSE: "+ str(mean_squared_error(y[test],ypred)))

    plt.plot(y[test],ypred,'.')
    # plt.xlim(4.2,5)
    # plt.ylim(4.2,5)
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression Predictions for Value ratings')
    plt.show()


# graph3d()
# graph2dprices()
# graph2damenities()
predict()
# baseline()
