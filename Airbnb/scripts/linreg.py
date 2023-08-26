import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor

df=pd.read_csv('../data/cleaned_listings.csv')
df=df.dropna()
df.reset_index(drop=True, inplace=True) # remove unimportant columns
df2=pd.DataFrame()
df2 = df.loc[:,'host_is_superhost':'number_of_reviews']  # Select columns by range
print(df2)
X=df2.to_numpy()
y=df['review_scores_location']
print(y)

def predict():
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LinearRegression().fit(X[train], y[train])
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
    plt.ylim(0,5)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression Predictions for Location ratings')
    plt.show()

    plt.plot(y[test],ydummy,'.')
    # plt.xlim(4.2,5)
    # plt.ylim(4.2,5)
    xmin, xmax = plt.xlim()
    plt.plot([xmin, xmax], [xmin, xmax], '--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Dummy Baseline for Overall ratings')
    plt.show()

predict()
