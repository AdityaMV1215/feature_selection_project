# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here
def forward_selected(data, model):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    cols = list(X.columns.values)
    r2 = -100
    f = []
    r = []
    flag = 1
    while flag == 1:
        if len(r) == 0:
            for val in cols:
                if r2 == -100:
                    model.fit(X.loc[:,val].values.reshape(-1,1), y)
                    y_pred = model.predict(X.loc[:,val].values.reshape(-1,1))
                    r2 = r2_score(y, y_pred)
                    feature = val

                else:
                    model.fit(X.loc[:,val].values.reshape(-1,1), y)
                    y_pred = model.predict(X.loc[:,val].values.reshape(-1,1))
                    if r2_score(y, y_pred) > r2:
                        r2 = r2_score(y, y_pred)
                        feature = val

            f.append(feature)
            r.append(r2)
            cols.remove(feature)

        else:
            flag = 1
            while flag == 1:
                r2 = -100
                prevlen = len(f)
                for val in cols:
                    if r2 == -100:
                        f.append(val)
                        model.fit(X.loc[:,f], y)
                        y_pred = model.predict(X.loc[:,f])
                        r2 = r2_score(y, y_pred)
                        feature = val

                    else:
                        f.remove(feature)
                        f.append(val)
                        model.fit(X.loc[:,f], y)
                        y_pred = model.predict(X.loc[:,f])
                        if r2_score(y, y_pred) > r2:
                            r2 = r2_score(y, y_pred)
                            feature = val

                        else:
                            f.append(feature)
                            f.remove(val)

                if not r2 == -100:
                    r.append(r2)
                if feature in cols:
                    cols.remove(feature)
                if len(f) <= prevlen:
                    flag = 0

    return f,r
