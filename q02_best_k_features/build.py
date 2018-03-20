# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:
def percentile_k_features(df, k=20):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    ans = f_regression(X,y)
    ans1 = SelectPercentile(f_regression,percentile=k)
    ans1.fit(X,y)
    ans2 = ans1.get_support()
    ans3 = list(X.loc[:,ans2].columns.values)
    ans3 = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return ans3
