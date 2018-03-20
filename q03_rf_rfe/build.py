# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    ans = RFE(RandomForestClassifier(), n_features_to_select=int(X.shape[1]/2))
    ans.fit(X,y)
    return list(X.loc[:,ans.support_].columns.values)
