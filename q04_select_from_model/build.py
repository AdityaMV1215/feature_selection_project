# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')


# Your solution code here
def select_from_model(df):
    np.random.seed(9)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    ans = SelectFromModel(RandomForestClassifier())
    ans.fit(X,y)
    ans1 = ans.get_support()
    return list(X.loc[:,ans1].columns.values)
