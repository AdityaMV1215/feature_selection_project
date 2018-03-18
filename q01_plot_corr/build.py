# Default imports
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap

data = pd.read_csv('data/house_prices_multivariate.csv')


# Write your solution here:
def plot_corr(data, size=11):
    corr = data.corr()
    fig, ax = subplots(figsize=(size, size))
    set_cmap("YlOrRd")
    ax.matshow(corr)
    xticks(range(len(corr.columns)), corr.columns, rotation=90)
    yticks(range(len(corr.columns)), corr.columns)
    return ax
