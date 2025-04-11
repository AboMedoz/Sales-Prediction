import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

# https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
df = pd.read_csv('Dataset/Train.csv')
df = df.dropna()  # Will impute them later in the Model dropping for test only

plt.rc('figure', autolayout=True)
plt.rc(
    'axes',
    labelweight='bold',
    labelsize='large',
    titleweight='bold',
    titlesize=14,
    titlepad=10
)


def make_mi_scores(x, y):
    x = x.copy()
    for col in x.select_dtypes(['object', 'category']):
        x[col], _ = x[col].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in x.dtypes]
    mi_score = mutual_info_regression(x, y, discrete_features=discrete_features, random_state=0)
    mi_score = pd.Series(mi_score, name='MI Scores', index=x.columns)
    mi_score = mi_score.sort_values(ascending=False)
    return mi_score


def plot_mi_scores(mi_scores):
    mi_scores = mi_scores.sort_values(ascending=True)
    width = np.arange(len(mi_scores))
    ticks = list(mi_scores.index)
    plt.barh(width, mi_scores)
    plt.yticks(width, ticks)
    plt.title('Mutual Information')


x = df.copy()
y = x.pop('Item_Outlet_Sales')

mi_score = make_mi_scores(x, y)
print(mi_score)

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_score)
plt.show()
