from sklearn.tree import DecisionTreeClassifier
import pandas as pd

import os
for dirname, _, filenames in os.walk('/opt/openknowledge'):
    for filename in filenames:
        print(dirname + " : " + filename)

df_train: pd.DataFrame = pd.read_csv('./input/train.csv')
df_train.describe()

# DecisionTreeClassifier()


