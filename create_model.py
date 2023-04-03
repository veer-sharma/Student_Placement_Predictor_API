import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

df.shape

df.sample(5)

X = df.drop(columns=['placed'])
y = df['placed']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)

import pickle
pickle.dump(rf, open('model.pkl', 'wb'))