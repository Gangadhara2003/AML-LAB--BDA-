import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


data = pd.read_csv('data/mushrooms.csv')
for label in data.columns:
    data[label] = LabelEncoder().fit(data[label]).transform(data[label])


target_col = 'class' if 'class' in data.columns else 'target'
X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', max_depth=1)
ada = AdaBoostClassifier(estimator=model, n_estimators=400, learning_rate=1.0, random_state=42)
boostmodel = ada.fit(X_train, y_train)
print(type(boostmodel).__name__)
print(round(ada.score(X_test, y_test), 4))
