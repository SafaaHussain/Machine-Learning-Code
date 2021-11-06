from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data = load_iris()

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])

rf = RandomForestClassifier(max_depth=3, n_estimators=200)

rf.fit(X_train, y_train)

print("Train score: {:.2f}".format(rf.score(X_train, y_train)))
print("Test  score: {:.2f}".format(rf.score(X_test, y_test))) 