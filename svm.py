from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()

print(data['DESCR'])


X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])

# Scale data so that the features are between 0 and 1.


train_min   = X_train.min(axis=0)
train_max   = X_train.max(axis=0)
train_range = train_max - train_min

X_train_scaled = (X_train - train_min) / train_range
X_test_scaled  = (X_test  - train_min) / train_range


s = SVC(kernel='rbf', C =9, gamma=0.1)

s.fit(X_train_scaled, y_train)

print("Train score: {:.2f}".format(s.score(X_train_scaled, y_train)))
print("Test  score: {:.2f}".format(s.score(X_test_scaled, y_test)))


# Deep learning

# Feature engineering

# Apply a model to a real dataset




