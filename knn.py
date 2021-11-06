from sklearn.neighbors import KNeighborsClassifier

from sklearn.neighbors import KNeighborsRegressor

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_iris()

# print(type(iris))

# print(iris['DESCR'])


print(iris.keys())

print(iris['target'])
print(iris['target_names'])

print(iris['data'][:5])

print(iris['feature_names'])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'])

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print("Knn train score", knn.score(X_train, y_train))
print("Knn test score ", knn.score(X_test, y_test))



X_new = np.array([5.4, .1, 1., 0.7])
X_new = X_new.reshape(1, -1)

print(knn.predict(X_new))

['setosa', 'v', 'a']
print("Prediction:", iris['target_names'][knn.predict(X_new)[0]])

