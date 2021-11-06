# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.datasets import load_boston
# from sklearn.model_selection import train_test_split

# data = load_boston()

# X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])

# lr = LinearRegression()
# lr.fit(X_train, y_train)

# print("Train score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test score : {:.2f}".format(lr.score(X_test, y_test)))

# print("\n")

# lr = Ridge(alpha=4)
# lr.fit(X_train, y_train)

# print("Train score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test score : {:.2f}".format(lr.score(X_test, y_test)))

# print("\n")

# lr = Lasso(alpha=1000)
# lr.fit(X_train, y_train)

# print("Train score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test score : {:.2f}".format(lr.score(X_test, y_test)))

# weights = lr.coef_
# num_nonzeros = 0
# for weight in weights:
#     if weight != 0:
#         num_nonzeros += 1
# print("Num features used: {}".format(num_nonzeros))

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
wine = load_wine()

# print(wine.DESCR)

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target)

lr = LogisticRegression(max_iter=100, C=0.1)
lr.fit(X_train, y_train)

print("Train score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test score : {:.2f}".format(lr.score(X_test, y_test)))

lr = LinearSVC(max_iter=100, C=3)
lr.fit(X_train, y_train)

print("Train score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test score : {:.2f}".format(lr.score(X_test, y_test)))

