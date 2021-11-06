# import os
# import mglearn
# import pandas as pd

# path = os.path.join(mglearn.datasets.DATA_PATH, "adult.data")

# data = pd.read_csv(path, header=None, index_col=False,
#             names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
#  'marital-status', 'occupation', 'relationship', 'race', 'gender',
#  'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
#  'income'])

# data = data[['age', 'workclass', 'education', 
#             'gender', 'hours-per-week', 'occupation', 'income']]


# print(data.head())


# print(data.gender.value_counts())

# print("Before:\n {}".format(list(data.columns)))

# data_dummy = pd.get_dummies(data)

# print("After:\n {}".format(list(data_dummy.columns)))

# print(data_dummy.head())

# print(list(data_dummy.iloc(0)))


# import mglearn
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt

# X, y = mglearn.datasets.make_wave()

# # X_train, X_test, y_train, y_test = train_test_split(X, y)
# # lr = LinearRegression().fit(X_train, y_train)
# # print("Before: {:.2f}".format(lr.score(X_test, y_test)))

# from sklearn.preprocessing import KBinsDiscretizer

# kbd = KBinsDiscretizer(n_bins=10, strategy='uniform', encode="onehot-dense")
# kbd.fit(X)

# # print("edges: ", kbd.bin_edges_)

# X_binned = kbd.transform(X)

# X_combined = np.hstack([X, X_binned])

# # print(X_combined)
# # # print(X)
# # # print(X_binned)
# # # print(X_binned.toarray())
# # X_train, X_test, y_train, y_test = train_test_split(X_binned, y)
# # lr = LinearRegression().fit(X_train, y_train)
# # print("After: {:.2f}".format(lr.score(X_test, y_test)))

# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# line_binned = kbd.transform(line)

# line_combined = np.hstack([line, line_binned])

# lr = LinearRegression().fit(X_combined, y)

# plt.plot(line, lr.predict(line_combined))
# plt.scatter(X[: ,0], y)
# plt.vlines(kbd.bin_edges_[0], -3, 3)
# plt.show()

# import mglearn
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt

# X, y = mglearn.datasets.make_wave()

# # X_train, X_test, y_train, y_test = train_test_split(X, y)
# # lr = LinearRegression().fit(X_train, y_train)
# # print("Before: {:.2f}".format(lr.score(X_test, y_test)))

# from sklearn.preprocessing import KBinsDiscretizer

# kbd = KBinsDiscretizer(n_bins=10, strategy='uniform', encode="onehot-dense")
# kbd.fit(X)

# # print("edges: ", kbd.bin_edges_)

# X_binned = kbd.transform(X)

# X_combined = np.hstack([X_binned, X * X_binned])

# # print(X_combined)
# # # print(X)
# # # print(X_binned)
# # # print(X_binned.toarray())
# # X_train, X_test, y_train, y_test = train_test_split(X_binned, y)
# # lr = LinearRegression().fit(X_train, y_train)
# # print("After: {:.2f}".format(lr.score(X_test, y_test)))

# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# line_binned = kbd.transform(line)

# line_combined = np.hstack([line_binned, line * line_binned])

# lr = LinearRegression().fit(X_combined, y)

# plt.plot(line, lr.predict(line_combined))
# plt.scatter(X[: ,0], y)
# plt.vlines(kbd.bin_edges_[0], -3, 3)
# plt.show()

# import mglearn
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import PolynomialFeatures

# X, y = mglearn.datasets.make_wave()

# pf = PolynomialFeatures(degree=10, include_bias=False)
# pf.fit(X)

# X_p = pf.transform(X)

# print(X_p.shape)
# print(X_p)
# print(pf.get_feature_names())

# line = np.linspace(-3, 3, 1000).reshape(-1, 1)
# line_p = pf.transform(line)

# lr = LinearRegression().fit(X_p, y)

# plt.plot(line, lr.predict(line_p))
# plt.scatter(X[: ,0], y)
# plt.show()

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

data = load_boston()

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pf = PolynomialFeatures(degree=2).fit(X_train_scaled)

X_train_poly = pf.transform(X_train_scaled)
X_test_poly  = pf.transform(X_test_scaled)

print(X_train.shape)
print(X_train_poly.shape)

print(pf.get_feature_names())

from sklearn.linear_model import LinearRegression

lr = LinearRegression().fit(X_train_scaled, y_train)
print("Before: {:.2f}".format(lr.score(X_test_scaled, y_test)))

lr = LinearRegression().fit(X_train_poly, y_train)
print("After: {:.2f}".format(lr.score(X_test_poly, y_test)))




