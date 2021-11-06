# Univariate Statistics Feature Selection. 
# - Statistically signifigance for output pertaining each single feature.

# from sklearn.feature_selection import SelectPercentile
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# import numpy as np
# import matplotlib.pyplot as plt


# data = load_breast_cancer()

# rand = np.random.RandomState(42)

# noise = rand.normal(size=(len(data.data), 50))

# X_noisy = np.hstack([data.data, noise])

# #|   normal   |     noise                |
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 1
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0

# X_train, X_test, y_train, y_test = train_test_split(X_noisy, data.target)


# sp = SelectPercentile(percentile=40)

# X_train_s = sp.fit_transform(X_train, y_train)
# X_test_s  = sp.transform(X_test)
# # sp.fit(X_train, y_train)
# # X_train_s = sp.transform(X_train)

# # print(X_train.shape)
# # print(X_train_s.shape)

# mask = sp.get_support() # Boolean mask

# # print(mask)

# plt.matshow(mask.reshape(1, -1))
# plt.yticks()
# plt.show()

# [x0, x1, x2, x3, x4, x5,...] data
# [1 , 1 , 0 , 1 , 0 , 0, ...] mask
# [x0, x1, x3,...]             result



# Model-based feature selection
# - model needs some measure of importance for each feature.
#      - Tree models have feature_importances_
#      - Linear models have coefficients - abs().

# Consider all features at once, capturing interaction, if the
# model can. 
# from sklearn.feature_selection import SelectFromModel
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_breast_cancer
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestClassifier


# data = load_breast_cancer()

# rand = np.random.RandomState(42)

# noise = rand.normal(size=(len(data.data), 50))

# X_noisy = np.hstack([data.data, noise])

# #|   normal   |     noise                |
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 1
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# # 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0

# X_train, X_test, y_train, y_test = train_test_split(X_noisy, data.target)


# sp = SelectFromModel(
#     RandomForestClassifier(n_estimators=60),
#     threshold="1.25*median"
# )

# X_train_s = sp.fit_transform(X_train, y_train)
# X_test_s  = sp.transform(X_test)
# # sp.fit(X_train, y_train)
# # X_train_s = sp.transform(X_train)

# # print(X_train.shape)
# # print(X_train_s.shape)

# mask = sp.get_support() # Boolean mask

# # print(mask)

# plt.matshow(mask.reshape(1, -1))
# plt.yticks()
# plt.show()

# Iterative feature selection
# - series of models are built with varying numbers of features.

# Two ways:
#    - Start with no features and add one feature at a time.
#    - Start with all features and take one feature away at a time.

# Computationally expensive

# Recursive Feature Elimination (RFE)
# 1. start with all features
# 2. build a model
# 3. discard the least important one.
# 4. Repeat steps 2-3 with left over features 
#    until a set amount of features is reached

from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


data = load_breast_cancer()

rand = np.random.RandomState(42)

noise = rand.normal(size=(len(data.data), 50))

X_noisy = np.hstack([data.data, noise])

#|   normal   |     noise                |
# 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 1
# 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0
# 30, 1.12.. , 4, 34, 14012k, 14, 21, 412     ---> 0

X_train, X_test, y_train, y_test = train_test_split(X_noisy, data.target)


rfe = RFE(
    RandomForestClassifier(n_estimators=60),
    n_features_to_select=30
)

X_train_s = rfe.fit_transform(X_train, y_train)
X_test_s  = rfe.transform(X_test)
# rfe.fit(X_train, y_train)
# X_train_s = rfe.transform(X_train)

# print(X_train.shape)
# print(X_train_s.shape)

mask = rfe.get_support() # Boolean mask

#[a, b, c, d, e, f, g]
#[0, 1, 1, 1, 0, 0, 0]

# print(mask)

plt.matshow(mask.reshape(1, -1))
plt.yticks()
plt.show()
