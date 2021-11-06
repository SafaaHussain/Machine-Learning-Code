

# Got data, split into a test and training set.
# Fit on the training data
# Evaluate the model with both test and training via R^2 (score method).

# More robust way to assess generalization - cross-validation.


# cross-validation - statistical method that is more stable and 
# thorough.
# Split the data repeatedly and train multiple models.

# k-folds cross-validation - partition the data into k "folds" of
# ~ equal size. 
# Train a sequence of models such that:
# Model 1 uses first fold as test set and other folds as training set.
# Model 2 uses second fold as test set and other folds as training set.
# Model 3 uses third fold....
#...
# Model k used kth fold...

# Partition: [1, 2, 3, 4, 5] -- [1, 2], [3, 4], [5]

# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score

# iris = load_iris()

# lr = LogisticRegression()

# scores = cross_val_score(lr, iris.data, iris.target, cv=3)

# print(scores)

# Benefits:
# train test split - performs a single random split of data
#   This can lead to "lucky" splits or unlucky splits.
# with cross-validation, this cannot happen bnecuase we use
# all the data points - each fold is used once.

# Multiple splits and so multiple scores so we can see just how
# senstive we are to the selection of split.


# Use data more effectively. With train test split, we use 75%
# data for training, 25% for testing. With 
# 5-fold cross-validation 4/5 (80%) data for training, and 20% for testing
# 10-fold cross-validation 9/10 (90%) data for training, and 10% for testing

# Downside - the computational cost - we train k models, so ~k times slower.

# cross validation is not meant to be applied to new data. It is meant to see
# how well a model can perform on a dataset.

# print(iris.target)

# Stratified k-fold cross-validation - Splits the data such that
# proportions between classes are the same in each fold as they are
# in the entire datasets.


# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold

# iris = load_iris()

# lr = LogisticRegression()

# kf = KFold(n_splits=3, shuffle=True)


# scores = cross_val_score(lr, iris.data, iris.target, cv=kf)
# print(scores)

# Defaults:
#   - Regression: kFold
#   - Classification: Stratified kFold


# Leave-one-out cross-validation
# - k-fold validation where each fold (partition) is a single data point.
# - means pick a single point to be test set, and rest be training set.
# - do that for all N data points. 
# - every data point is tested with new models. 

# Can be very time consuming, but its usually better on smaller datasets.

# from sklearn.datasets import load_iris
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import LeaveOneOut

# iris = load_iris()

# lr = LogisticRegression()

# lo = LeaveOneOut()

# scores = cross_val_score(lr, iris.data, iris.target, cv=lo)
# print(scores)

# num_right = 0
# for score in scores:
#     if score == 1:
#         num_right += 1
# print("Percent correct: {:.2f}".format(num_right/len(scores)))


# Shuffle-split cross validation - Each split sample "train_size" many points for training
# set and "test_size" many points for test set. These are disjoint. We repeat this
# "n_splits" times.

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

iris = load_iris()

lr = LogisticRegression()

ss = ShuffleSplit(train_size=0.6, test_size=0.3, n_splits=10)

scores = cross_val_score(lr, iris.data, iris.target, cv=ss)
print(scores)

# Groups
from sklearn.model_selection import GroupKFold
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=10)
groups = [0, 0, 1, 2, 1, 2, 3, 3, 1, 0]

lr = LogisticRegression()

scores = cross_val_score(lr, X, y, groups, cv=GroupKFold(n_splits=2))
print(scores)

