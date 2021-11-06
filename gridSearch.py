# Grid search - a way to find the best parameters of a model.
# svm(gamma=0.1, C=1)

# SVM compare values for paramters:
#   C   : 0.01, 0.1, 1, 10, 100
#  gamma: 0.01, 0.1, 1, 10, 100

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()

X_train_val, X_test, y_train_val, y_test = train_test_split(iris.data, iris.target)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val)

print("Train size: {}".format(X_train.shape))
print("Test  size: {}".format(X_test.shape))
print("valid size: {}".format(X_valid.shape))

from sklearn.model_selection import cross_val_score
import numpy as np

best_score = 0
best_params = {}
for gamma in [0.01, 0.1, 1, 10, 100]:
    for C in [0.01, 0.1, 1, 10, 100]:
        svm = SVC(gamma=gamma, C=C)
        scores = cross_val_score(svm, X_train_val, y_train_val)

        score = np.mean(scores)
        if score > best_score:
            best_score = score
            best_params = {"C": C, "gamma": gamma}

svm = SVC(**best_params)
# svm = SVC(C = best_params["C"]..)
svm.fit(X_train_val, y_train_val)

print("Score: {:.2f}".format(svm.score(X_test, y_test)))

# print("Best score: {:.2f}".format(best_score))
# print("Best params: {}".format(best_params))

# Grid search with cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {"C": [0.01, 0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10, 100]}
gs = GridSearchCV(SVC(), param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

gs.fit(X_train, y_train)

# print("Score: {:.2f}".format(gs.score(X_test, y_test)))
# print("params: {}".format(gs.best_params_))
# print("cross-val score: {:.2f}".format(gs.best_score_))

# print(gs.best_estimator_)

# svc = gs.best_estimator_
# Fit model...
# print(svc.predict([[1, 1, 1, 1]]))

import pandas as pd
import mglearn
import matplotlib.pyplot as plt

# print(pd.DataFrame(gs.cv_results_))

results = pd.DataFrame(gs.cv_results_)
scores = np.array(results.mean_test_score).reshape(5, 5)

mglearn.tools.heatmap(scores, xlabel="gamma", xticklabels=param_grid["gamma"],
     ylabel="C", yticklabels=param_grid["C"])
plt.show()


