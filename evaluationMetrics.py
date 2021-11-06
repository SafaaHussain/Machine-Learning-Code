
"""
Kinds of errors. 

Cancer screening.
    False positive: Classify a healthy patient as positive (having cancer), needs to take more tests.
        - more cost and some inconvience 
        - Type I error
    Flase negative: Classify an unhealthy patient as negative - this can be bad.
        - Type II error

imbalanced datasets - when one class is much more frequent than the other.
Let's say 99% of ads the user ignore, and 1% they click.

what if we built a classifier with 99% accuracy? 
It sounds impressive, but it neglet the imbalance aspect.
"""

# from sklearn.dummy import DummyClassifier - Always predicts majority

"""
Comphrensive ways to represent the result of evaluating binary classification
is with confusion matrices.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

lr = LogisticRegression(C=0.1).fit(X_train, y_train)

cf = confusion_matrix(y_test, lr.predict(X_test))
print(cf)

dc = DummyClassifier().fit(X_train, y_train)

cf = confusion_matrix(y_test, dc.predict(X_test))
print(cf)

# Relation to accuracy - computes the accuracy.
# Accuracy = (TP + TN) / (TP+TN+FP+FN)

# Precision: how many samples predicted as positive are actually positive.
#  use when trying to limit the number of false positive.
# Precision = TP / (TP + FP)

# Recall: Measures how many positive samples are captured by the positive predictions.
# used to identify all positive samples (want to avoid false negatives)
# Recall = TP / (TP + FN)

# f-score = 2 * (precision * recall) / (precision + recall)
# f_1-score = ------- 
from sklearn.metrics import f1_score
print("f1 dummy: {:.2f}".format(f1_score(y_test, dc.predict(X_test))))
print("f1 logic: {:.2f}".format(f1_score(y_test, lr.predict(X_test))))

from sklearn.metrics import classification_report

print(classification_report(y_test, lr.predict(X_test), target_names=['noncancerous', 'cancerous']))


# Practice with real life data

# Deep learning

# Reinforcement learning 

# Something else entirely

