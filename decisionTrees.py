from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt

data = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

print("Train score: {:.2f}".format(tree.score(X_train, y_train)))
print("Test score : {:.2f}".format(tree.score(X_test, y_test)))

export_graphviz(tree, out_file="tree.dot", class_names=["malignant", 'benign'], feature_names=data['feature_names'])


def plot_feature_importances(model, data):
    num_features = data['data'].shape[1]

    plt.barh(range(num_features), model.feature_importances_)

    plt.xlabel("Feature importances")
    plt.ylabel("Feature")

    plt.yticks(range(num_features), data['feature_names'])


plot_feature_importances(tree, data)
plt.title("Feature importances")
plt.show()






