from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


data = load_diabetes()

X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])

scaler = Normalizer()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

print("Feature mins before scaling:\n {} ".format(X_train.min(axis=0)))
print("==============================================")
print("Feature mins after scaling:\n {} ".format(X_train_scaled.min(axis=0)))
print("==============================================")

print("Feature maxes before scaling:\n {} ".format(X_train.max(axis=0)))
print("==============================================")
print("Feature maxes after scaling:\n {} ".format(X_train_scaled.max(axis=0)))

X_test_scaled = scaler.transform(X_test)
