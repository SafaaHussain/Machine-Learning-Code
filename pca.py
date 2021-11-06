from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import mglearn
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
# data = load_breast_cancer()

# s = StandardScaler()
# s.fit(data['data'])
# X_scaled = s.transform(data["data"])


# fig, axes = plt.subplots(15, 2)
# m = data['data'][data['target'] == 0]
# b = data['data'][data['target'] == 1]

# axis = axes.ravel()
#
# for i in range(30):
#     _, bins = np.histogram(data['data'][:, i])
#     axis[i].hist(m[:, i], bins=bins, alpha=0.5)
#     axis[i].hist(b[:, i], bins=bins, alpha=0.5)
#     axis[i].set_title(data['feature_names'][i])
#     axis[i].set_yticks(())
# axis[0].legend(["Malinant", "Benign"], loc="best")
# plt.show()

# pca = PCA(n_components=2)
# pca.fit(X_scaled)
# X = pca.transform(X_scaled)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], data['target'])
# plt.show()

# print(pca.components_)

# plt.matshow(pca.components_, cmap="viridis")
# plt.yticks([0, 1], ["first Component", "Second component"])
# plt.colorbar()
# plt.xticks(range(len(data["feature_names"])), data["feature_names"], rotation=60)
# plt.xlabel("Feature")
# plt.ylabel("Components")
# plt.show()

people = fetch_lfw_people(min_faces_per_person=10, resize=0.7)

# print(people.images[0].shape)
#
# fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={"xticks":(), "yticks": ()})
#
# for target, img, axis in zip(people.target, people.images, axes.ravel()):
#     axis.imshow(img)
#     axis.set_title(people.target_names[target])
#
# plt.show()

print("shape:", people.images.shape)
print("Num people:", len(people.target_names))

# Count how often each person appears.
counts = np.bincount(people.target)

for i, (count, name) in enumerate(zip(counts, people.target_names)):
    print("{0:25} {1:3}".format(name, count), end='      ')
    if (i + 1) % 3 == 0:
        print()


mask = np.zeros(people.target.shape, dtype=np.bool)
for target in np.unique(people.target):
    mask[np.where(people.target == target)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.data[mask]


