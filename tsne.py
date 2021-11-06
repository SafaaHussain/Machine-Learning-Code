from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = load_digits()

# fig, axes = plt.subplots(2, 5, subplot_kw = {"xticks": (), "yticks": ()})

# for ax, img in zip(axes.ravel(), digits.images):
#     ax.imshow(img)

# plt.show()


colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]


digits_tsne = TSNE().fit_transform(digits.data)

plt.figure(figsize=(10, 10))
plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max()+1)
plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max()+1)

for i in range(len(digits.data)):
    #            x                   y                   digit                 color
    plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]), color=colors[digits.target[i]])

plt.xlabel("t-SNE Feature 0")
plt.ylabel("t-SNE Feature 1")
plt.show()
