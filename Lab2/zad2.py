from sklearn import datasets 
from sklearn.decomposition import PCA 
import pandas as pd 
import matplotlib.pyplot as plt


iris = datasets.load_iris(as_frame=True)
# CSV'ka kwaitków
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
# Series z wystąpowania kwiatków
y = pd.Series(iris.target, name='FlowerType') 
print(X.head()) 
 
# Zaczynamy od 4, żeby zachować >95% wariancji można usunąć do 2 kolumn
# bo 0.924 + 0.053 > 0.95
pca_iris = PCA(n_components=2).fit(iris.data) 
print("pca_iris: ", pca_iris) 

# [0.92461872 0.05306648 0.01710261 0.00521218]
# wypisuje współczynniki wariancji - kolumna zachowałą x informacji
print("variance_ratio: ", pca_iris.explained_variance_ratio_)

# n wektórów, każdy z nich ma wagi mówiące ile dana kolumna ma wpływu na wynik
print("pca components: ", pca_iris.components_) 
print("pca_iris transform: ", pca_iris.transform(iris.data))


# Wykres

n = 2

if n == 2:
    X_reduced = PCA(n_components=n).fit_transform(iris.data)

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_reduced[:,0], X_reduced[:,1], c=y)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA Iris gdzie n = 2')
    plt.legend(
        scatter.legend_elements()[0],
        iris.target_names.tolist())
    plt.grid(True)
    plt.show()

if n == 3: 
    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

    X_reduced = PCA(n_components=n).fit_transform(iris.data)
    scatter = ax.scatter(
        X_reduced[:, 0],
        X_reduced[:, 1],
        X_reduced[:, 2],
        c=iris.target,
        s=40,
    )

    ax.set(title="PCA Iris gdzie n = 3")
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    # Add a legend
    legend1 = ax.legend(
        scatter.legend_elements()[0],
        iris.target_names.tolist(),
        loc="upper right"
    )
    ax.add_artist(legend1)

    plt.show()