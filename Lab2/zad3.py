from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import pandas as pd 
import matplotlib.pyplot as plt


iris = datasets.load_iris(as_frame=True)
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType') 

data = X[["sepal length (cm)", 'sepal width (cm)']]

print("min: \n", data.min())
print("\nmax: \n", data.max())
print('\nmean: \n', data.mean())
print("\nstd. deviation: \n", data.std())


# skaluje dane tak, żeby wszystkie były w przedziale <0,1>
# wartość min staje się 0, max 1
minmax = MinMaxScaler()
dataMinMax = pd.DataFrame(minmax.fit_transform(data), columns=data.columns)

# śrędnią z danych ustawia jako punkt 0
# Dla każdej wartości wylicza stosunek wybranej danej do średniej
# w skali odchylenia std. 
# x - średnia / odchylenie std.
zScore = StandardScaler()
dataZScore = pd.DataFrame(zScore.fit_transform(data), columns=data.columns)


fig, axes = plt.subplots(1, 3, figsize=(18,5))
titles = ['Original Dataset', 'Z-Core Scaled Dataset', 'Min-Max Normalised Dataset']
datas = [ data, dataMinMax, dataZScore]

i = 0
for ax in axes:
    scatter = ax.scatter(datas[i].iloc[:,0], datas[i].iloc[:,1], c=y)
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title(titles[i])
    ax.legend(
        scatter.legend_elements()[0],
        iris.target_names.tolist())
    ax.grid(True)

    i += 1

plt.show()
print(zScore)