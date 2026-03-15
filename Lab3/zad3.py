import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv("iris_big 1.csv")

(train_set, test_set) = train_test_split(df.values, train_size=0.7, test_size=0.3, random_state=13)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# Najbliżsi sąsiedzi

nbrs3 = KNeighborsClassifier(n_neighbors=3)
nbrs5 = KNeighborsClassifier(n_neighbors=5)
nbrs11 = KNeighborsClassifier(n_neighbors=11)
# Klasyfikuje nowe dane, obliczając prawdopodobieństwo ich przynależności do każdej
# z możliwych klas, a następnie wybiera klasę z najwyższym wynikiem
# Zakłada że wszystkie cechy są od siebie niezależne
naiveBayes = GaussianNB()

mlp = MLPClassifier(max_iter=5000, random_state=13)


classifiers = {
    'NN3': nbrs3,
    "NN5": nbrs5,
    "NN11": nbrs11,
    "Naive Bayes": naiveBayes,
    "MLP": mlp 
}

accuracyDic = {}

for name, clf in classifiers.items():
    clf.fit(train_inputs, train_classes)
    
    predictions = clf.predict(train_inputs)
    confMatrix = confusion_matrix(train_classes, predictions)
    
    labels = ["setosa", "versicolor", "virginica"]
    cm_df = pd.DataFrame(confMatrix, index=labels, columns=labels)
    
    accuracy = clf.score(train_inputs, train_classes)
    
    accuracyDic[name] = accuracy
    
    print(f"============ {name} ============")
    print(cm_df)
    print(f"Dokładność modelu: {(accuracy*100):.3f}%")
    print(f"============        ============")


bestAcc = max(accuracyDic, key=accuracyDic.get)
print(f"{bestAcc} - {accuracyDic[bestAcc]*100:.3f}%")