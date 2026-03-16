import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns

df = pd.read_csv("diagnosis.csv")


fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection="3d")

healthy = df[df['diagnosis'] == 0]
sick = df[df['diagnosis'] == 1]


ax.scatter(healthy['param1'], healthy['param2'], healthy['param3'], c='blue', label='zdrowi')
ax.scatter(sick['param1'], sick['param2'], sick['param3'], c='red', label='chorzy')

ax.set_xlabel('Parametr 1')
ax.set_ylabel('Parametr 2')
ax.set_zlabel('Parametr 3')
ax.set_title('Wykres 3D')
plt.legend()
# plt.show()


# Accuracy i inne

(train_set, test_set) = train_test_split(df.values, train_size=0.7, test_size=0.3, random_state=13)

train_inputs = train_set[:, 0:3]
train_classes = train_set[:, 3]
test_inputs = test_set[:, 0:3]
test_classes = test_set[:, 3]


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
    
    labels = ["zdrowi", "chorzy"]
    cm_df = pd.DataFrame(confMatrix, index=labels, columns=labels)
    
    # ile identyfikacji jest poprawnych
    accuracy = clf.score(train_inputs, train_classes)
    # sprawdza wiarygosność modelu => TP / TP + FP
    # FP - identyfuje chory, ale zdrowy
    precision = precision_score(train_classes, predictions)
    # stosunek liczby trafnie sklasyfikowanych próbek dodatnich 
    # do wszystkich próbek, które powinny być dodatnie
    # TP / TP + FN
    # FN - identyfikuje zdrowy, ale chory
    rec = recall_score(train_classes, predictions)
    
    accuracyDic[name] = accuracy

    plt.figure()
    sns.heatmap(confMatrix, annot=True, fmt="d",
                xticklabels=["zdrowi","chorzy"],
                yticklabels=["zdrowi","chorzy"])

    plt.title(name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


# Nie klasyfikowanie zdrowych jako chorych (FP) => precision
#                    chorych jako zdrowych (FN) => recall

# Nie jest, np. 9 zdrowych, 1 chory, zidentyfikowano tylko zdrowych,
# poprawność "90%".