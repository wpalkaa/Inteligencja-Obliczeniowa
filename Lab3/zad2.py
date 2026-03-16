import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris_big 1.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# Drzewo decyzyjne

dtc = tree.DecisionTreeClassifier()
# Bierze dane, patrzy na jedną kolumnę i szuka takiej wartości danej, aż
# wszystkim będzie mozna przydzielić tylko jedną klasę (liść drzewa)
# Nastepnie sprawdza dla obu zbiorów kolejną kolumnę i znowu dzieli na 2 aż do liści
dtc.fit(train_inputs, train_classes)
dtcScore = dtc.score(test_inputs, test_classes) # sprawdza poprawność modelu


textTree = tree.export_text(dtc, feature_names=["sl", 'sw', 'pl', 'pw'])
print(textTree)

plt.figure(figsize=(20, 10))
tree.plot_tree(dtc, 
          feature_names=train_inputs, 
          class_names=train_classes, 
          filled=True)
plt.title("Drzewo Decyzyjne - Iris Big")
plt.show()

print(f"Dokładność drzewa: {(dtcScore*100):.3f}%")  


# Macierz błędów
predictions = dtc.predict(test_inputs)
confMatrix = confusion_matrix(test_classes, predictions)

labels = ["setosa", "versicolor", "virginica"]

cm_df = pd.DataFrame(confMatrix, index=labels, columns=labels)


print(cm_df)