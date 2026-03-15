import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("iris_big 1.csv")

# Podział na zbiór testowy (30%) i trenignowy (70%), ziarno losowości = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

# print(test_set.shape[0]) # .shape() - zwraca wymiary obiektu - (wiersze x kolumny)

train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]

# print(test_classes)