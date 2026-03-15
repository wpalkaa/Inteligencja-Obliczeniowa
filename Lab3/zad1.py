import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=np.inf)  # wyświetli wszystkie elementy

df = pd.read_csv("iris_big 1.csv")

# Podział na zbiór testowy (30%) i trenignowy (70%), ziarno losowości = 13
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=13)

# print(test_set.shape[0]) # .shape() - zwraca wymiary obiektu - (wiersze x kolumny)

# train_inputs = train_set[:, 0:4]
# train_classes = train_set[:, 4]
# test_inputs = test_set[:, 0:4]
# test_classes = test_set[:, 4]



# Wyliczanie MIN MAX cech kwiatków

features = ["sl", "sw", "pl", "pw"]

train_set_setosa_only = train_set[train_set[:, -1] == "setosa"]
train_set_versicolor_only = train_set[train_set[:, -1] == "versicolor"]
train_set_virginica_only = train_set[train_set[:, -1] == "virginica"]

setosa_ranges = {}
versicolor_ranges = {}
virginica_ranges = {}
for i in range(len(features)): 
    setosa_ranges[features[i]] = ( train_set_setosa_only[:, i].min(), train_set_setosa_only[:, i].max() )
    versicolor_ranges[features[i]] = ( train_set_versicolor_only[:, i].min(), train_set_versicolor_only[:, i].max() )
    virginica_ranges[features[i]] = ( train_set_virginica_only[:, i].min(), train_set_virginica_only[:, i].max() )

print("setosa: ", setosa_ranges, "\nversiolor: ", versicolor_ranges, "\nvirginica: ", virginica_ranges)



# Identyfikacja

def calc_treshold(rangeMaxA, rangeMinB, feature):
    maxA = rangeMaxA[feature][1]
    minB = rangeMinB[feature][0]
    
    return (maxA + minB) / 2


def classify_iris(sl, sw, pl, pw): 
    if pw <= calc_treshold(setosa_ranges, versicolor_ranges, 'pw'): # setosaMax + coś tam
        return "setosa" 
    elif pw >= calc_treshold(versicolor_ranges, virginica_ranges, 'pw'):  # virginicaMin + coś tam
        return "virginica" 
    else: 
        return "versicolor"
good_predictions = 0
lines = test_set.shape[0]

for i in range(lines):
    sl, sw, pl, pw, tg = test_set[i]
    if classify_iris(sl, sw, pl, pw) == test_set[i][4]:
        good_predictions = good_predictions + 1

print(good_predictions, f"({good_predictions/lines*100}%)")

# train_set_sorted = train_set[train_set[:,-1].argsort()]
# print(train_set_sorted)

# Wyliczenie min, max wartości dla danych kwiatków
