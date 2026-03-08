
import pandas as pd
import re
import numpy as np

pd.set_option('display.max_rows', None)  # Pokaż wszystkie wiersze
pd.set_option('display.max_columns', None)  # Pokaż wszystkie kolumny
pd.set_option('display.width', 1000)  # Szerokość terminala
pd.set_option('display.max_colwidth', None)  # Pełna szerokość kolumn tekstowych

fileName = 'iris_big_with_errors.csv'




# =================== a ===================

# Zmiana , na . - 6 kolumn zamiast pięciu
with open(fileName) as f:
    lines = f.readlines()

newLines = []

for line in lines:
    newLine = re.sub(r'([0-9]),([0-9])', r'\1.\2', line.strip())
    newLines.append(newLine)

newLines = [ line.split(',') for line in newLines ]
df = pd.DataFrame(newLines[1:], columns=newLines[0])

# Zmiana nazwy kolumn
new_names = {
    'sepal length (cm)': 'sepal_length',
    'sepal width (cm)': 'sepal_width',
    'petal length (cm)': 'petal_length',
    'petal width (cm)': 'petal_width',
    'target name': 'target_name'
}
df.columns = df.columns.str.strip('"')
df.rename(columns=new_names, inplace=True)





# =================== b ===================

# Zamiana liczb stringów na floaty
for col in df.columns[:4]:
    df[col] = df[col].str.replace('"', '')
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Zamiana nazw kwiatków
# flowers = ['setosa', 'versicolor', 'virginica']

def cleanTargetName(name):
    name = re.sub(r'[^a-z]', '', name.lower())

    if 'setosa' in name: return 'setosa'
    if 'versicolor' in name: return 'versicolor'
    if 'virginica' in name: return 'virginica'

    return np.nan

df['target_name'] = df['target_name'].apply(cleanTargetName)

print('============== Brakujące dane ==============')
print(df.isna().sum())
print('============================================\n')

# c) i d)
# Wyliczenie średniej wielkości i uzupełnienie NaNów
numColumns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
avg = df[numColumns].mean().round(2).to_dict()

for col in numColumns:
    df[col] = df[col].fillna(avg[col])
    
    # Sprawdzenie czy są w zakresie (0; 15) jak nie to średnia
    df[col] = df[col].mask( (df[col] <= 0) | (df[col] > 15), avg[col] )

df['target_name'] = df['target_name'].fillna(df['target_name'].mode()[0])    

print('============== Uzupełnienie danych ==============')
print(f"Średnie: {avg}\n")
print(df.isna().sum())
print('============================================\n')

print(df)
df.to_csv('new_iris_big.csv', index=False, )