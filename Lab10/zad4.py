import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

df_raw = pd.read_csv("titanic.csv")

df = pd.DataFrame()
df['Class'] = df_raw['Class']
df['Sex'] = df_raw['Sex'].str.capitalize()
df['Age'] = df_raw['Age']
df['Survived'] = df_raw['Survived']
# Kodowanie One-Hot (wymagane przez bibliotekę mlxtend) - robi każdą możliwą wartość na wektor [1rd class, 2nd class..., male, female] => [1, 0, 0, 0, 1, 1, 0]
df_encoded = pd.get_dummies(df)

# patrzy jakie grupy cech występują najczęściej razem
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)

# Wyszukanie reguł
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

rules = rules.sort_values(by='confidence', ascending=False)

# usunięcie niepotrzebnych reguł, chemy tylko czy zyje czy nie
rules_survived = rules[
    (rules['consequents'] == frozenset({'Survived_Yes'})) | 
    (rules['consequents'] == frozenset({'Survived_No'}))
]

print(rules_survived[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# support - czy ważne statystycznie (ile danych stanowi ten przypadek)
# confidence - jak często B gdy A

top_rules = rules_survived.head(10).copy()

# zamiana frozenset na czytelny string
top_rules['rule'] = top_rules['antecedents'].astype(str) + " → " + top_rules['consequents'].astype(str)

plt.figure(figsize=(10, 6))
plt.barh(top_rules['rule'], top_rules['confidence'])
plt.xlabel("Confidence")
plt.title("Reguły")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



plt.figure(figsize=(8, 6))
plt.scatter(rules_survived['support'], rules_survived['confidence'])
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support x Confidence")
plt.tight_layout()
plt.show()