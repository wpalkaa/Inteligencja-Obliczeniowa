import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

with open('./article.txt', 'r') as f:
    text = f.read().lower()

# b)
tokens = nltk.word_tokenize(text)

# print(tokens)

# c) i d)
stop_words = stopwords.words('english')
stop_words.extend([',', '.', '/', '`', ';', ':', '\'', '\"', '`', '[', ']', '``', "''", '\'s', '-', '(', ')'])

cleaned_tokens = [t for t in tokens if t not in stop_words]

print(cleaned_tokens)

print("cleaned: ", len(tokens), '=>', len(cleaned_tokens))


# e)
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(w) for w in cleaned_tokens]
print("lem: ", len(lemmatized_tokens)) # zmienia chyba us na u   ???

# f)
word_counts = Counter(lemmatized_tokens)
common_words = word_counts.most_common(10)

words, counts = zip(*common_words)


plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='skyblue')
plt.title('10 najczęściej występujących słów')
plt.xlabel('Słowa')
plt.ylabel('Liczba wystąpień')
plt.savefig('./zad1HIST.png')

# g)
processed_text = ' '.join(lemmatized_tokens)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(processed_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('./zad1MAP.png')