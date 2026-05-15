import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import text2emotion as te
from transformers import pipeline

nltk.download('vader_lexicon')

# a)

pos_rev = "Studio rooms were very nice and spacious. We were celebrating a birthday and were given a complimentary upgrade which was great! Will definitely visit again :)"
neg_rev = "Depressing for Central London. So we booked the same day as staying based on including breakfast (at the restaurant that had good reviews). On arrival we were told the restaurant had gone into administration. We were still expected to pay full price for the stay. Additionally our room was a double, but really only suitable for one person. It had mismatched furniture, only one chair, and no room for luggage store so had to leave bags out. Without catering facilities we had to find cafes, which there are not a great selection close by (Prets or an English Greasy spoon on the council estate being the best options). I asked about cutlery to eat and was said that I could upgrade our room to have a kitchen. This was a bit of an insult given the reason we were having to sort food was that the hotel no longer had facilities. The room overall was also not in a great state of repair. We found it upsetting given for the same price we could have stayed elsewhere with full facilities, and would have done if we had been told on booking there was no catering. On check-out the hotel arranged a refund on the breakfast, but made no consideration for the inconvenience and impact on our holiday. Overall very depressing experience for the start of new year in London."

reviews = {
    "Positive": pos_rev,
    "Negative": neg_rev
}

# b)
print('vader')
sia = SentimentIntensityAnalyzer()
for k, review in reviews.items():
    scores = sia.polarity_scores(review)
    print(f"{k}: {scores}")

# c)
# TextBlob: Wynik to Polarity (od -1 do 1) i Subjectivity (od 0 do 1)
print('textblob:')
for k, review in reviews.items():
    blob = TextBlob(review)
    print(f"{k}: {blob.sentiment.polarity:.4f}")

# text2emotion: Zwraca rozkład podstawowych emocji (Happy, Angry, Surprise, Sad, Fear) w skali 0–1
print('text2emotion')
for k, review in reviews.items():
    emotions = te.get_emotion(review)
    print(f"{k}: {emotions}")

# HuggingFace: Zwraca konkretne emocje (np. joy, anger, sadness)
print('huggingface')
classifier = pipeline("sentiment-analysis", model="j-hartmann/emotion-english-distilroberta-base")
for k, review in reviews.items():
    result = classifier(review)
    print(f"{k}: {result}")
