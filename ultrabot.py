import json
import requests
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


# Pra-pemrosesan data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("indonesian"))
    filtered_tokens = [
        word for word in tokens if word.isalnum() and word not in stop_words
    ]
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    return " ".join(stemmed_tokens)


# Load dataset
df = pd.read_csv("training_info.csv")

# Preprocess dataset
tfidf_vectorizer = TfidfVectorizer()
# stemmer = StemmerFactory().create_stemmer()
# lemmatizer = WordNetLemmatizer()
# vectorizer = CountVectorizer()

x = tfidf_vectorizer.fit_transform(df["Pertanyaan"])
y = df["Jawaban"]

# Train a classifier
classifier = MultinomialNB()
classifier.fit(x, y)



class ultraChatBot:
    def __init__(self, json):
        self.json = json
        self.dict_messages = json["data"]
        self.ultraAPIUrl = "https://api.ultramsg.com/instance85526/"
        self.token = "2i2gb9eble75v11d"

    def send_requests(self, type, data):
        url = f"{self.ultraAPIUrl}{type}?token={self.token}"
        headers = {"Content-type": "application/json"}
        answer = requests.post(url, data=json.dumps(data), headers=headers)
        return answer.json()

    def send_message(self, chatID, text):
        data = {"to": chatID, "body": text}
        answer = self.send_requests("messages/chat", data)
        return answer

    def predict_answer(self, user_input):
        # # Pre-process tahap awal
        # preprocessed_message = preprocess_text(user_input)

        # # Stemming & Lemmatize
        # processed_tokens = [
        #     lemmatizer.lemmatize(stemmer.stem(word)) for word in preprocessed_message
        # ]

        # preprocessed_text = " ".join(processed_tokens)

        user_input_tfidf = tfidf_vectorizer.transform([user_input])

        answer = classifier.predict(user_input_tfidf)[0]

        return answer

    def Processingـincomingـmessages(self):
        if self.dict_messages != []:
            message = self.dict_messages
            # text = message["body"]
            text = message["body"]

            if not message["fromMe"]:
                chatID = message["from"]
                answer = self.predict_answer(text)
                return self.send_message(chatID, answer)
            else:
                return "NoCommand"
