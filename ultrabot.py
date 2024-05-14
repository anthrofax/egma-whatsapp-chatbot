import json
import requests
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df = pd.read_csv("training_info.csv")

# Preprocess dataset
tfidf_vectorizer = TfidfVectorizer()
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
