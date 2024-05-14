import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Fungsi untuk melakukan tokenisasi dan penghapusan stopwords
def preprocess_text(text):
  # Tokenisasi
  tokens = word_tokenize(text.lower())

  # Penghapusan stopwords
  stop_words = set(stopwords.words('indonesian'))
  filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

  return filtered_tokens

words = preprocess_text('Apa definisi dari istilah MSIB?')

print(words)

# Import modul yang diperlukan

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')

# Inisialisasi stemmer dan lemmatizer
stemmer = StemmerFactory().create_stemmer()
lemmatizer = WordNetLemmatizer()

# Stemming (menggunakan Sastrawi) dan Lemmatisasi (WordNetLemmatizer)
processed_tokens = [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]

print(processed_tokens)

# Gabungkan kata-kata yang telah diolah
preprocessed_text = ' '.join(processed_tokens)

from sklearn.feature_extraction.text import CountVectorizer

# Contoh data teks yang telah di-stemming dan di-lemmatisasi
processed_texts = [
    "fasilitas tersedia hotel eleven",
    "apakah fasilitas tersedia hotel eleven",
    "berapa fasilitas hotel eleven",
    "hotel eleven menyediakan fasilitas"
]

# Inisialisasi objek CountVectorizer untuk melakukan pendekatan Bag of Words
vectorizer = CountVectorizer()

# Melakukan transformasi teks menjadi representasi Bag of Words
X = vectorizer.fit_transform(processed_texts)

# Mendapatkan fitur kata yang digunakan dalam Bag of Words
feature_names = vectorizer.get_feature_names_out()

# Menampilkan representasi Bag of Words dalam bentuk DataFrame
import pandas as pd
df = pd.DataFrame(X.toarray(), columns=feature_names)
# Mengubah nama baris menjadi 'Pertanyaan 1', 'Pertanyaan 2', dst
df.index = ['Pertanyaan {}'.format(i+1) for i in range(len(processed_texts))]

print(df)

# Load dataset
df = pd.read_csv('training_info.csv')

# Preprocess dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

tfidf_vectorizer = TfidfVectorizer()
x = tfidf_vectorizer.fit_transform(df['Pertanyaan'])
y = df['Jawaban']

# Train a classifier
classifier = MultinomialNB()
classifier.fit(x, y)

# Function to get answer
def get_answer(user_input):
  user_input_tfidf = tfidf_vectorizer.transform([user_input])
  answer = classifier.predict(user_input_tfidf)[0]

  return answer

# Test Chatbot
user_input = "Apa fasilitas yang tersedia di Hotel Egma?"
answer = get_answer(user_input)
print("Jawaban: ", answer)

from sklearn.model_selection import train_test_split

# Pembagian data menjadi data pelatihan dan data validasi
X_train, X_validasi, y_train, y_validasi = train_test_split(x, y, test_size=0.2, random_state=42)

# Prediksi pada data validasi
prediksi_validasi = classifier.predict(X_validasi)

# Menghitung jumlah data yang benar
jumlah_benar = (y_validasi == prediksi_validasi).sum()

# Menghitung total jumlah data
total_data = len(y_validasi)

# Menampilkan jumlah data yang benar dan total jumlah data
print('Jumlah data yang benar: ', jumlah_benar)
print('Total data: ', total_data)