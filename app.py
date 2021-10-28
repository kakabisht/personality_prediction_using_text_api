import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
import pandas as pd

import re
from tqdm import tqdm


from sklearn.model_selection import train_test_split

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('mbti_1.csv')


def clear_text(data):
    data_length = []
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(data.posts):
        sentence = sentence.lower()

        # Remove |||
        sentence = sentence.replace('|||', "")

        # Remove URLs, links etc
        sentence = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', sentence, flags=re.MULTILINE)

        # Remove puntuations
        puncs1 = ['@', '#', '$', '%', '^', '&', '*',
                  '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '|', '\\', '"', "'", ';', ':', '<', '>', '/']
        for punc in puncs1:
            sentence = sentence.replace(punc, '')

        puncs2 = [',', '.', '\n']
        for punc in puncs2:
            sentence = sentence.replace(punc, ' ')

        # Remove extra white spaces
        sentence = re.sub('\s+', ' ', sentence).strip()

        data_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text, data_length


# 2. Create the app object
app = FastAPI()
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000


class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]


@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence


@app.post('/predict')
def predict_per(user_text: str):
    user_input = []
    user_input.append(user_text)

    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=42, stratify=data.type)

    train_data.posts, train_length = clear_text(train_data)

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words='english', tokenizer=Lemmatizer())

    vectorizer.fit(train_data.posts)

    user_input = vectorizer.transform(user_input).toarray()
    target_encoder = LabelEncoder()

    train_target = target_encoder.fit_transform(train_data.type)

    prediction = np.array_str(target_encoder.inverse_transform(classifier.predict(
        user_input)))
    # prediction = (classifier.predict(user_input))
    return{
        'prediction': prediction
    }


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000/docs
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# Run the api uvicorn app:app --reload
