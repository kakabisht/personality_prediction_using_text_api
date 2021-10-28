from flask import Flask, request, jsonify
from flask import jsonify
import uvicorn

import numpy as np
import pickle
import pandas as pd

import re
from tqdm import tqdm


import nltk
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# -*- coding: utf-8 -*-

app = Flask(__name__)

df = pd.read_csv('mbti_1.csv')
pickle_in = open("model.pkl", "rb")
classifier = pickle.load(pickle_in)
nltk.download('stopwords')


def clear_text(df):
    df_length = []
    lemmatizer = WordNetLemmatizer()
    cleaned_text = []
    for sentence in tqdm(df.posts):
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

        df_length.append(len(sentence.split()))
        cleaned_text.append(sentence)
    return cleaned_text, df_length


class Lemmatizer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, sentence):
        return [self.lemmatizer.lemmatize(word) for word in sentence.split() if len(word) > 2]


@app.route("/", methods=['POST'])
def helloWorld():
    data = request.json
    # print(data[0])
    # dict_data=dict(data[0])
    # print(dict_data)

    train_data, test_data = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.type)

    train_data.posts, train_length = clear_text(train_data)

    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words='english', tokenizer=Lemmatizer())

    vectorizer.fit(train_data.posts)

    data[0]['data'] = [data[0]['data']]

    data[0]['data'] = vectorizer.transform(data[0]['data']).toarray()
    target_encoder = LabelEncoder()

    train_target = target_encoder.fit_transform(train_data.type)

    prediction = np.array_str(target_encoder.inverse_transform(classifier.predict(
        data[0]['data'])))

    return jsonify({"data": prediction})


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
