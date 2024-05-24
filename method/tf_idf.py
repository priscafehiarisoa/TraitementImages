# utilities
import re
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score




# Defining dictionary containing all emojis with their meaning



def preprocess(textdata):
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordlemm = WordNetLemmatizer()
    snowstem = SnowballStemmer("english")

    for tweet in textdata:
        tweet = tweet.lower()

        # Replace all URls with 'URL'
        pattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
        tweet = re.sub(pattern, ' URL', tweet)

        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])

        # Replace @USERNAME to 'USER'.
        tweet = re.sub('@[^\s]+', ' USER', tweet)

        # Replace all non alphabets.
        tweet = re.sub("[^a-zA-Z0-9]", " ", tweet)

        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(r"(.)\1\1+", r"\1\1", tweet)

        tweetwords = ''

        for word in tweet.split():

            # Checking if the word is a stopword.
            if word not in stopwordlist:

                if len(word) > 1:
                    # Lemmatizing the word.
                    word = wordlemm.lemmatize(word)

                    # Stemming the word.
                    word = snowstem.stem(word)

                    tweetwords += (word + ' ')

        processedText.append(tweetwords)

    return processedText

# read csv and add column name


if __name__ == '__main__':
    emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', '>-)':
        'evilgrin', ':(': 'sad', ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry',
              ':-O': 'surprised', ':-*': 'kissing', ':-@': 'shocked', ':-$': 'confused',
              ':-\\': 'annoyed', ':-#': 'mute', '(((H)))': 'hugs', ':-X': 'kissing',
              '`:-)': 'smile', ':^)': 'smile', ':-&': 'confused', '<:-)': 'smile',
              ':->': 'smile', '(-}{-)': 'kissing', ':-Q': 'smoking', '$_$': 'greedy',
              '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':*)': 'smile',
              ':@': 'shocked', ':-0': 'yell', ':-----)': 'liar', '%-(': 'confused',
              '(:I': 'egghead', '|-O': 'yawning', ':@)': 'smile', 'O.o': 'confused',
              '<(-_-)>': 'robot', 'd[-_-]b': 'dj', '~:0': 'baby', '-@--@-': 'eyeglass',
              ":'-)": 'sadsmile', '{:-)': 'smile', ';)': 'wink', ';-)': 'wink',
              'O:-)': 'angel', 'O*-)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'}

    ## Defining set containing all stopwords in english.
    stopwordlist = set(stopwords.words('english'))



    csv_dir = "/Users/priscafehiarisoadama/me/S4/mr_Tsinjo/12_tp_topic_modeling/tweets data/tweets.csv"
    df = pd.read_csv(csv_dir, header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user_name', 'tweet']
    df = df.drop(['id', 'date', 'query', 'user_name'], axis=1)
    df.tweet = df.tweet.apply(lambda x: preprocess(x))
    print(df.tail())