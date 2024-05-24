from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')
print(stop_words)
from method.Tweet import Tweet
import pandas as pd
import numpy as np
label_to_sentiment = {0:"Negative", 4:"Positive"}
def mapper(label):
     return label_to_sentiment[label]
if __name__ == '__main__':
    csv_dir = "/Users/priscafehiarisoadama/me/S4/mr_Tsinjo/12_tp_topic_modeling/tweets data/tweets.csv"
    df = pd.read_csv(csv_dir, header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user_name', 'tweet']
    df = df.drop(['id', 'date', 'query', 'user_name'], axis=1)
    df.head()
    Tweet.get_all_Datas(csv_dir)
    df.sentiment = df.sentiment.apply(lambda x: mapper(x))
    import seaborn as sns

    plt.figure(dpi=100)
    sns.countplot(df['sentiment'])