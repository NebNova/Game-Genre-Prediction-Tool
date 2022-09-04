import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("games_final.csv")

genres = {"action": 'action', "adv": 'adventure', "casual": 'casual', "ea": 'early_access',
          "f2p": 'free_to_play', "indie": 'indie', "mm": 'massively_multiplayer', "rpg": 'rpg',
          "racing": 'racing', "sim": 'simulation', "sports": 'sports', "strategy": 'strategy'}

def genre_sgd(genre, age):
    genre_df = get_genre_df(genre)
    X = genre_df.drop('mean_owners', axis = 1)
    y = genre_df['mean_owners']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.23, random_state = 42)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(X_train, Y_train)
    pred_clf = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    return score

def get_genre_df(genre):
    genre_df = data[data[genre] == 1.0]
    return genre_df


for key in genres:
    genre_score = genre_sgd(genres[key], 14)
    genre_score = genre_score * 100
    print(genres[key]," has an accuacry of: ", "{0:.2f}".format(genre_score), "%", sep='')
