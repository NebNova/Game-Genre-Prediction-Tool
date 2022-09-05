import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("games_final.csv")

genres = {'action': 2, 'adventure': 3, 'casual': 4, 'early_access': 5,
          'free_to_play': 6, 'indie': 7, 'massively_multiplayer': 8, 'rpg': 9,
          'racing': 10, 'simulation': 11, 'sports': 12, 'strategy': 13}

def genre_sgd(genre, age):
    genre_df = data[data[genre] == 1.0]
    X = genre_df.drop('mean_owners', axis = 1)
    y = genre_df['mean_owners']
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.23,
    random_state = 42)
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
    clf.fit(X_train, Y_train)
    pred_clf = clf.predict(X_test)
    score = clf.score(X_test, Y_test)
    return score

for key in genres:
    genre_score = genre_sgd(key, 14)
    genre_score = genre_score * 100
    print(key," has an accuacry of: ", "{0:.2f}".format(genre_score),
    "%", sep='')
