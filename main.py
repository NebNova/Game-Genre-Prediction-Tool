import pandas as pd
import seaborn as sns
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("games_final.csv")

genres = {'action': 2, 'adventure': 3, 'casual': 4, 'early_access': 5,
          'indie': 7, 'rpg': 9,'racing': 10, 'simulation': 11,
          'sports': 12, 'strategy': 13}

def genre_sgd(genre, age):
  results = {'score': 0, 'sales': 0}
  genre_df = data[data[genre] == 1.0]
  X = genre_df.drop('mean_owners', axis = 1)
  y = genre_df['mean_owners']
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.23,
  random_state = 42)
  clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)
  clf.fit(X_train, Y_train)
  results['score'] = clf.score(X_test, Y_test)
  results['sales'] = clf.predict(create_pred_df(genre, age))
  return results

def create_pred_df(genre, age):
    pred_dict = {'game_age': 0, 'action': 0, 'adventure': 0, 'casual': 0,
    'early_access': 0, 'free_to_play': 0, 'indie': 0, 'massively_multiplayer': 0,
    'rpg' :0, 'racing': 0, 'simulation': 0, 'sports': 0, 'strategy': 0}
    pred_dict['game_age'] = age
    pred_dict[genre] = 1
    pred_df = pd.DataFrame(pred_dict, index=[0])
    return pred_df


for key in genres:
  age = 14
  result = genre_sgd(key, age)
  result['score'] = result['score'] * 100
  result['sales'] = np.array2string(result['sales'])
  result['sales'] = result['sales'].replace('[', '').replace(']', '').replace('.', '')
  genre = key.replace('_', ' ')
  print("A game of genre: ", str.title(genre), " that has been on the market for ", age,
  " years, has predicted sales of: ", result['sales'],
  " units. With a prediction accuracy of: ",
  "{0:.2f}".format(result['score']), "%.", sep='')
