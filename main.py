# Program dependencies
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import streamlit as st
import statistics as stats

# ML algorithm and test-train split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

# load dataset to dataframe
data = pd.read_csv("games_final.csv")

# list variables for graphs
owners_list = []
age_list = []

# Genre array for drop down menus
genre_arr = np.array(['action', 'adventure', 'casual','early_access',
'indie', 'rpg', 'racing', 'simulation', 'sports', 'strategy'])

# Genre dictionary for prediction use
genres = {'action': 2, 'adventure': 3, 'casual': 4, 'early_access': 5,
          'indie': 7, 'rpg': 9,'racing': 10, 'simulation': 11,
          'sports': 12, 'strategy': 13}

# Stochastic Gradient Descent function
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

# Function to create single element dataframe to be used in predictions
def create_pred_df(genre, age):
    pred_dict = {'game_age': 0, 'action': 0, 'adventure': 0, 'casual': 0,
    'early_access': 0, 'free_to_play': 0, 'indie': 0, 'massively_multiplayer': 0,
    'rpg' :0, 'racing': 0, 'simulation': 0, 'sports': 0, 'strategy': 0}
    pred_dict['game_age'] = age
    pred_dict[genre] = 1
    pred_df = pd.DataFrame(pred_dict, index=[0])
    return pred_df

# Creates results stings
def display_results(pred):
  sales_str = ('The average estimated possible sales are: ' + str(round(pred['AvgSales'], 2)))
  acc_str = ('The average prediction accuracy is: ' + str(round(pred['AvgAcc'], 2)) +
  '% with a variance of +/- ' + str(round(pred['AccVar'], 2)) + '%')
  st.write(sales_str)
  st.write(acc_str)
  return None

# Calculate results from input
def calc_sales(g1, g2, g3, game_age):
  if g1 != g2 and g2 != g3 and g3 != g1:
    result1 = genre_sgd(g1, game_age)
    result2 = genre_sgd(g2, game_age)
    result3 = genre_sgd(g3, game_age)
    result_list = [result1, result2, result3]
    for res in result_list:
      res['score'] = res['score'] * 100
      res['sales'] = np.array2string(res['sales'])
      res['sales'] = res['sales'].replace('[', '').replace(']', '').replace('.', '')
      res['sales'] = int(res['sales'])
    score_results = np.array([result1['score'], result2['score'], result3['score']])
    sales_results = np.array([result1['sales'], result2['sales'], result3['sales']])
    score_avg = np.average(score_results)
    score_var = stats.variance(score_results)
    sales_avg = np.average(sales_results)
    pred_results = {'AvgAcc': score_avg, 'AvgSales': sales_avg, 'AccVar': score_var}
    display_results(pred_results)
    return None
  else:
    st.error('Genre selections must be different!', icon="🚨")
    return None


# GUI header and explanation
st.write("""
# Game Genre Sales Prediction Tool

Select three ***different*** genres and a market age to get predicted sales.\n
Keep in mind that sales predictions are only possible sales, not all video games are a hit.
***
""")
# Interactice GUI elements
genre1 = st.selectbox('Select first genre.', genre_arr)
genre2 = st.selectbox('Select second genre.', genre_arr, index=1)
genre3 = st.selectbox('Select third genre.', genre_arr, index=2)
selected_age = st.slider('Select a market age for your game.', 1, 25, 5)
st.markdown("***") # Blank line
st.button('Predict Possible Game Sales', on_click=calc_sales(genre1, genre2, genre3, selected_age))

# Graph section header and description
st.write("""
***
***
***
### Data Descriptions

The below graphs help describe and understand the data that is being used to make predictions.
***
""")

# Histogram
age = data['game_age'] 
fig = plt.figure(figsize=(10, 7))
plt.hist(age, edgecolor='black')
plt.xlabel("Years on market")
plt.ylabel("Games")
plt.title("Games per years on market in dataset")
st.pyplot(fig)

# Create owner and age list for graph use
for key in genres:
  genre_df = data[data[key] == 1.0]
  owners_list.append(genre_df['mean_owners'].mean())
  age_list.append(genre_df['game_age'].mean())

# Convert lists to arrays and arrays to list for graph use
genre_labels = genre_arr.tolist()
owners_arr = np.array(owners_list)
age_arr = np.array(age_list)

# Custom auto percent for pie chart
def autopct(vals):
    def genre_autopct(percent):
        totals = sum(vals)
        pct_vals = int(round(percent*totals/100.0))
        return '{p:.2f}%\n({v:d})'.format(p=percent,v=pct_vals)
    return genre_autopct

# Blank Line
st.markdown("***")

# Pie Chart
fig = plt.figure(figsize=(10, 7))
plt.pie(owners_arr, labels=genre_labels,
autopct=autopct(owners_list), startangle=90)
plt.title("Average owners per genre in dataset")
st.pyplot(fig)

# Blank Line
st.markdown("***")

# Bar Graph
fig = plt.figure(figsize=(10, 7))
plt.bar(genre_arr, age_arr)
plt.xlabel("Genre")
plt.ylabel("Game Age")
plt.title("Average age per genre in dataset")
st.pyplot(fig)

