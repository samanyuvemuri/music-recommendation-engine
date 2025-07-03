import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

# helper functions to get the song index
def get_index_from_name(track_name):
   matches = df[df['track_name'] == track_name]
   if matches.empty:
         print(f"Song '{track_name}' not found.")
         sys.exit()
   return matches.index[0]
def get_name_from_index(idx):
   return df.loc[idx, 'track_name']

# read dataset csv file
df = pd.read_csv('dataset.csv')

# features that model takes into account
features = ['artists', 'track_genre', 'popularity', 'tempo', 'valence']


# gets rid of the NaN values and converts the features to string
for feature in features:
    df[feature] = df[feature].astype(str).fillna('')

# combines the features into a single string for each row
def combine_features(row):
    return row['artists'] + ' ' + row['track_genre'] + ' ' + row['popularity'] + ' ' + row['tempo'] + ' ' + row['valence']

# creates a new column 'combined_features' that contains the combined features
df['combined_features'] = df.apply(combine_features, axis=1)

# creates a count matrix from the 'combined_features' column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])

# ask the user for a song they like and obtain the index of that song
song_user_likes = input("Enter a song you like: ")
song_index = get_index_from_name(song_user_likes)

# creates a matrix of cosine similarities between the song the user likes and all other songs
song_vector = count_matrix[song_index]
similarities = cosine_similarity(song_vector, count_matrix).flatten()
similar_songs = list(enumerate(similarities))
sorted_similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)


song_count = 0
print(f"Songs similar to '{song_user_likes}':")
for song in sorted_similar_songs[1:11]:  # skip the first song as it is the same song
   print(get_name_from_index(song[0]))
   song_count += 1
   if song_count == 10:
       break