# %%
import pandas as pd
import numpy as np
import random
from utils import get_dfs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import csv

# %%
CLUSTERS = 25

# Build model for prediction
data_df, audio_features_df, holdout_df = get_dfs()
audio_features_df = audio_features_df.drop(
    columns=["mode", "key", "loudness", "duration_ms", "track_popularity"])
transformer = StandardScaler()
scaled_audio_features = transformer.fit_transform(audio_features_df)
k_means_model = KMeans(init='k-means++', n_clusters=CLUSTERS,
                       random_state=0).fit(scaled_audio_features)
data_df['cluster'] = k_means_model.labels_

# %%
# Define audio-columns
audio_columns = ['danceability', 'energy', 'speechiness',
                 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
holdout_numerical_df = holdout_df[audio_columns]

# Scale audio-columns
transformer = StandardScaler()
holdout_numerical_df = transformer.fit_transform(holdout_numerical_df)

# Replace non-scaled values with scaled values
holdout_numerical_df = pd.DataFrame(
    holdout_numerical_df, columns=audio_columns, index=holdout_df.index)
holdout_df = pd.concat(
    [holdout_df.drop(columns=audio_columns), holdout_numerical_df], axis=1)

# Method for recommending a song based on track_id from holdout_df
def recommend_song(track_id):
    audio_columns = ['danceability', 'energy', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

    # Find song in holdout_df
    song = holdout_df.loc[(holdout_df['track_id'] == track_id)]
    print(
        f"Listening to {song['track_name'].iloc[0]} by artist {song['track_artist'].iloc[0]}")

    # Predict the cluster of the song
    song_audio_features = song[audio_columns]
    predicted_cluster = k_means_model.predict(
        song_audio_features.to_numpy())[0]

    # Find songs in same cluster
    cluster_songs = data_df.loc[(data_df['cluster'] == predicted_cluster) & (
        data_df['track_popularity'].ge(70)) & (data_df['track_id'] != song['track_id'].iloc[0])]
    # Print to evaluate candidate songs
    print("20 Candidate songs for recommendation:")
    print(cluster_songs.head(20)[['track_name', 'track_artist']])
    # pick a random song from reduced df
    recommended_song = cluster_songs.sample()
    # print(recommended_song.iloc[0])
    print(
        f"Recommended song is {recommended_song['track_name'].iloc[0]} by artist {recommended_song['track_artist'].iloc[0]}\n")


for i in range(20):
    sample_song_track_id = holdout_df.sample()['track_id'].iloc[0]
    recommend_song(sample_song_track_id)

# %%

#Method for continuosly supplying track_id to recommend songs. Meant for testing recommender
def start_recommender():
    while True:
        print("Please supply a valid track-id from holdout_df:")
        track_id = input()
        # Find song in df
        song = holdout_df.loc[(holdout_df['track_id'] == track_id)]
        if song.empty:
            print('Song not found. Please try again')
            continue
        recommend_song(track_id=track_id)

#Uncomment and run to start recommender
#start_recommender()

# %%
