# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def get_dfs():
    df = pd.read_csv("spotify_songs.csv")

    # Remove duplicate tracks (saving first occurence of each track):
    # All songs: 32833
    # Unique songs: 28356
    unique_df = df.drop_duplicates(subset="track_id", keep="first")
    unique_df = date_to_year(unique_df)
    unique_df = unique_df.dropna()
    # create holdout_df to save for final testing of models
    data_df, test_df = train_test_split(
        unique_df, random_state=42, test_size=0.2)
    # numerical datapoints: 'track_popularity', 'track_album_release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'

    nominal_columns = ['track_name', 'track_id', 'track_artist', 'playlist_subgenre', 'track_album_id',
                       'playlist_genre', 'track_album_name', 'playlist_name', 'playlist_id', 'playlist_genre', 'year']

    data_numerical_df = data_df.drop(nominal_columns, axis=1).copy()

    return data_df, data_numerical_df, test_df


def date_to_year(df):
    # Using try except for reading datatime because we had an issue of the code not working on some computers despite using the same anaconda environment.
    try:
        df['track_album_release_date'] = pd.to_datetime(
            df['track_album_release_date'])
    except:
        df['track_album_release_date'] = pd.to_datetime(
            df['track_album_release_date'], format='mixed')

    df['year'] = df['track_album_release_date'].dt.year
    df = df.drop('track_album_release_date', axis=1).copy()
    return df


def year_to_decade(df):
    df['decade'] = (df['year'] // 10) * 10
    return df.drop('year', axis=1)


def print_confusion_matrix(y_test, ypred, y):
    conf_matrix = confusion_matrix(y_test, ypred)

    conf_matrix_df = pd.DataFrame(
        conf_matrix, index=np.unique(y), columns=np.unique(y))

    # heatmap for confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_df, annot=True, fmt='g',
                cmap='coolwarm', cbar=False)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('confusion matrix')
    plt.show()


def plot_feature_importances(model, X):
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importances = pd.DataFrame(
        feature_importances, feature_names, columns=['importance'])
    plt.figure(figsize=(10, 8))
    importances.plot(kind='bar', legend=False)
    plt.title('feature importances')
    plt.xlabel('features')
    plt.ylabel('importance')
    plt.show()

# %%
