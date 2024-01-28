# %%
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_df = pd.read_csv("spotify_songs.csv")

print(list(data_df.keys()))

#Ordinal datapoints: 'track_popularity', 'track_album_release_date', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'

# Checking for extreme values/malformed data entries. 

nominal_columns = ['track_popularity', 'track_name','track_id','track_artist','playlist_subgenre', 'track_album_id','playlist_genre','track_album_name','playlist_name','playlist_id', 'playlist_genre','track_album_release_date']
ordinal_columns = [ x for x in list(data_df.keys()) if x not in nominal_columns]

spotify_df = data_df.drop(['track_popularity', 'track_name','track_id','track_artist','playlist_subgenre', 'track_album_id','track_album_name','playlist_name','playlist_id','track_album_release_date','playlist_genre'], axis=1)

spotify_df.head()
spotify_df.shape

# %%
################
spotify_df_labels = data_df['playlist_genre']
spotify_df_labels.shape
################## 
#using numpy to reshape into two dimensional data before concatting
labels = spotify_df_labels.values.reshape(-1, 1)

#concatenating with original df
final_spotify_df = np.concatenate([spotify_df,labels], axis=1)
final_spotify_df.shape
final_df = pd.DataFrame(final_spotify_df)
#without column names
final_df.head()

features = np.array(spotify_df.columns)
features_labels = np.append(features,'label')
final_df.columns = features_labels

final_df.head()
# %%
from sklearn.preprocessing import StandardScaler

allCols = final_df.loc[:, features].values
allCols = StandardScaler().fit_transform(allCols) #normalize all features
allCols.shape
#checking mean and standard deviation
np.mean(allCols), np.std(allCols)

#converting into tabular format: 
feat_cols = ['feature '+str(i) for i in range(allCols.shape[1])]
normalized_spotify_df = pd.DataFrame(allCols,columns=feat_cols)
print(normalized_spotify_df)
normalized_spotify_df.shape
# %%
#projecting the 12-dimensional data into 2-dimensional principal components
from sklearn.decomposition import PCA
pca_spotify = PCA(n_components=5)
principalComponents = pca_spotify.fit_transform(allCols)

#creating datafame
number_of_components = pca_spotify.n_components_
component_names = [f"principal component: {n+1}" for n in range(number_of_components)]
PC_spotify_df = pd.DataFrame(data = principalComponents, columns= component_names)
print(PC_spotify_df)

print("Each principal component represents a percentage of total variation captured from the data")
print('Explained variation(information) per principal component: {}'.format(pca_spotify.explained_variance_ratio_))

print('Explained variation(information) of the entire subspace by taking the sum of all PCs: {}'.format(sum(pca_spotify.explained_variance_ratio_)))

# %%

# %%
