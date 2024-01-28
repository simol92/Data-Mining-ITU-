# %%
from sklearn.metrics import silhouette_score
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from utils import get_dfs
(full_df, float_df, holdout) = get_dfs()
full_df.columns

# %% #imports and data loading

# data = pd.read_csv("spotify_songs.csv")


# %%
# First experiment:
# Setting range myself via dictionary by which I determine mood

# selecting relevant columns
columns = ["track_name", "energy", "valence", "danceability"]

# new datafram with selected columns
data = full_df.loc[:, columns]
data

# %% # creating mood dictionaries

# in this mood dict i tried to assign distinct values to each attribute/mood. however, ~19000 tuples were assigned with undefined mood
distinct_moods = {
    'energy boost': {'valence': (0.9, 1.0), 'energy': (0.9, 1.0)},
    'i am happy': {'valence': (0.6, 0.8), 'energy': (0.6, 0.8)},
    'just give me a loving background soundscape': {'valence': (0.5, 0.5), 'energy': (0.5, 0.5)},
    'i am sad and a bit depressed': {'valence': (0.0, 0.4), 'energy': (0.0, 0.4)}
}

# with this mood dict i tried to overlap some ranges. however, this will lead to low distinction between clusters.
# it resulted in ~15500 tuples assigned to undefined mood
overlapping_moods = {
    'energy boost': {'valence': (0.8, 1.0), 'energy': (0.8, 1.0)},
    'i am happy': {'valence': (0.5, 0.8), 'energy': (0.5, 0.8)},
    'just give me a loving background soundscape': {'valence': (0.3, 0.5), 'energy': (0.3, 0.5)},
    'i am sad and a bit depressed': {'valence': (0.0, 0.3), 'energy': (0.0, 0.3)}
}

# %% # defining a function that assigns moods as labels defined by a specific range with assitence in code writing from chat gpt


def assign_mood_label(entry):
    for mood, ranges in distinct_moods.items():
        if all(ranges[feature][0] <= entry[feature] <= ranges[feature][1] for feature in ['valence', 'energy']):
            print(f"Assigned '{mood}' to entry {entry.name}")
            return mood
    print(
        f"Assigned 'undefined mood' to entry {entry.name}: valence={entry['valence']}, energy={entry['energy']}")
    return 'undefined mood'

# %%


def assign_mood_label(entry):
    for mood, ranges in overlapping_moods.items():
        if all(ranges[feature][0] <= entry[feature] <= ranges[feature][1] for feature in ['valence', 'energy']):
            print(f"Assigned '{mood}' to entry {entry.name}")
            return mood
    print(
        f"Assigned 'undefined mood' to entry {entry.name}: valence={entry['valence']}, energy={entry['energy']}")
    return 'undefined mood'


# %% # appending the new column with the moods
data['mood_label'] = data.apply(assign_mood_label, axis=1)

# %%
# checking the distribution of moods
data['mood_label'].value_counts()

# First experiment conclusion:
# Defining clusters myself didn't work since too many tuples falls outside of the range of moods I defined.
# Furthermore, it does not really qualify as a unsupervised method, as it inherently label the tuples based on hard coding the moods.

# %%
# Second experiment:
# Iterate a grid search look alike setup to find best combination of features
# that gives the highest possible silhouette score.
# Applying k-means


# read data again to start from scratch in experiment 2
data = pd.read_csv("spotify_songs.csv")

# features to combine
columns_float = ["energy", "valence", "danceability", "loudness", "tempo"]

# defining new df with the chosen features
df_float = data.loc[:, columns_float]

# defining what my features are
features = columns_float

# create combinations of features up until four units in the combination
all_possible_feature_combinations = list(combinations(features, 4))

# convert combinations to a list of tuples so that the iterator can run over the combinations
all_possible_feature_combinations = [
    tuple(combination) for combination in all_possible_feature_combinations]

# initialize variables to store the best feature combination and silhouette score
best_features = None
best_silhouette_score = -1

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_float)

# specify the range of clusters to consider. i specified the range to be between 2 and 10 clusters.
cluster_range = range(2, 10)

# %%

# iterate through the combinations (grid search)

# iterate over all possible feature combinations. the code was developed with help from chat gpt
for feature_combination in all_possible_feature_combinations:
    print(f"\nEvaluating Feature Combination: {feature_combination}")

    # Subsetting the DataFrame with selected features
    subset_data = df_float[list(feature_combination)].values

    # Iterate over different numbers of clusters
    for num_clusters in cluster_range:
        print(f"\n  Evaluating Number of Clusters: {num_clusters}")

        # Apply K-Means clustering
        kmeansModel = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeansModel.fit_predict(subset_data)

        # Evaluate clustering quality using silhouette score
        silhouette_avg = silhouette_score(subset_data, labels)
        print(f"    Silhouette Score: {silhouette_avg}")

        # Update if a better combination is found
        if silhouette_avg > best_silhouette_score:
            best_silhouette_score = silhouette_avg
            best_features = feature_combination
            best_num_clusters = num_clusters

print("\nBest Feature Combination:", best_features)
print("Best Number of Clusters:", best_num_clusters)
print("Best Silhouette Score:", best_silhouette_score)

# Second experiment conclusion:
# The best feature combination is the four features 'energy', 'valence', 'danceability', and 'tempo'
# with a number of three clusters. It gives a score at 0.62. This is alright, however not great. Therefore we continue with investigating the clusters.

# %%

# After having found the best feature combination and clusters, I make a model based on these findings

#  after having found the best combo i read data and build df again to start from scratch
# data = pd.read_csv("spotify_songs.csv")
columns = ["energy", "valence", "danceability", "tempo"]
df_kmeans = full_df.loc[:, columns]
clusters = 3

# standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_kmeans)

# %%
# apply K-Means
kmeansModel = KMeans(n_clusters=clusters, random_state=42)
df_kmeans['cluster'] = kmeansModel.fit_predict(scaled_data)
# df_samples = StratifiedShuffleSplit(df_kmeans)
# sample = df_samples[0]
# sample
sample = df_kmeans.sample(2500)
# %% # plotting the resulting clusters in 3d for visual inspection

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample['energy'], sample['valence'], sample['danceability'],
           c=sample['cluster'], cmap='viridis', s=50)
# ax.scatter(kmeansModel.cluster_centers_[:, 0], kmeansModel.cluster_centers_[
#    :, 1], kmeansModel.cluster_centers_[:, 2], marker='x', s=200, linewidths=3, color='r')
ax.set_xlabel('energy')
ax.set_ylabel('valence')
ax.set_zlabel('danceability')
plt.show()
# %%
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(sample['valence'], sample['danceability'], sample['energy'],
           c=sample['cluster'], cmap='viridis', s=50, alpha=0.2)
# ax.scatter(kmeansModel.cluster_centers_[:, 0], kmeansModel.cluster_centers_[
#    :, 1], kmeansModel.cluster_centers_[:, 2], marker='x', s=200, linewidths=3, color='r')
ax.set_xlabel('energy')
ax.set_ylabel('valence')
ax.set_zlabel('danceability')
plt.show()
# Visualisation conclusion:
# By visualising the clusters as well as considering the silhouette score, the clustering
# is not really super great. there is not distinct seperation between the data points,
# and thereby not really any distinct clusters.

# Below i will explore a bit further with descriptive statistics


# %%

# using stats to look at the clusters
stats = df_kmeans.groupby("cluster").agg(['mean', 'median', 'std'])
print(stats)

# Below I explore a bit further plotting the cluster features and their mean score

# %%

# plotting on some of the features
# dropping tempo since it skews the plot

columns = ["energy", "valence", "danceability", "cluster"]
energy_valence_danceability = df_kmeans.loc[:, columns]

cluster_means = energy_valence_danceability.groupby('cluster').median()
cluster_means.plot(kind='bar')
plt.title("features")
plt.ylabel("mean")
plt.show()

# %% # joining the data frame with cluster label with data frame with track info

columns_to_join = data[['track_name', 'track_artist']]
joined_df = df_kmeans.join(columns_to_join)
joined_df

# %%
# building the recommender even though the clustering is not working greatly
# map real moods to cluster labels based on my interpretation of the clusters

cluster_rename = {
    0: 'i_need_an_energy_boost',
    1: 'i_feel_rhythmic',
    2: 'i_feel_happy_pop_and_dancy'
}

joined_df['cluster'] = joined_df['cluster'].replace(cluster_rename)

# %%

# build the method
columns_to_final_df = ["cluster", "track_name", "track_artist"]
final_df = joined_df.loc[:, columns_to_final_df]
final_df


def recommend_song(final_df, mood):
    filtered_df = final_df[final_df['cluster'] == mood]

    row = filtered_df.sample().iloc[0]
    song = row['track_name']
    artist = row['track_artist']
    return song, artist

# %%

# testing the method


# enter the mood here
mood = "i_need_an_energy_boost"

# call the method
song, artist = recommend_song(final_df, mood)
print(song, artist)

# %%

# Ideas for furhter studies:
# Write about the experiements, and expreess how feature engineering could have been performed such as:
# Create bins/Discreatization for popularity and tempo
# low-medium-high

# Combine features such as energy * loudness and then square it
# High score indicates high impact. Squaring it will emphasize the differences in the data.

# %%
