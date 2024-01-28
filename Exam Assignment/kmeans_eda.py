# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from utils import get_dfs

data_df, numerical_df, holdout_df = get_dfs()
X = numerical_df
# K-MEANS
# sum of squared distances
ssd = []

for k in range(2, 15):
    model = KMeans(n_clusters=k)
    model.fit(X)
    # add the models sum of squared distances (inertia_) to ssd list to be able to do elbow plot
    ssd.append(model.inertia_)

# plot ssd
plt.plot(range(2, 15), ssd, 'o--')
plt.xlabel("k value")
plt.ylabel("sum of squared distances")
# %%
# data_df['playlist_genre'].unique()
# k=6 virker som et meget godt bud, især siden det er det antal genrer der allerede er defineret
model = KMeans(n_clusters=6)
data_df['cluster'] = model.fit_predict(X)
data_df.head()

sns.countplot(data_df, x=data_df['cluster'])
# %%
sns.barplot(data_df, x='cluster', y='track_popularity')
# %%
sns.countplot(data_df, x=data_df['playlist_genre'])
# %%
sns.countplot(data=data_df, x='playlist_subgenre')
plt.xticks(rotation=45, ha='right')
plt.show()
# %%

ax = sns.countplot(data_df, x='playlist_genre', hue='cluster')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
# %%
ax = sns.countplot(data_df, x='cluster', hue='playlist_genre')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()
# %%
cross_tab = pd.crosstab(data_df['playlist_genre'], data_df['cluster'])
sns.heatmap(cross_tab, cmap='Blues')
# %%
cross_tab
# %%
# EDA
# find ud af hvad der kendetegner hver genre og cluster
feature_stats = {}
for feature in X.keys():
    genre_stats = {}
    cluster_stats = {}
    for genre in data_df['playlist_genre'].unique():
        genre_stats[genre] = data_df[feature][data_df['playlist_genre']
                                              == genre].mean()

    for cluster in data_df['cluster'].unique():
        cluster_stats[cluster] = data_df[feature][data_df['cluster']
                                                  == cluster].mean()
    feature_stats[f'{feature}_genre'] = genre_stats
    feature_stats[f'{feature}_cluster'] = cluster_stats

# %%
for feature_key, stats in feature_stats.items():
    if 'genre' in feature_key:
        plt.bar(stats.keys(), stats.values())
        plt.title(f'mean values for {feature_key}')
        plt.xlabel('genre')
        plt.ylabel(feature_key)
        plt.show()

# %%
for feature_key, stats in feature_stats.items():
    if 'cluster' in feature_key:
        plt.bar(stats.keys(), stats.values())
        plt.title(f'mean values for {feature_key}')
        plt.xlabel('cluster')
        plt.ylabel(feature_key)
        plt.show()
# %%
