# %% #imports and data loading
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv("spotify_songs.csv")

# %%
data.columns
data.dropna(inplace=True)
data.drop_duplicates(subset="track_id", keep="first", inplace=True)
# %%
data
# %%
# choosing numerical columns

columns =  'track_popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
df = data.loc[:, columns]
df

# %%
# scaling the data

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# %%
# initializing the optics model an testing with different parameter values

#optics = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
#optics = OPTICS(min_samples=50, xi=0.5, min_cluster_size=0.5)
#optics = OPTICS(min_samples=500, min_cluster_size=50)
optics = OPTICS(eps=5, min_samples=500, min_cluster_size=50)

#%%
# fitting the data to the model

optics.fit(scaled_df)

# %%
# storing the results
# below code is develoepd with help from sci-kit learns documentation

from sklearn.cluster import cluster_optics_dbscan
import numpy as np

# Producing the labels according to the DBSCAN technique with eps = 0.5
labels1 = cluster_optics_dbscan(reachability = optics.reachability_,
								core_distances = optics.core_distances_,
								ordering = optics.ordering_, eps = 0.5)

# Producing the labels according to the DBSCAN technique with eps = 2.0
labels2 = cluster_optics_dbscan(reachability = optics.reachability_,
								core_distances = optics.core_distances_,
								ordering = optics.ordering_, eps = 2)

# Creating a numpy array with numbers at equal spaces till
# the specified range
space = np.arange(len(scaled_df))

# Storing the reachability distance of each point
reachability = optics.reachability_[optics.ordering_]

# Storing the cluster labels of each point
labels = optics.labels_[optics.ordering_]

print(labels)

# %%
# visualizing the resulting clusters

from matplotlib import gridspec

# Defining the framework of the visualization
plt.figure(figsize =(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Plotting the Reachability-Distance Plot
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
	Xk = space[labels == Class]
	Rk = reachability[labels == Class]
	ax1.plot(Xk, Rk, colour, alpha = 0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha = 0.3)
ax1.plot(space, np.full_like(space, 2., dtype = float), 'k-', alpha = 0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype = float), 'k-.', alpha = 0.5)
ax1.set_ylabel('Reachability Distance')
ax1.set_title('Reachability Plot')

# Assuming scaled_df is a NumPy array and optics.labels_ is a NumPy array
colors = ['c.', 'b.', 'r.', 'y.', 'g.']
for Class, colour in zip(range(0, 5), colors):
    # Mask for the current class
    class_mask = optics.labels_ == Class
    # Selecting rows for the current class using the mask
    Xk = scaled_df[class_mask]
    # Plotting using the first two columns of the array
    ax2.plot(Xk[:, 0], Xk[:, 1], colour, alpha=0.3)

# Handling noise points
noise_mask = optics.labels_ == -1
# Selecting noise points using the mask
noise_points = scaled_df[noise_mask]
# Plotting noise points
ax2.plot(noise_points[:, 0], noise_points[:, 1], 'k+', alpha=0.1)

ax2.set_title('OPTICS Clustering')

colors = ['c', 'b', 'r', 'y', 'g', 'greenyellow']
for Class, colour in zip(range(0, 6), colors):
    class_mask = labels1 == Class
    Xk = scaled_df[class_mask]
    ax3.plot(Xk[:, 0], Xk[:, 1], colour, alpha=0.3, marker='.')

noise_mask = labels1 == -1
ax3.plot(scaled_df[noise_mask, 0],
         scaled_df[noise_mask, 1],
         'k+', alpha=0.1)
ax3.set_title('DBSCAN clustering with eps = 0.5')

colors = ['c.', 'y.', 'm.', 'g.']
for Class, colour in zip(range(0, 4), colors):
    class_mask = labels2 == Class
    Xk = scaled_df[class_mask]
    ax4.plot(Xk[:, 0], Xk[:, 1], colour, alpha=0.3)

noise_mask = labels2 == -1
ax4.plot(scaled_df[noise_mask, 0],
         scaled_df[noise_mask, 1],
         'k+', alpha=0.1)
ax4.set_title('DBSCAN Clustering with eps = 2.0')

plt.tight_layout()
plt.show()

# %%
