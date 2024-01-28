import numpy  as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2


class kMeans:

    #defining the constructor to have a default k = 3
    #also initializing an empty centroid class variable for later use
    def __init__(self, k=3):
        self.k = k
        self.centroids = None

    #np automatically sum up the dp's distance to every centroid, then we take
    #the square root of the sum to find the euclidian distance!
    @staticmethod
    def euc_distance(dp, centroids):
        distance = np.sqrt(np.sum((centroids - dp)**2, axis=1))
        return distance
    
    #default max iteration = 200, we concluded that a max iteration above 200 didnt do much difference in
    #terms of the resulting clustering
    def fit(self, sample, max_iterations=200):
        #making sure that the sample values dtype is float
        sample = sample.values.astype(float)

        # we set a random uniform distribution 
        # for every dimension in our sample data, we set a min and a max value for each respective diension
        # to make sure that the centroids are inside the dimensions and not outside
        self.centroids = np.random.uniform(np.amin(sample, axis=0), np.amax(sample, axis=0),size=(self.k, sample.shape[1]))

        for iteration in range(max_iterations):

            #cluster labels is assigned to every DP after clustering depending on k
            cluster_labels = []

            for dp in sample:
                #for every datapoint, compute the distance to each centroid
                distance = kMeans.euc_distance(dp, self.centroids)
                #take the centroid with the lowest distance
                assigning_dp_to_clusters = np.argmin(distance)
                #assign the DP to a cluster
                cluster_labels.append(assigning_dp_to_clusters)

            #create an array instead of a list for better indexing
            cluster_labels = np.array(cluster_labels)

            #below: for every datapoint, which cluster does it belong to?
            #also: for each cluster, which indices (datapoints) belong to that cluster?
            clu_indices = []
            for i in range(self.k):
                clu_indices.append(np.argwhere(cluster_labels == i))
            
            #creating a new list for possible changes in centroid positions
            new_centroids_pos = []

            for i, number_of_dps in enumerate(clu_indices):
                #if the cluster has no datapoints assigned, just stay where "you" are!
                if len(number_of_dps) == 0:
                    new_centroids_pos.append(self.centroids[i])
                else:
                    #assign the new centroid position to the average / mean position 
                    # of its assigned datapoints by taking the first value in the appended list
                    new_centroids_pos.append(np.mean(sample[number_of_dps],axis=0)[0])
            
            #if the difference between the current centroid positions and the new centroid positions
            # are less than 0.0001, just break the loop 
            # otherwise, reassign the existing centroids to the new positions
            if np.max(self.centroids - np.array(new_centroids_pos)) < 0.0001:
                break
            else:
                #assigning new positions to centroids
                self.centroids = np.array(new_centroids_pos)
            
        return cluster_labels

import pandas as pd

df = pd.read_csv('transformed_data.csv')
df = df.rename(columns={'Which programme are you studying?': 'Target'})

cluster_cols = ['Your mean shoe size (In European Continental system)', 'Your height (in International inches)']

new_df = df[cluster_cols]

# We will now show the elbow method using WCSS:
# WCSS = within-cluster sum of squares
# for a range from 1 to 7 clusters, what is the most optimal k-number of clusters? 
wcss = []

for i in range (1,7):
    tempkmeans = kMeans(k=i)
    templabels = tempkmeans.fit(new_df, max_iterations=200)
    #calculating wcss
    wcss.append(sum(np.min(np.square(new_df - tempkmeans.centroids[templabels]), axis=1)))

plt2.plot(range(1, 7), wcss, marker='o')
plt2.title('Elbow Method')
plt2.xlabel('Number of Clusters')
plt2.ylabel('WCSS')
plt2.show()

######### clustering with the most optimal WCSS score, k = 3

kmeans = kMeans(k=3)
labels = kmeans.fit(new_df, max_iterations=200)
#visualizing the cluster
plt.scatter(new_df.iloc[:, 0], new_df.iloc[:, 1], c=labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c=range(len(kmeans.centroids)), marker='*', s=100)
plt.title('Clustering')
plt.xlabel('Shoe Size')
plt.ylabel('Height')
plt.show()


df['clusters'] = labels

print(df)

df.to_csv('data_with_clusters.csv', index=False)