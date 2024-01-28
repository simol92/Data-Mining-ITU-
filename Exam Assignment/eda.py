
# %%
# IMPORTS:
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import get_dfs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# define functions for plotting features by year


def plot_features_by_year(df):
    for feature in numerical_df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=df, x='year', y=feature)
        plt.title(f"{feature} over years")
        plt.show()


def plot_yearly_features_by_genre(df):
    for genre in df['playlist_genre'].unique():
        plot_features_by_year(data_df[data_df['playlist_genre'] == genre])


def plot_features_by_year_compare_genres(df):
    for feature in numerical_df.columns:
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=data_df, x='year', y=feature, hue='playlist_genre')
        plt.title(f"{feature} over years")
        plt.show()


def print_min_max_features():
    for feature in numerical_df.keys():
        feature_max = data_df.loc[data_df[feature].idxmax()]
        feature_min = data_df.loc[data_df[feature].idxmin()]
        print(f"\n############{feature}#########\n ")
        print(f"min {feature}: \n{feature_min}")
        print(f"\nfeature: {feature}\ntrack: {feature_max['track_name']}\n")
        print(f"max {feature}: \n{feature_max}")
        print(f"\nfeature: {feature}\ntrack: {feature_min['track_name']} \n ")


data_df, numerical_df, holdout_df = get_dfs()
data_df.describe()

# %%
# Full plot of 2-way data combinations (and distribution for each attribute)
sns.pairplot(data_df)

# %%
# TAKES A LONG TIME!
sns.pairplot(data_df, hue='playlist_genre')
# %%

# %%
# Looking at relationships between audio features
correlation_matrix = numerical_df.corr()
correlation_matrix
# %%
sns.heatmap(correlation_matrix, cmap='coolwarm', fmt='.1f')
plt.show()
# %%
plt.hist(data_df['year'], bins=62)
# %%
sns.barplot(data_df, x='playlist_genre', y='track_popularity')
# %%
sns.barplot(data_df, x='playlist_subgenre', y='track_popularity')
plt.xticks(rotation=45, ha='right')
plt.show()

# %%
# plotting all audio features in relation to genre
for feature in numerical_df.columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='playlist_genre', y=feature, data=data_df)
    plt.show()


# %%
for feature in numerical_df.keys():
    sns.barplot(data_df, x='playlist_subgenre', y=feature)
    plt.xticks(rotation=45, ha='right')
    plt.show()
# %%
for feature in numerical_df.keys():
    sns.barplot(data_df, x='playlist_genre', y=feature)
    plt.xticks(rotation=45, ha='right')
    plt.show()

# %%
plot_features_by_year(data_df)


# INVESTIGATING MOST EXTREME TRACKS FOR EACH MUSICAL FEATURE
# %%
print_min_max_features()

# %%
# Plot distribution of each audio feature:
for feature in numerical_df.keys():
    plt.clf()
    plt.title(f'Histogram of {feature} distribution')
    plt.hist(numerical_df[feature], bins=100)
    plt.show()

# EXTRA JUST FOR
# %%
# investigating whether key and mode make sense
bjork = data_df[data_df['track_artist'] == 'Björk']
bjork.head()
# %%
# HUMBLE. is both categorized as pop and rap
s = data_df[data_df['track_artist'] == 'Kendrick Lamar']
s.head()


# %%
# plotting the evolution of all features
# MAYBE OVERKILL
plot_features_by_year_compare_genres(data_df)
# %%
plot_yearly_features_by_genre(data_df)
# %%
