# %%
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import get_dfs, print_confusion_matrix, plot_feature_importances, year_to_decade
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, multilabel_confusion_matrix
from pprint import pprint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

data_df, numerical_df, test_df = get_dfs()

# function to easily print and plot metrics in one line for all models


def classification_metrics(X, y, model, y_test, y_pred):
    classification_report(y_test, y_pred)
    print_confusion_matrix(y_test, y_pred, y)
    plot_feature_importances(model, X)


# %%

# BASELINE MODEL FOR COMPARISON
X = numerical_df
y = data_df['playlist_genre']
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)
dummy = DummyClassifier(strategy='uniform')


dummy.fit(X_train, y_train)

y_pred_majority = dummy.predict(X_validation)


print_confusion_matrix(y_validation, y_pred_majority, y)
# %%
# TRYING TO PREDICT GENRE BASED ON MUSICAL FEATURES
X = numerical_df
y = data_df['playlist_genre']
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)

classification_metrics(X, y, model, y_validation, y_pred)

# FINAL PREDICTION USING TEST SET
# %%
# creating test data
X_test = test_df[X.columns]
y_test = test_df['playlist_genre']
# final prediction
y_pred = model.predict(X_test)

classification_metrics(X, y, model, y_test, y_pred)
# %%
# IMPROVED VERSION OF DECISION TREE INCLUDING GRID SEARCH, DROPPING "MODE" AND "KEY"
#  dropping 'mode' and 'key' since they do not make any difference in the result
X = numerical_df.drop(['mode', 'key'], axis=1)
y = data_df['playlist_genre']
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier(random_state=42)
param_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'min_samples_split': [2, 4, 8],
    'max_depth': [5, 10, 20, 30]
}

grid_search = GridSearchCV(model, param_grid, scoring='accuracy')
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_validation)
print(grid_search.best_params_)

classification_metrics(X, y, grid_search.best_estimator_, y_validation, y_pred)

# FINAL PREDICTION USING TEST SET
# %%
# creating test data
X_test = test_df[X.columns]
y_test = test_df['playlist_genre']
# final prediction
y_pred = grid_search.predict(X_test)
classification_metrics(X, y, grid_search.best_estimator_, y_test, y_pred)

# %%
# PREDICTING PlAYLIST GENRE BASED ON track_name AND track_album_name


data_df['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
X = data_df['text']
y = data_df['playlist_genre']
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe = Pipeline([
    # ('count', CountVectorizer()),

    ('tfidf', TfidfVectorizer(ngram_range=(1, 6))),
    # ('naive_bayes', MultinomialNB()),
    # ('svc', SVC()),
    # ('lr', LogisticRegression()),
    ('dt', DecisionTreeClassifier(criterion='gini', random_state=42)),
    # ('rf', RandomForestClassifier()),
    # ('knn', KNeighborsClassifier(n_neighbors=300)),
    # ('gb', GradientBoostingClassifier()),
    # ('mlp', MLPClassifier())
])

pipe.fit(X_train, y_train)


ypred = pipe.predict(X_validation)
print(classification_report(y_validation, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_validation, ypred, y)

# FINAL PREDICTION USING TEST SET
# %%
test_df['text'] = test_df['track_name'] + ' ' + test_df['track_album_name']
# creating test data
X_test = test_df['text']

y_test = test_df['playlist_genre']
# final prediction
y_pred = pipe.predict(X_test)

print_confusion_matrix(y_test, y_pred, y)
# %%

# COMBINING TEXT AND MUSICAL FEATURES FOR GENRE PREDICTION


X = numerical_df
X['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
y = data_df['playlist_genre']
# preprocessing steps for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 6)), 'text'),
        # using numeric columns only
        ('scaler', StandardScaler(), X.select_dtypes(include='number').columns)
    ]
)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('dt', DecisionTreeClassifier(criterion='gini', random_state=42))
])
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe.fit(X_train, y_train)
ypred = pipe.predict(X_validation)
print(classification_report(y_validation, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_validation, ypred, y)
# %%
# FINAL PREDICTION USING TEST SET

X_test = test_df[numerical_df.columns]

X_test['text'] = test_df['track_name'] + ' ' + test_df['track_album_name']

# creating test data

y_test = test_df['playlist_genre']
# final prediction
y_pred = pipe.predict(X_test)

print_confusion_matrix(y_test, y_pred, y)


########### models not included in report #######
# %%
# PREDICTING SUBGENRE BASED ON MUSICAL FEATURES
X = numerical_df.drop(['key', 'mode'], axis=1)
y = data_df['playlist_subgenre']
X = X.drop('text', axis=1)

X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)
classification_metrics(X, y, model, y_validation, y_pred)

# %%
# PREDICTING SUBGENRE BY COMBINED TEXT AND MUSICAL FEATURES
X = numerical_df
X['text'] = data_df['track_name'] + ' ' + data_df['track_album_name']
y = data_df['playlist_subgenre']
# preprocessing steps for different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(ngram_range=(1, 6)), 'text'),
        # using numeric columns only
        ('scaler', StandardScaler(), X.select_dtypes(include='number').columns)
    ]
)
pipe = Pipeline([
    ('preprocessor', preprocessor),
    # ('poly_features', PolynomialFeatures()),
    ('regression', LogisticRegression())
])
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)


pipe.fit(X_train, y_train)
ypred = pipe.predict(X_validation)
print(classification_report(y_validation, ypred))
# ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test)
print_confusion_matrix(y_validation, ypred, y)


#################### ADDITIONAL MODELS ##################
# %%
# TRYING TO PREDICT TRACK POPULARITY BASED ON MUSICAL FEATURES
X = numerical_df.drop(['track_popularity', 'text'], axis=1)
y = data_df['track_popularity']
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('poly_features', PolynomialFeatures()),
    ('regression', Ridge())
])
param_grid = {
    'poly_features__degree': [1, 2, 3,],
    'regression__alpha': [0.1, 0.5, 1, 10,]
}

grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

y_pred = grid_search.predict(X_validation)

mae = mean_absolute_error(y_validation, y_pred)
mse = mean_squared_error(y_validation, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_validation, y_pred)

metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
# Horrible performance
pprint(metrics)
print(grid_search.best_params_)


# %%
# Trying to predict decade based on audio features
df = year_to_decade(data_df)
y = df['decade']

# X = X[['speechiness', 'danceability']]
X = numerical_df.drop('text', axis=1)
X_train, X_validation, y_train, y_validation = train_test_split(
    X, y, test_size=0.33, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_validation)

classification_metrics(X, y, model, y_validation, y_pred)
# %%
# %%
# param_grid = {
#     'poly_features__degree': [1, 2, 3,],
#     # 'regression__C': [0.1, 0.5, 1, 10,]
# }
# pipe = Pipeline([
#     ('preprocessor', preprocessor),
#     ('poly_features', PolynomialFeatures()),
#     ('regression', LogisticRegression())
# ])
# model = GridSearchCV(pipe, param_grid, scoring='f1')
# model.fit(X_train, y_train)
# ypred = model.predict(X_test)
# classification_metrics(X, y, model, y_test, y_pred)

# %%
