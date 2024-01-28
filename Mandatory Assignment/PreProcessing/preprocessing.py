import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2

df = pd.read_csv('data.csv')

#getting overview of dataframe
print (df.shape)
print(df.dtypes)

#how many null values = 58
print("How many null values: \n " + str(df.isnull().sum()) + "\n")

#fixing invalid values
df['How many letters are there in the word "Seattle"?'] = df['How many letters are there in the word "Seattle"?'].replace('Seven', 7)
#converting the discrete data column from object type to float
df['How many letters are there in the word "Seattle"?'] = df['How many letters are there in the word "Seattle"?'].astype(float)

#find the mode (most frequent value) and fill up the na values with the mode

def mode(column):
    freq = {}
    for value in column:
        freq.setdefault(value,0)
        #everytime the same index (the index = value) is getting called, the value of the index gets incremented by 1
        freq[value] += 1
    highest_freq = max(freq.values())
    mode_lst = [key for key, value in freq.items() if value == highest_freq]
    mode_value = mode_lst[0]
    return mode_value

mode_of_lettersInSeattle = mode(df['How many letters are there in the word "Seattle"?'])

df = df.fillna(value = mode_of_lettersInSeattle)

#Here we convert the timestamp column from object into datetime datatype
df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

#fixing centimeter inputs in height column (inches)
def height_conversion(height):
    #if height is in cm
    if height > 100:
        height = height / 2.54
    #if height is entered in feet
    elif height < 10:
        height = height * 12
    return height

converted_values = df['Your height (in International inches)'].apply(
    lambda height: round(height_conversion(height), 1))

df['Your height (in International inches)'] = converted_values

#fixing shoe size converting to eu standards

us_and_eu_shoeSize= {
    6.0: 38.0,
    6.5: 38.5,
    7.0: 39.0,
    7.5: 40.0,
    8.0: 41.0,
    8.5: 41.5,
    9.0: 42.0,
    9.5: 42.5,
    10.0: 43.0,
    10.5: 44.0,
    11.0: 44.5,
    11.5: 45.0
}

def shoesize_conversion(size):
    if size in us_and_eu_shoeSize:
        size = us_and_eu_shoeSize[size]
    return size

converted_values = df['Your mean shoe size (In European Continental system)'].apply(
    lambda size: round(shoesize_conversion(size), 1))

df['Your mean shoe size (In European Continental system)'] = converted_values

#now we will normalize the string values in the discrete data type columns
 
most_used_contractions = {
        "I'd": "I had,  I would",
        "It'll": "It will",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "let's": "let us",
    }

#split the contractions into two seperate words for precision!

def no_contractions(string):
    list_of_words = string.split()
    for i in range(len(list_of_words)):
        if list_of_words[i] in most_used_contractions:
            list_of_words[i]= most_used_contractions[list_of_words[i]]

    final_string = ' '.join(list_of_words)
    return final_string

# Preprocess text data
def normalize_strData(string):
        string = no_contractions(string)
        string = string.lower()
        string = re.sub(r'[^\w\s]', '', string)
        string = string.strip()
        temp_lst = string.split()
        return temp_lst

df['Why are you taking this course?'] = df['Why are you taking this course?'].apply(normalize_strData)

#since programme is a target column, ill categorize the values into numerical values depending on programme for the supervised method (NB)
df['Which programme are you studying?'] = df['Which programme are you studying?'].apply(normalize_strData)

#how many distinct programmes do we have?

distinct_programmes_count = np.unique(df['Which programme are you studying?'])

print("\n Number of distinct masters: " + str(len(distinct_programmes_count)) + "\n")

#so we have 4 different educations present: 
def target_column(column):
    for lst in column:
        if "software" in lst:
            return 1.0
        elif "computer" in lst:
            return 2.0
        elif "games" in lst:
            return 3.0
        elif "elsewhere" in lst:
            return 4.0

df['Which programme are you studying?'] = df['Which programme are you studying?'].apply(target_column)

#creating function for zscore to exclude outliers

###############
def zscore(column, mean, std):
    temp = []
    for dp in column:
        score = (dp - mean) / std
        temp.append(score)
    return temp

column_to_score = ['Your mean shoe size (In European Continental system)', 'Your height (in International inches)','How many letters are there in the word "Seattle"?']

threshold = 2.0

for column in column_to_score:
    df['zscores'] = np.abs(zscore(df[column],df[column].mean(),df[column].std()))
    outliers = np.where(df['zscores'] > threshold)
    df = df.drop(df.index[outliers])

#dropping temporary zscores column

df = df.drop('zscores', axis=1)

##################

print(df)

print(df.dtypes)

##visualizing frequencies in the featured columns used in the clustering and NB

plt.figure(figsize=(10, 6))
plt.hist(df['Your mean shoe size (In European Continental system)'], bins=50, color='skyblue', alpha=0.7, rwidth=0.85)
plt.title('Gaussian distribution of Shoe Size')
plt.xlabel('Shoe Size')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df['Your height (in International inches)'], bins=50, color='skyblue', alpha=0.7, rwidth=0.85)
plt.title('Gaussian distribution of height')
plt.xlabel('Height')
plt.ylabel('Frequency')
plt.show()

df.to_csv('clustering_data.csv', index=False)