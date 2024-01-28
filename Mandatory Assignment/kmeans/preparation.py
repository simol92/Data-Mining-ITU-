import numpy as np
import pandas as pd


#preparing our data to be clustered with focus on the height and shoesize columns
df = pd.read_csv('clustering_data.csv')

# removing the first and last column
df = df.iloc[:, 1:-1]

# setting up "target" column
df = df.rename(columns={'Which programme are you studying?': 'Target'})
df = df.rename(columns={'Why are you taking this course?': 'Number of words written'})

# converting'Number of words written' into a single numerical value
df['Number of words written'] = df['Number of words written'].apply(lambda row: len(eval(row)))

# scaling the shoe size numerical values into a range from 0 to 1
df['Your mean shoe size (In European Continental system)'] = round((df['Your mean shoe size (In European Continental system)'] - df['Your mean shoe size (In European Continental system)'].min()) / (df['Your mean shoe size (In European Continental system)'].max() - df['Your mean shoe size (In European Continental system)'].min()), 3)

# scaling the height numerical values into a range from 0 to 1
df['Your height (in International inches)'] = round((df['Your height (in International inches)'] - df['Your height (in International inches)'].min()) / (df['Your height (in International inches)'].max() - df['Your height (in International inches)'].min()), 3)

# scaling the number of words from 0 to 1
df['Number of words written'] = round((df['Number of words written'] - df['Number of words written'].min()) / (df['Number of words written'].max() - df['Number of words written'].min()), 3)

print(df.dtypes)
df.to_csv('transformed_data.csv', index=False)
