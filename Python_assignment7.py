1. Load the dataset using pandas:


import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

print(df.head())    

2. Explore the structure of the dataset:


print(df.dtypes)

print(df.isnull().sum())

3. Clean the dataset:

If there are missing values, you can either drop them or fill them with a specific value (e.g., mean).


df.fillna(df.mean(), inplace=True)

print(df.isnull().sum())


Task 2: Basic Data Analysis

1. Compute basic statistics for numerical columns:

                                                                                                                                                                 
print(df.describe())

2. Perform groupings on a categorical column (e.g., species) and compute the mean:

species_grouped = df.groupby('species').mean()
print(species_grouped)

3. Identify any interesting findings:

Task 3: Data Visualization

1. Line chart showing trends over time

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(data=df, x='species', y='sepal length (cm)', marker='o')

plt.title('Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

2. Bar chart showing comparison of a numerical value across categories (e.g., average petal length per species):


sns.barplot(x='species', y='petal length (cm)', data=df)

plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()
3. Histogram of a numerical column to understand its distribution (e.g., sepal width):


sns.histplot(df['sepal width (cm)'], kde=True, bins=10)

plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()
4. Scatter plot to visualize the relationship between two numerical columns (e.g., sepal length vs. petal length):


sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', data=df)

plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.show()

Error Handling

   
    df = pd.read_csv('dataset.csv')
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")
except pd.errors.EmptyDataError:
    print("No data found. Please check the file content.")
except Exception as e:
    print(f"An error occurred: {e}")