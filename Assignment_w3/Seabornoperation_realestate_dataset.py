import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
url = "Assignment_w3/RealEstate-USA.csv"
df = pd.read_csv(url)

# Display DataFrame
print(df.head())
print(df.info())
print(df.dtypes)
print(df.describe())
print(df.shape)

# line plot

plt.figure(figsize=(10,5))
sns.lineplot(x="city", y="price", data=df)
plt.title("Line Plot - City vs Price")
plt.xticks(rotation=90)
plt.show()

# CatPlot (city vs price)

sns.catplot(x="city", y="price", kind="bar", data=df, height=6, aspect=2)
plt.xticks(rotation=90)
plt.show()

#KDE Plot (zip_code vs price)

plt.figure(figsize=(10,5))
sns.kdeplot(x="zip_code", y="price", data=df, fill=True, cmap="viridis")
plt.title("KDE Plot - Zip Code vs Price")
plt.show()

#Scatter Plot (zip_code vs price)

plt.figure(figsize=(10,5))
sns.scatterplot(x="zip_code", y="price", data=df)
plt.title("Scatter Plot - Zip Code vs Price")
plt.show()
