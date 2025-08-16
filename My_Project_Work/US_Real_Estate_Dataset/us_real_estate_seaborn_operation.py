import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample data
data = pd.DataFrame({'x': np.arange(100), 'y': np.random.rand(100).cumsum()})

# Set the theme
sns.set_theme(style='darkgrid')
# Alternatively
# sns.set_style('darkgrid')

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

# Other themes can be set similarly
sns.set_theme(style='whitegrid')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='dark')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='white')
sns.lineplot(x='x', y='y', data=data)
plt.show()

sns.set_theme(style='ticks')
sns.lineplot(x='x', y='y', data=data)
plt.show()

# Customize the theme
sns.set_theme(style='darkgrid', rc={'axes.facecolor': 'grey', 'grid.color': 'white'})

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()

#Zameencom data - based examples
# Load data from a CSV file
df = pd.read_csv("\\US_Real_Estate_Dataset\\RealEstate-USA.csv", delimiter=";")
print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)
sns.set(style="whitegrid")

g=sns.displot(data=dffilter, x="agency" , y="price" , hue="agent",  kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=agency , y=price , hue=agent,  kind='hist'  )"  )

# Display the plot
g.figure.show()
read = input("Wait for me....")