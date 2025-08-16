import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Seaborn-property-data-by-Kaggle.py.csv',delimiter=",")

print(df.dtypes)
dffilter= df.head(40)
dffilter100= df.head(100)
print(df)
sns.set(style="whitegrid")
g=sns.displot(data=dffilter, x="house_size" , y="price" , kind='hist'  )
g.figure.suptitle("sns.displot(data=dffilter, x=house_size , y=price ,  kind='hist'  )"  )
g.figure.show()

wait = input("wait...")

# Sample data
data = pd.DataFrame({'x': np.arange(100), 'y': np.random.rand(100).cumsum()})

# Set the theme
sns.set_theme(style='darkgrid')
# Alternatively
# sns.set_style('darkgrid')

# Create a plot
sns.lineplot(x='x', y='y', data=data)
plt.show()