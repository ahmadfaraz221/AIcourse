# Using linear regression via Skit Learn on Real Estate Sales Dataset
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

df = pd.read_csv("Assignment_W4/Real_Estate_Sales_2001-2022_GL-Short.csv",index_col="Serial Number")
print("Info(): ", df.info())
print("datatype(): ", df.dtypes)
print("describe(): ", df.describe())
print("shape(): ", df.shape)

df.plot.scatter(x = "Assessed Value", y = "Sale Amount", title = "Scatter plot of Assessed value and Sale Amount")
plt.show()

print("df['Assessed Value']: ",df['Assessed Value'])
print("df['Sale Amount']: ", df["Sale Amount"])

x = df["Assessed Value"].values.reshape(-1,1)
y = df["Sale Amount"].values.reshape(-1,1)

print("x :  ",x)
print("y :  ",y)

print(df["Assessed Value"].values)
print(df["Assessed Value"].values.shape)

print(x.shape)
print(x)

SEED = 42

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.10, random_state=SEED)
print(x_train)
print(y_train)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope, intercept, Assessed_Value):
    return slope*Assessed_Value + intercept
score = calc(regressor.coef_, regressor.intercept_,12)
print(score)

score = regressor.predict([[12]])
print(score)

y_pred = regressor.predict(x_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'predict' : y_pred.squeeze()})
print(df_preds)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}') # Mean absolute error: 139999.37
print(f'Mean squared error: {mse:.2f}') #Mean squared error: 53563564913.92
print(f'Root mean squared error: {rmse:.2f}') # Root mean squared error: 231438.04
print(f'R2 Score: {r2:.2f}') # R2 Score: 0.95

