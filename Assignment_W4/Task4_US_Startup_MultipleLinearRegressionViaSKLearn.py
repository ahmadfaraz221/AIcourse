# Using multiple linear regression on the us startup dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("Assignment_W4/50_Startups (1).csv")
print("Info(): ", df.info())
print("datatype(): ", df.dtypes)
print("describe(): ", df.describe())
print("shape(): ", df.shape)

variables = ['R&D Spend', 'Administration', 'Marketing Spend']
for var in variables:
    plt.figure()

    sns.regplot(x=var, y='Profit', data=df).set(title=f'Regression plot of {var} and Profit')
    plt.show()

read = input("Wait here: \n")
plt.figure()
correlations = df.select_dtypes(include=['number']).corr()
print(correlations)

g = sns.heatmap(correlations, annot=True).set(title='Heat map of Consumption Data - Pearson Correlations')

plt.show()
read = input("Wait for me....")

y = df['Profit']
x = [['R&D Spend', 'Administration', 'Marketing Spend']]


SEED = 200

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10,random_state=SEED)
print("x.shape :     \n", x.shape) 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, y_train)
print("regressor.intercept_......\n", regressor.intercept_)
print("regressor.coef_ " , regressor.coef_)

feature_names = x.columns
model_coefficients = regressor.coef_

coefficients_df = pd.DataFrame(data = model_coefficients, 
                              index = feature_names, 
                              columns = ['Coefficient value'])
print(coefficients_df)

y_pred = regressor.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("Actual vs Predicted.....\n" , results)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')

actual_minus_predicted = sum((y_test - y_pred)**2)
actual_minus_actual_mean = sum((y_test - y_test.mean())**2)
r2 = 1 - actual_minus_predicted/actual_minus_actual_mean
print('R²:', r2)
print(" R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class...\n", regressor.score(X_test, y_test))

