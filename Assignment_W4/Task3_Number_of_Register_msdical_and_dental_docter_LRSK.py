# number of rigester medical and dental docter using linear regration via skit learn 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("Assignment_W4/number-of-registered-medical-and-dental-doctors-by-gender-in-pakistan (1).csv", index_col="Years", sep=",")
print("Info(): ", df.info())
print("datatype(): ", df.dtypes)
print("describe(): ", df.describe())
print("shape(): ", df.shape)

df.plot.scatter(x = "Female Doctors", y = "Female Dentists", title = "Scatter plot of Female Doctors and Female Dentists")
plt.show()

print("df['Female Doctors']: ",df['Female Doctors'])
print("df['Female Dentists']: ", df["Female Dentists"])

x = df["Female Doctors"].values.reshape(-1,1)
y = df["Female Dentists"].values.reshape(-1,1)

print("x :  ",x)
print("y :  ",y)

print(df["Female Doctors"].values)
print(df["Female Doctors"].values.shape)

print(x.shape)
print(x)

SEED = 42

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30, random_state=SEED)
print(x_train)
print(y_train)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)
print(regressor.intercept_)
print(regressor.coef_)

def calc(slope, intercept, Assessed_Value):
    return slope*Assessed_Value + intercept
score = calc(regressor.coef_, regressor.intercept_,10)
print(score)

score = regressor.predict([[10]])
print(score)

y_pred = regressor.predict(x_test)

df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'predict' : y_pred.squeeze()})
print(df_preds)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}') # Mean absolute error: 33522110.57
print(f'Mean squared error: {mse:.2f}') # Mean squared error: 1933948878921756.75
print(f'Root mean squared error: {rmse:.2f}') # Root mean squared error: 43976685.63
print(f'R2 Score: {r2:.2f}') # R2 Score: -3.84