import pandas as pd

df = pd.read_csv("My_Project_Work/startup_growth_investment_data/startup_growth_investment_data.csv",delimiter=",")

print(df)

print("df-data type" , df.dtypes)

print("df.info():  ",df.info() )
# display the first three rows
print('First Three Rows:')
print(df.head(3))

# display the first three rows
print('First Three Rows:')
print(df.head(3))

# of SSummarytatistics of DataFrame using describe() method.
print("Summary of Statistics of DataFrame using describe() method", df.describe())

#Counting the rows and columns in DataFrame using shape(). It returns the no. of rows and columns enclosed in a tuple.
print("Counting the rows and columns in DataFrame using shape() : " ,df.shape)
print()

# access the Name column
Country = df['Country']
print("access the Name column: df : ")
print(Country)
print()

"""
0            Real Biz International
1                       Khan Estate
2                   Shahum Estate 2
3                               NaN
4                               NaN
                   ...
144    Harum Real Estate & Builders
145                     Almo Estate
146              Gateway Properties
147             Chughtai Associates
148             Chughtai Associates
"""

# access multiple columns
# = df[['agency','agent']]
Industry_Number_of_Investors = Industry_Number_of_Investors = df[['industry', 'number_of_investers']]
print(Industry_Number_of_Investors)
print()


# access multiple columns
country_keys = df[['country','keys']]
print("access multiple columns: df : ")
print(country_keys)
print()



#Selecting a single column using .loc
second_row5 = df.loc[:1,"country_keys"]
print("#Selecting a single column using .loc")
print(second_row5)
print()

#Selecting multiple columns using .loc
second_row6 = df.loc[:1,['country','keys']]
print("#Selecting multiple columns using .loc")
print(second_row6)
print()

#Selecting multiple columns using .loc
second_row7 = df.loc[:1,['country','keys']]
print("#Selecting multiple columns using .loc")
print(second_row7)
print()

#Combined row and column selection using .loc
second_row8 = df.loc[df['country'] == 'keys','address':'country']
print("#Combined row and column selection using .loc")
print(second_row8)
print()
# Case 1 : using .loc - default case - ends here
