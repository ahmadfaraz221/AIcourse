import pandas as pd

df = pd.read_csv("FastFoodRestaurants.csv",delimiter=",")

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
country = df['country']
print("access the Name column: df : ")
print(country)
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
country_keys = df ("access multiple columns: df : " ,)
print(country_keys)
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

print("# Case 3 : Using .iloc - starts here")
# Case 3 : Using .iloc - starts here
"""Using .iloc: Selection by Integer Position
.iloc selects by position instead of label. This is the standard syntax of using .iloc: df.iloc[row_indexer, column_indexer]. There are two special things to look out for:

Counting starting at 0: The first row and column have the index 0, the second one index 1, etc.
Exclusivity of range end value: When using a slice, the row or column specified behind the colon is not included in the selection."""

#Selecting a single row using .iloc
second_row = df.iloc[0]
print("#Selecting a single row using .iloc")
print(second_row)
print()

#Selecting multiple rows using .iloc
second_row2 = df.iloc[[1, 3,5]]
print("#Selecting multiple rows using .iloc")
print(second_row2)
print()

#Selecting a slice of rows using .iloc
second_row3 = df.iloc[2:5]
print("#Selecting a slice of rows using .iloc")
print(second_row3)
print()

#Selecting a single column using .iloc
second_row5 = df.iloc[:,2]
print("#Selecting a single column using .iloc")
print(second_row5)
print()

#Selecting multiple columns using .iloc
second_row6 = df.iloc[:,[2,4]]
print("#Selecting multiple columns using .iloc")
print(second_row6)
print()

#Selecting a slice of columns using .iloc
second_row7 = df.iloc[:,2:4]
print("#Selecting a slice of columns using .iloc")
print(second_row7)
print()

#Combined row and column selection using .iloc
second_row8 = df.iloc[[1, 3,5],2:4]
print("#Combined row and column selection using .iloc")
print(second_row8)
print()

# Case 3 : Using .iloc - ends here

# Next Run 
print("Next Run")

df.loc[len(df.index)] = [3477952,82,""] 
print("Modified DataFrame - add a new row:")
print(df)
print()


#Remove Rows/Columns from a Pandas DataFrame


# delete row with index 1
df.drop(1, axis=0, inplace=True)
# delete row with index 1
df.drop(index=2, inplace=True)
# delete rows with index 3 and 5
df.drop([3, 5], axis=0, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame - Remove Rows:")
print(df)



# delete age column
df.drop('page_url', axis=1, inplace=True)
# delete marital status column
df.drop(columns='property_type', inplace=True)
# delete height and profession columns
df.drop(['location', 'city'], axis=1, inplace=True)
# display the modified DataFrame after deleting rows
print("Modified DataFrame -  delete page_url ,property_type , location , city , column :")
print(df)


#Rename Labels in a DataFrame
# rename column 'Name' to 'First_Name'
df.rename(columns= {'province_name': 'province_nameChanged'}, inplace=True)
# rename columns 'Age' and 'City'
df.rename(mapper= {'bedrooms': 'bedrooms_Changed', 'date_added':'date_added_Changed'}, axis=1, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame  - Rename Labels :")
print(df)


#Example: Rename Row Labels
# rename column one index label
df.rename(index={0: 7}, inplace=True)
# rename columns multiple index labels
df.rename(mapper={1: 10, 2: 100}, axis=0, inplace=True)
# display the DataFrame after renaming column
print("Modified DataFrame - Rename Row - 0  >>> 7 , 1 >>> 10 , 2 >>> 100  Labels:")
print(df)



#query() to Select Data
#The query() method in Pandas allows you to select data using a more SQL-like syntax.

# select the rows where the age is greater than 25
selected_rows = df.query('agency == \'Gateway Properties\' or price > 11000000')

print(selected_rows.to_string())
print(len(selected_rows))



# sort DataFrame by price in ascending order
sorted_df = df.sort_values(by='price')
print(sorted_df.to_string(index=False))

#Sort Pandas DataFrame by Multiple Columns

# 1. Sort DataFrame by 'Age' and then by 'Score' (Both in ascending order)
df1 = df.sort_values(by=['price', 'location_id'])

print("Sorting by 'price' (ascending) and then by 'location_id' (ascending):\n")
print(df1.to_string(index=False))

#Pandas groupby
#In Pandas, the groupby operation lets us group data based on specific columns. This means we can divide a DataFrame into smaller groups based on the values in these columns.

# group the DataFrame by the location_id column and
# calculate the sum of price for each category
grouped = df.groupby('location_id')['price'].sum()

print(grouped.to_string())
print("grouped :" , len(grouped))


""""Pandas Data Cleaning
Data cleaning means fixing and organizing messy data. Pandas offers a wide range of tools and functions to help us clean and preprocess our data effectively.
"""
# use dropna() to remove rows with any missing values
df_cleaned = df.dropna()
print("Cleaned Data:\n",df_cleaned)


# filling NaN values with 0
df.fillna(0, inplace=True)

print("\nData after filling NaN with 0:\n", df)



# create a list named data
data = [2, 4, 6, 8]
# create Pandas array using data
array1 = pd.array(data)
print(array1)
"""<IntegerArray>
[2, 4, 6, 8]
Length: 4, dtype: Int64"""


# creating a pandas.array of integers
int_array = pd.array([1, 2, 3, 4, 5], dtype='int')
print(int_array)
print()

#Pandas Reshape
#In Pandas, reshaping data refers to the process of converting a DataFrame from one format to another for better data visualization and analysis.
#Pandas provides multiple methods like pivot(), pivot_table(), stack(), unstack() and melt() to reshape data. We can choose the method based on our analysis requirement.


# to be continued....