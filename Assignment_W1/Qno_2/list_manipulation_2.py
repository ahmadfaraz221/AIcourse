# Turn every item of a list into its square 
my_list = [2,4,5,6,7,8,]
print(type(my_list))
print("before a list: ",my_list)
square_list = [num**2 for num in my_list]
print("After square a list: ",square_list)