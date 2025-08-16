# Check if a value exists in a dictionary 
my_dictionary = {"a":80,"b":86,"c":70,"d":78}
print(type(my_dictionary))
print(my_dictionary)
value = 60
if value >= 60:
    print("value are exist in dictionary")
else:
    print("value are not exist in dictionary")

print("NEXT HERE...")

#2  Get the key of a minimum value from the following dictionary 
min_value = min(my_dictionary.values())
print("min value: ",min_value)

print("NEXT HERE...")

#3 Delete a list of keys from a dictionary 
del my_dictionary["d"]
print("after delete a key in dictionary: ",my_dictionary)

