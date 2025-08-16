# Remove empty strings from the list of strings 
my_list = ["hello","pakistan"," ","shan",]
print(type(my_list))
print(my_list)
my_list.remove(" ")
print("after remove empty string: ",my_list)

print("NEXT HERE...")

#4 add a new item in list
my_list.append("apple") # in the append we give only item to add 
print("after append: ",my_list)
my_list.insert(2,"friend") # in the insert we give item with palce whre we palce it
print("after insert: ",my_list)

print("NEXT HERE...")
#5 Replace list’s item with new value if found
my_list[0] = "Hello"
print("after replace: ",my_list)
