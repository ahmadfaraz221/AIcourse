# Find the Larger String without using Built-in Functions
a = input("Enter first string: ")
b = input("Enter second string: ")

count1 = 0
for i in a:
    count1 = count1 + 1

count2 = 0
for i in b:
    count2 = count2 + 1

if count1 > count2:
    print("Larger string is:", a)
elif count2 > count1:
    print("Larger string is:", b)
else:
    print("Both strings are equal in length.")   