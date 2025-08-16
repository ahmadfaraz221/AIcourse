#This is a Python Program to display which letters are in the two strings but not in both.
a = input("Enter first string: ")
b = input("Enter second string: ")

# Convert to sets
set1 = set(a)
set2 = set(b)

# Find letters in one but not both
result = set1 ^ set2   # symmetric difference = [we can write "result = set1.symmetric_difference(set2)" ]

# Print result
print("Different letters:", result)