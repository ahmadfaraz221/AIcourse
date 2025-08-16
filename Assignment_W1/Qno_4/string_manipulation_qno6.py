#  Check if Two Strings are Anagram
a = input("Enter first string: ")
b = input("Enter second string: ")

# Remove spaces and convert to lowercase
a = a.replace(" ", "").lower()
b = b.replace(" ", "").lower()

# Convert strings to lists and sort them
list_a = list(a)
list_b = list(b)

list_a.sort()
list_b.sort()

# Compare sorted lists
if list_a == list_b:
    print("Strings are anagrams.")
else:
    print("Strings are not anagrams.")