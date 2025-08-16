#Count Number of Uppercase and Lowercase Letters in a String
text = input("Enter a string: ")

uppercase_count = 0
lowercase_count = 0

for char in text:
    if "A" <= char <= "Z":
        uppercase_count += 1
    elif "a" <= char <= "z":
        lowercase_count += 1

print("Number of uppercase letters:", uppercase_count)
print("Number of lowercase letters:", lowercase_count)    

