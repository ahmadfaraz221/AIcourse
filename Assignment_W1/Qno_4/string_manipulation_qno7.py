#Check if the Substring is Present in the Given String
main_str = input("Enter the main string: ")
sub_str = input("Enter the substring to search: ")

# Check if substring is in the main string
if sub_str in main_str: #using in function for this 
    print("Substring is present in the string.")
else:
    print("Substring is not present in the string.")