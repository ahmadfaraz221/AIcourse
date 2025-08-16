#  Create a New String Made up of First and Last 2 Characters
input_string = input("Enter a string: ")

# Check if the string has at least 2 characters
if len(input_string) < 2:
    print("The string is too short!")
else:
    # Create new string with first 2 and last 2 characters
    new_string = input_string[:2] + input_string[-2:]
    print("New string:", new_string)