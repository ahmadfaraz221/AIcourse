#create a new string made of an input string’s first, middle, and last character
String = input("Enter a string: ")
# hre we print the first ,midle and last character of the input
New_String = String[0] + String[len(String)//2] + String[-1] 
# Print output that we given
print(String) 
print("New String: ",New_String) 

print("NEXT HERE...")

#2 print the length of the given input 
print(len(String)) 

print("NEXT HERE...")
#3 Reverse a given string
Reverse_String = String[::-1]
print("Reversed String: ",Reverse_String)

print("NEXT HERE...")

#4 Split a string on hyphens 
Split = String.split("-")
print("Splite string: ",Split)

