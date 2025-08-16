# Find the Area of a Triangle
import math
a = float(input("Enter first side: "))
b = float(input("Enter second side: "))
c = float(input("Enter third side: "))
# Calculate semi-perimeter
s = (a+b+c)/2
area = math.sqrt(s * (s - a) * (s - b) * (s - c))
print("Area of the triangle is:", area)

