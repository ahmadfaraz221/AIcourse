# Take list input from user
numbers = list(map(int, input("Enter numbers separated by space: ").split()))

# Find the largest number
largest = max(numbers)

# Display result
print("The largest number in the list is:", largest)