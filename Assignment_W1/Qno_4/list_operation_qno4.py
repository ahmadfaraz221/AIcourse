# Take list input from user
numbers = list(map(int, input("Enter numbers separated by space: ").split()))
# Calculate average
average = sum(numbers) / len(numbers)

print("The average of the list elements is:", average)