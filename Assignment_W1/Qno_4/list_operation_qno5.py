# Take list input from user
numbers = list(map(int, input("Enter numbers separated by space: ").split()))

# Take the number to search
search_num = int(input("Enter the number to count: "))

# Count occurrences
count = numbers.count(search_num)

# Display result
print(f"The number {search_num} appears {count} time(s) in the list.")
