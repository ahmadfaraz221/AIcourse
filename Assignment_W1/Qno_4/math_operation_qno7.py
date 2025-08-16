def sum_of_digits(num):
    return sum(int(digit) for digit in str(num))

# Take input range from user
start = int(input("Enter the start of the range: "))
end = int(input("Enter the end of the range: "))

# Find smallest and largest integer for perfect squares
smallest = int(start ** 0.5)
largest = int(end ** 0.5)

# Generate perfect squares with sum of digits < 10
perfect_squares = []
for i in range(smallest, largest + 1):
    square = i * i
    if start <= square <= end and sum_of_digits(square) < 10:
        perfect_squares.append(square)

# Display result
print("Perfect squares with digit sum < 10:", perfect_squares)