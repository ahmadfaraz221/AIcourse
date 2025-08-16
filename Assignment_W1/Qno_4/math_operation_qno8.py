num = int(input("Enter a number: "))

# Find the number of digits
n = len(str(num))

# Calculate the sum of digits raised to the power n
sum_of_powers = sum(int(digit) ** n for digit in str(num))

# Check and display result
if sum_of_powers == num:
    print(num, "is an Armstrong number.")
else:
    print(num, "is NOT an Armstrong number.")