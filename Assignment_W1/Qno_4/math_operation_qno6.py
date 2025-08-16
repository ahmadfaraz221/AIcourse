# Check If Two Numbers are Amicable Numbers or Not
def sum_of_divisors(num):
    divisors_sum = 1  # 1 is a divisor of all numbers
    for i in range(2, num // 2 + 1):
        if num % i == 0:
            divisors_sum += i
    return divisors_sum

# Take input from the user
num1 = int(input("Enter the first number: "))
num2 = int(input("Enter the second number: "))

# Check for amicable numbers
if sum_of_divisors(num1) == num2 and sum_of_divisors(num2) == num1:
    print(num1, "and", num2, "are Amicable Numbers.")
else:
    print(num1, "and", num2, "are NOT Amicable Numbers.")