# Find the LCM of Two Numbers
def compute_gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def compute_lcm(a, b):
    return abs(a * b) // compute_gcd(a, b)

# Input two numbers
num1 = int(input("Enter first number: "))
num2 = int(input("Enter second number: "))

# Compute and print LCM
lcm = compute_lcm(num1, num2)
