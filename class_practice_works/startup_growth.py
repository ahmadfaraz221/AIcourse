import numpy as np

Industry ,Funding_Rounds , Number_of_Investors ,Country = np.genfromtxt('startup_growth_investment_data.csv', delimiter=',', usecols=(1,2,5,6), unpack=True, dtype=None,skip_header=1)

print(Industry)
print(Funding_Rounds)
print(Number_of_Investors)
print(Country)

# Zameen.com price  - statistics operations
print("Zameen.com Price average: " , np.average(Number_of_Investors))
print("Zameen.com Price std: " , np.std(Number_of_Investors))
print("Zameen.com Price mean: " , np.mean(Number_of_Investors))
print("Zameen.com Price mod: " , np.median(Number_of_Investors))
print("Zameen.com Price percentile - 25: " , np.percentile(Number_of_Investors,25))
print("Zameen.com Price percentile  - 75: " , np.percentile(Number_of_Investors,75))
print("Zameen.com Price percentile  - 3: " , np.percentile(Number_of_Investors,3))
print("Zameen.com Price min : " , np.min(Number_of_Investors))
print("Zameen.com Price max : " , np.max(Number_of_Investors))

# Zameen.com price  - maths operations
print("Zameen.com Price square: " , np.square(Number_of_Investors))
print("Zameen.com Price sqrt: " , np.sqrt(Number_of_Investors))
print("Zameen.com Price pow: " , np.power(Number_of_Investors,Number_of_Investors))
print("Zameen.com Price abs: " , np.abs(Number_of_Investors))


# Perform basic arithmetic operations
addition = Funding_Rounds + Number_of_Investors
subtraction = Funding_Rounds - Number_of_Investors
multiplication = Funding_Rounds * Number_of_Investors
division = Funding_Rounds / Number_of_Investors

print(" Zameen.com Long - lat - Addition:", addition)
print(" Zameen.com Long - lat - Subtraction:", subtraction)
print(" Zameen.com Long - lat - Multiplication:", multiplication)
print(" Zameen.com Long - lat - Division:", division)


#Trigonometric Functions

Number_of_Investors = (Number_of_Investors/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(Number_of_Investors)
cosine_values = np.cos(Number_of_Investors)
tangent_values = np.cos(Number_of_Investors)

print("Zameen.com Price - div - pie  - Sine values:", sine_values)
print("Zameen.com Price - div - pie Cosine values:", cosine_values)
print("Zameen.com Price - div - pie Tangent values:", tangent_values)

print("Zameen.com Price - div - pie  - Exponential values:", np.exp(Number_of_Investors))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(Number_of_Investors)
log10_array = np.log10(Number_of_Investors)

print("Zameen.com Price - div - pie  - Natural logarithm values:", log_array)
print("Zameen.com Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(Number_of_Investors)
print("Zameen.com Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(Number_of_Investors)
print("Zameen.com Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(Number_of_Investors)
print("Zameen.com Price - div - pie   -Hyperbolic Tangent values:", tanh_values)