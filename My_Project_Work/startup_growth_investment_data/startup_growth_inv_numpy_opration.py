import numpy as np

# Load data into one variable
data = np.genfromtxt(
    r'C:\Users\NS COMPUTERS\Documents\GitHub\AIcourse\My_Project_Work\startup_growth_investment_data\startup_growth_investment_data.csv',
    delimiter=',',
    usecols=(1, 2, 5, 6),  # Adjust based on actual CSV
    dtype=float,           # or 'float' if all 4 columns are numeric
    skip_header=1,
    invalid_raise=False
)

# Ensure data is not empty
if data.size == 0:
    raise ValueError("⚠️ No data was loaded. Check file path, format, and column indices.")

# Make sure it's at least 2D
data = np.atleast_2d(data)

# Now slice columns
industry = data[:, 0]
founding_round = data[:, 1]
investment_amount_USD = data[:, 2]
number_of_invester = data[:, 3]

print(industry)

# Zameen.com price  - statistics operations
print("startup_growth_investment.com mean: " , np.mean(number_of_invester))
print("startup_growth_investment.com average: " , np.average(number_of_invester))
print("startup_growth_investment.com std: " , np.std(number_of_invester))
print(" mod: " , np.median(number_of_invester))
print("startup_growth_investment.com percentile - 25: " , np.percentile(number_of_invester,25))
print("startup_growth_investment.com percentile  - 75: " , np.percentile(number_of_invester,75))
print("startup_growth_investment.com percentile  - 3: " , np.percentile(investment_amount_USD,3))
print("startup_growth_investment.com min : " , np.min(number_of_invester))
print("startup_growth_investment.com max : " , np.max(number_of_invester))

# Zameen.com price  - maths operations
print("startup_growth_investment.com square: " , np.square(number_of_invester))
print("startup_growth_investment.com sqrt: " , np.sqrt(number_of_invester))
print("startup_growth_investment.com pow: " , np.power(number_of_invester,number_of_invester))
print("startup_growth_investment.com abs: " , np.abs(number_of_invester))



# Perform basic arithmetic operations
addition = industry + founding_round
subtraction = industry - founding_round
multiplication = industry * founding_round
division = industry / founding_round

print(" startup_growth_investment.com Long - lat - Addition:", addition)
print(" startup_growth_investment.com Long - lat - Subtraction:", subtraction)
print(" startup_growth_investment.com Long - lat - Multiplication:", multiplication)
print(" startup_growth_investment.com Long - lat - Division:", division)


#Trigonometric Functions

pricePie = (number_of_invester/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("startup_growth_investment.com - div - pie  - Sine values:", sine_values)
print("startup_growth_investment.com Price - div - pie Cosine values:", cosine_values)
print("startup_growth_investment.com Price - div - pie Tangent values:", tangent_values)

print("startup_growth_investment.com Price - div - pie  - Exponential values:", np.exp(pricePie))


# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("startup_growth_investment.com Price - div - pie  - Natural logarithm values:", log_array)
print("startup_growth_investment.com Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("startup_growth_investment.com Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)
print("startup_growth_investment.com Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("startup_growth_investment.com - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("startup_growth_investment.com - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("startup_growth_investment.com Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Zameen.com Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([industry,
                  founding_round])

