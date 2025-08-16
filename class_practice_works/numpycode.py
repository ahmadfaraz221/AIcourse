import numpy as np

address, latitude , longitude, name = np.genfromtxt('FastFoodRestaurants.csv',
                                                     delimiter=',', usecols=(0,4,5,6), unpack=True, dtype=None,skip_header=1,invalid_raise=False)

print(address)
print(latitude)
print(longitude)
print(name)

print("fastfoodresturent longitude mean: ",np.mean(latitude))
print("fastfoodresturent longitude average: ", np.average(latitude))
print("fastfoodresturent longitude std",np.std(latitude))
print("fastfoodresturent longitude",np.median(latitude))
print("fastfoodresturent longitude percentile - 25: " , np.percentile(longitude,25))
print("fastfoodresturent longitude percentile  - 75: " , np.percentile(longitude,75))
print("fastfoodresturent longitude percentile  - 3: " , np.percentile(longitude,3))
print("fastfoodresturent longitude min : " , np.min(longitude))
print("fastfoodresturent longitude max : " , np.max(longitude))

# Zameen.com price  - maths operations
print("Zameen.com Price square: " , np.square(longitude))
print("Zameen.com Price sqrt: " , np.sqrt(longitude))
print("Zameen.com Price pow: " , np.power(longitude,longitude))
print("Zameen.com Price abs: " , np.abs(longitude))

# Perform basic arithmetic operations
addition = longitude + latitude
subtraction = longitude - latitude
multiplication = longitude * latitude
division = longitude / latitude

print(" Zameen.com Long - lat - Addition:", addition)
print(" Zameen.com Long - lat - Subtraction:", subtraction)
print(" Zameen.com Long - lat - Multiplication:", multiplication)
print(" Zameen.com Long - lat - Division:", division)