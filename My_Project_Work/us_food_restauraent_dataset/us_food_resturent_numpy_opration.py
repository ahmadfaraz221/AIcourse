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
print("fastfoodresturent Price square: " , np.square(longitude))
print("fastfoodresturent Price sqrt: " , np.sqrt(longitude))
print("fastfoodresturent Price pow: " , np.power(longitude,longitude))
print("fastfoodresturent Price abs: " , np.abs(longitude))

# Perform basic arithmetic operations
addition = longitude + latitude
subtraction = longitude - latitude
multiplication = longitude * latitude
division = longitude / latitude

print(" fastfoodresturent Long - lat - Addition:", addition)
print(" fastfoodresturent Long - lat - Subtraction:", subtraction)
print(" fastfoodresturent Long - lat - Multiplication:", multiplication)
print(" fastfoodresturent Long - lat - Division:", division)

#Trigonometric Functions

pricePie = (longitude/np.pi) +1
# Calculate sine, cosine, and tangent
sine_values = np.sin(pricePie)
cosine_values = np.cos(pricePie)
tangent_values = np.tan(pricePie)

print("fastfoodresturent Price - div - pie  - Sine values:", sine_values)
print("fastfoodresturent Price - div - pie Cosine values:", cosine_values)
print("fastfoodresturent Price - div - pie Tangent values:", tangent_values)

print("fastfoodresturent Price - div - pie  - Exponential values:", np.exp(pricePie))

# Calculate the natural logarithm and base-10 logarithm
log_array = np.log(pricePie)
log10_array = np.log10(pricePie)

print("fastfoodresturent Price - div - pie  - Natural logarithm values:", log_array)
print("fastfoodresturent Price - div - pie  = Base-10 logarithm values:", log10_array)

#Example: Hyperbolic Sine
# Calculate the hyperbolic sine of each element
sinh_values = np.sinh(pricePie)
print("fastfoodresturent Price - div - pie   - Hyperbolic Sine values:", sinh_values)


#Hyperbolic Cosine Using cosh() Function
# Calculate the hyperbolic cosine of each element
cosh_values = np.cosh(pricePie)

print("fastfoodresturent Price - div - pie   - Hyperbolic Cosine values:", cosh_values)

#Example: Hyperbolic Tangent
# Calculate the hyperbolic tangent of each element
tanh_values = np.tanh(pricePie)
print("fastfoodresturent Price - div - pie   -Hyperbolic Tangent values:", tanh_values)

#Example: Inverse Hyperbolic Sine

# Calculate the inverse hyperbolic sine of each element
asinh_values = np.arcsinh(pricePie)
print("fastfoodresturent Price - div - pie   -Inverse Hyperbolic Sine values:", asinh_values)

#Example: Inverse Hyperbolic Cosine
# Calculate the inverse hyperbolic cosine of each element
acosh_values = np.arccosh(pricePie)
print("fastfoodresturent Price - div - pie   -Inverse Hyperbolic Cosine values:", acosh_values)


#Zameen.com Long Plus Lat - 2 dimentional arrary
D2LongLat = np.array([longitude,
                  latitude])

print ("fastfoodresturent Long Plus Lat - 2 dimentional arrary - " ,D2LongLat)

# check the dimension of array1
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - dimension" , D2LongLat.ndim) 
# Output: 2

# return total number of elements in array1
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - total number of elements" ,D2LongLat.size)
# Output: 6

# return a tuple that gives size of array in each dimension
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - gives size of array in each dimension" ,D2LongLat.shape)
# Output: (2,3)

# check the data type of array1
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - data type" ,D2LongLat.dtype) 
# Output: int64

# Splicing array
D2LongLatSlice=  D2LongLat[:1,:5]
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1,:5] " , D2LongLatSlice)
D2LongLatSlice2=  D2LongLat[:1, 4:15:4]
print("fastfoodresturent Long Plus Lat - 2 dimentional arrary - Splicing array - D2LongLat[:1, 4:15:4] " , D2LongLatSlice2)