import numpy as np
address, latitude , longitude, name = np.genfromtxt('FastFoodRestaurants.csv',
                                                     delimiter=',', usecols=(0,4,5,6), unpack=True, dtype=None,skip_header=1,invalid_raise=False)

print(address)
print(latitude)
print(longitude)
print(name)


print("fastfoodresturent longitude mean: ",np.mean(longitude))
print("fastfoodresturent longitude average: ", np.average(longitude))
print("fastfoodresturent longitude std",np.std(longitude))
print("fastfoodresturent longitude",np.median(longitude))
print("")
print()
