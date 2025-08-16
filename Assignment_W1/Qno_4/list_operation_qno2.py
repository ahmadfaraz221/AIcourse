# print first and second large number
Numbers = list(map(int, input("Enter numbers separated by space: ").split()))
largest_number = max(Numbers)
Numbers.remove(largest_number)
second_largest = max(Numbers)

# Display results
print("The largest number in the list is:", largest_number)
print("The second largest number in the list is:", second_largest)