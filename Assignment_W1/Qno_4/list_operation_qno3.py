# Take list input from user
numbers = list(map(int, input("Enter numbers separated by space: ").split()))

# Filter even and odd numbers
even_numbers = [num for num in numbers if num % 2 == 0]
odd_numbers = [num for num in numbers if num % 2 != 0]

# Find largest even and odd if they exist
if even_numbers:
    largest_even = max(even_numbers)
    print("The largest even number is:", largest_even)
else:
    print("No even numbers in the list.")

if odd_numbers:
    largest_odd = max(odd_numbers)
    print("The largest odd number is:", largest_odd)
else:
    print("No odd numbers in the list.")
    