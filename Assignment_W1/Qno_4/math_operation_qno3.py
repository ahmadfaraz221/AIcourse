#Print an Identity Matrix 
n = int(input("Enter the size of the identity matrix: "))

# Print identity matrix
for i in range(n):
    for j in range(n):
        if i == j:
            print(1, end=" ")
        else:
            print(0, end=" ")
    print()