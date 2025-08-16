#convert to mintes hours and mintes

total_minutes = int(input("Enter total minutes: "))
hours = total_minutes // 60
minutes = total_minutes % 60
print(total_minutes, "minutes =", hours, "hours and", minutes, "minutes")