#total marks and perecntage of five subject
subject1 = int(input("enter marks of subject 1: "))
subject2 = int(input("enter marks of subject 2: "))
subject3 = int(input("enter marks of subject 3: "))
subject4 = int(input("enter marks of subject 4: "))
subject5 = int(input("enter marks of subject 5: "))
maximum_marks = 500
total_marks = subject1 + subject2 + subject3 + subject4 + subject5
print("/ntotal marks:",total_marks)
averag = (subject1 + subject2 + subject3 + subject4 + subject5) /5 
print("average marks:",averag)
percentage = (total_marks / maximum_marks) * 100
 
print("percentage:",percentage,"%")