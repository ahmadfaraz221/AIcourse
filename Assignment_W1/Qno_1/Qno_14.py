#percentage of currect answer

total_questions = int(input("Enter total number of questions: "))
correct_answers = int(input("Enter number of correct answers: "))

percentage = (correct_answers / total_questions) * 100
print("Percentage of correct answers:", percentage, "%")