def is_pangram(sentence):
    sentence = sentence.lower()  # convert to lowercase
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for letter in alphabet:
        if letter not in sentence:
            return False  # if any letter is missing, not a pangram
    return True
input_string = input("Enter a sentence: ")

# Check and display result
if is_pangram(input_string):
    print("The sentence is a pangram.")
else:
    print("The sentence is not a pangram.")

    