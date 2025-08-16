# list all words, with vowel in it
list_in_words = """The year was 2147. Humanity had long since ceded control of its daily functions to artificial 
intelligence. Cities operated like clockwork, transportation was seamless, and even emotions 
could be regulated by neural implants..."""
print(type(list_in_words))
print(list_in_words)

words = list_in_words.split()
vowel_words = [word for word in words if any(v in word.lower() for v in 'aeiou')]
print("Words with vowels:\n", vowel_words)

