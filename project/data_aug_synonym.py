from nltk.corpus import wordnet 
import random
synonyms = []

# for syn in wordnet.synsets("moving."):
# 	for lemma in syn.lemmas():
# 		synonyms.append(lemma.name())

# print(list(set(synonyms)))


fname = 'project/shakespeare.txt'
with open(fname, 'r') as f:
	file_data = f.read()


# file_data = file_data[:10000]

lines = file_data.split("\n")
print(len(lines))
new_file_data = []
for line in lines:
	new_line = ""
	words = line.split(" ")
	for j, word in enumerate(words):
		synonyms = []
		
		rand_num = random.random()

		if rand_num > 0.5:
			# print("synoym")
			if word == "It" or word == "it" or word == "I":
				new_line += word + " "
				continue


			special_char_list = [".", ",", "!", "?", ":", ";"] # "'", "\"", "(", ")", "[", "]", "{", "}"]
			pop_char = -1
			for e, char in enumerate(special_char_list):
				if char in word:
					char_idx = word.index(char)
					word = word[:char_idx]
					pop_char = e
					break

			for syn in wordnet.synsets(word):
				for lemma in syn.lemmas():
					synonyms.append(lemma.name())
			
			synonyms = list(set(synonyms))

			# under_score = False
			dummy_synonyms = synonyms.copy()

			for s in dummy_synonyms:
				if "_" in s or "-" in s:
					synonyms.remove(s)

			# keep word
			if len(synonyms) == 0:
				if pop_char != -1:
					synonym = word + special_char_list[pop_char]
					pop_char = -1
				else:
					synonym = word
				# new_line += synonym + " "
				# continue
			
			# replace word
			else:
				if len(synonyms) > 1:
					synonyms.pop(0)

				synonym = random.choice(synonyms)

				if pop_char != -1:
					synonym = synonym + special_char_list[pop_char]
					# print(pop_char)
					pop_char = -1

				if j == 0:
					synonym = synonym.capitalize()

			new_line += synonym + " "

		else:
			new_line += word + " "
			# print("append word")
	
	new_file_data += new_line + "\n"

new_file_data = ''.join(new_file_data)


print(new_file_data)

#write new_file_data to txt file
output_file = 'project/shakespeare_modified.txt'
with open(output_file, 'w') as f:
	f.write(file_data + '\n' + new_file_data)
print(f"Modified data written to {output_file}")


