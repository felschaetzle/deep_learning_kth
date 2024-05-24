
import random


fname = 'project/shakespeare.txt'
with open(fname, 'r') as f:
	file_data = f.read()

lines = file_data.split("\n")
# print(len(lines))
new_file_data = []

# delete a word from one line with 50% probability
for line in lines:
	new_line = ""
	words = line.split(" ")
	line_ran_num = random.random()

	if line_ran_num > 0.5:
		new_file_data += line + "\n"
		continue

	for j, word in enumerate(words):
		rand_num = random.random()

		if rand_num > 0.8:
			continue

		new_line += word + " "

	new_file_data += new_line + "\n"

new_file_data = "".join(new_file_data)
# print(new_file_data)

output_file = 'project/shakespeare_rand_delete.txt'
with open(output_file, 'w') as f:
	f.write(new_file_data)
print(f"Modified data written to {output_file}")