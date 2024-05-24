
import random
import nlpaug.augmenter.char as nac


fname = 'project/shakespeare.txt'
with open(fname, 'r') as f:
	file_data = f.read()

char_list = list(set(file_data))

print(len(file_data))

#add random characters to the text
new_file_data = ""
for char in file_data:
	rand_num = random.random()
	if rand_num > 0.98:
		new_file_data += char
		new_file_data += random.choice(char_list)
	else:
		new_file_data += char

output_file = 'project/shakespeare_rand_add.txt'
with open(output_file, 'w') as f:
	f.write(new_file_data)
print(f"Modified data written to {output_file}")