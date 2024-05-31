import numpy as np
import matplotlib.pyplot as plt
import random
import string
import re



class DataProcessing:
	def __init__(self) -> None:
		fname = 'assignment4/goblet_book.txt'
		with open(fname, 'r') as f:
			file_data = f.read()

		self.file_data = file_data
		self.book_length = len(file_data)
		self.char_to_int = dict((c, i) for i, c in enumerate(sorted(set(file_data))))
		self.int_to_char = dict((i, c) for i, c in enumerate(sorted(set(file_data))))

		self.num_chars = len(set(file_data))
		print(self.num_chars)

		# with open(fname) as f:
		# 	lines = f.readlines()

		# self.ex_sentences = [x.lower().strip() for x in lines]

		# for d in self.ex_sentences:
		# 	if len(d) < 2:
		# 		self.ex_sentences.remove(d)

	def decode_sequence(self, index):
		input_sequence = self.file_data[index:index + 25]
		decoded_input_sentence = [self.char_to_int[char] for char in input_sequence]

		output_sequence = self.file_data[index+1:index + 25 + 1]
		decoded_output_sentence = [self.char_to_int[char] for char in output_sequence]

		return decoded_input_sentence, decoded_output_sentence

	def create_one_hot(self, inp, outp, K=80):
		# create a one-hot vector
		x = np.zeros((25, 80))
		y = np.zeros((25, 80))
		for i, elm in enumerate(inp):
			# one_hot = np.zeros((K, 1))
			x[i, elm] = 1

		for i, elm in enumerate(outp):
			# one_hot = np.zeros((K, 1))
			y[i, elm] = 1

		return x, y

def get_word_list(data):
	# remove all non-alphanumeric characters
	data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
	# split the data into words
	words = data.split()
	return words

# def synthesise_chars(h0, x0, n):
# 	txt = [x0]
# 	h = h0
# 	x = x0
	
# 	for i in range(n):
# 		x = create_one_hot(x)
# 		h = np.tanh(np.dot(U, x) + np.dot(W, h) + b)
# 		o = np.dot(V, h) + c
# 		p = np.exp(o) / np.sum(np.exp(o))
# 		ix = np.random.choice(range(K), p=p.ravel())
# 		# ix = np.argmax(p)
# 		x = int_to_char[ix]
# 		txt += x
# 		# print(txt)
# 	return txt

class RNN(DataProcessing):
	def __init__(self, processed_data, m, eta, seq_length) -> None:
		self.m = m
		self.eta = eta
		self.seq_length = seq_length
		self.processed_data = processed_data
		self.K = self.processed_data.num_chars

		#initialize the weights
		self.b = np.zeros((m, 1))
		self.c = np.zeros((self.K, 1))

		self.U = np.random.randn(m, self.K) * 0.01
		self.W = np.random.randn(m, m) * 0.01
		self.V = np.random.randn(self.K, m) * 0.01

		self.dU, self.dW, self.dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
		self.db, self.dc = np.zeros_like(self.b), np.zeros_like(self.c)

		self.accum_dU, self.accum_dW, self.accum_dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
		self.accum_db, self.accum_dc = np.zeros_like(self.b), np.zeros_like(self.c)

		self.curr_h = np.zeros((self.seq_length, self.m))
		self.curr_a = np.zeros((self.seq_length, self.m))

		self.h0 = np.zeros((m, 1))
		self.x0 = 'H'
		print("self.K", self.K)

	def forward_pass(self, inputs, hprev):
		# Initialize the values of the forward pass
		xs, a_s, hs, os, ps = {},{}, {}, {}, {}
		hs[-1] = np.copy(hprev)
		loss = 0.0
		
		# Iterate over each time step
		for t in range(inputs.shape[0]):

			# Create a one-hot vector for the input at time step t
			xs[t] = inputs[t].reshape(-1,1)
			
			# Compute the hidden state at time step t
			a_s[t] = np.dot(self.U, xs[t]).reshape(-1,1) + np.dot(self.W, hs[t - 1]) + self.b
			hs[t] = np.tanh(a_s[t])	
				
			os[t] = np.dot(self.V, hs[t]) + self.c
			
			# Compute the softmax probability at time step t
			ps[t] = self.softmax(os[t])

		return {"xs": xs, "hs": hs, "os": os, "ps": ps, "loss": loss}
	
	def softmax(self, x):
		exp_x = np.exp(x - np.max(x))
		return exp_x / np.sum(exp_x, axis=0, keepdims=True)

	def cross_entropy_loss(self, predicted_probs, true_labels):

		loss = 0.0
		# Compute the cross-entropy loss
		for elm in range(len(true_labels)):
			loss += -np.dot(true_labels[elm], np.log(predicted_probs[elm] + 1e-8))
		return float(loss)

	def backward_pass(self, inputs, targets,xs, hs, ps):
		# Initialize the gradients
		dU, dW, dV = np.zeros_like(self.U), np.zeros_like(self.W), np.zeros_like(self.V)
		db, dc = np.zeros_like(self.b), np.zeros_like(self.c)
		
		# Initialize the hidden state
		dhnext = np.zeros_like(hs[0])
		
		# Iterate over each time step in reverse
		for t in reversed(range(inputs.shape[0])):
			# Compute the gradient of the output layer
			do = np.copy(ps[t])
			do = do - targets[t].reshape(-1,1)
			
			# Compute the gradient of the output layer with respect to the weights V
			dV += np.dot(do, hs[t].T)
		
			# Compute the gradient of the output layer with respect to the bias c
			dc += do
			
			# Compute the gradient of the hidden layer
			dh = np.dot(self.V.T, do) + dhnext
			
			# Compute the gradient of the hidden layer with respect to the hidden state
			dhraw = (1 - hs[t] * hs[t]) * dh
			
			# Compute the gradient of the hidden layer with respect to the weights W
			dW += np.dot(dhraw, hs[t - 1].T)
			
			# Compute the gradient of the hidden layer with respect to the weights U
			dU += np.dot(dhraw, xs[t].T)
			
			# Compute the gradient of the hidden layer with respect to the bias b
			db += dhraw
			
			# Update the gradient of the hidden state
			# dhnext = np.dot(self.W.T, dhraw)
			dhnext = dhraw
		
		for grad in [dU, dW, dV, db, dc]:
			np.clip(grad, -5, 5, out=grad)

		return dU, dW, dV, db, dc

	def adagrad(self, dU, dW, dV, db, dc, beta1=0, epsilon=1e-8):

		self.accum_dU += dU**2
		self.accum_dW += dW**2
		self.accum_dV += dV**2
		self.accum_db += db**2
		self.accum_dc += dc**2

		self.U -= self.eta * dU / (np.sqrt(self.accum_dU) + epsilon)
		self.W -= self.eta * dW / (np.sqrt(self.accum_dW) + epsilon)
		self.V -= self.eta * dV / (np.sqrt(self.accum_dV) + epsilon)
		self.b -= self.eta * db / (np.sqrt(self.accum_db) + epsilon)
		self.c -= self.eta * dc / (np.sqrt(self.accum_dc) + epsilon)

	def generate_text(self, x_gen, hprev_gen, n):
		xt = x_gen.reshape(-1, 1)
		ht = hprev_gen.reshape(-1, 1)
		output = []
		for i in range(n):
			# compute the hidden state
			ht = np.tanh(np.dot(self.U, xt) + np.dot(self.W, ht) + self.b)

			# compute the output probabilities
			y = self.softmax(np.dot(self.V, ht) + self.c)

			# sample the next character from the output probabilities
			idx = np.random.choice(range(self.K), p=y.ravel())

			# set the input for the next time step
			xt = np.zeros((self.K, 1))
			xt[idx] = 1

			# store the sampled character index in the list
			output.append(idx)

		return output

	def train(self):
		# Initialize the hidden state
		hprev = np.zeros((self.m, 1))
		
		# Initialize the loss
		loss = 0.0
		smooth_loss = None
		# smooth_loss = -np.log(1.0 / self.processed_data.num_chars) * self.seq_length
		epoch = 10
		# Iterate over each epoch
		for ep in range(epoch):
			# Iterate over each sentence
			for e in range(0, self.processed_data.book_length - self.seq_length - 1, 25):

				# Convert the characters in the input and target sequences to integers
				inputs, targets = self.processed_data.decode_sequence(e)
				
				X, Y = self.processed_data.create_one_hot(inputs, targets)

				res = self.forward_pass(X, hprev)
				hprev = res["hs"][24]
				# # Perform the backward pass
				dU, dW, dV, db, dc = self.backward_pass(X, Y, res["xs"], res["hs"], res["ps"])
				
				# self.adagrad(dU, dW, dV, db, dc)

				loss = self.cross_entropy_loss(res["ps"], Y)
				
				self.adagrad(dU, dW, dV, db, dc)

				if smooth_loss is None:
					smooth_loss = loss
				else:
					smooth_loss = smooth_loss * 0.999 + loss * 0.001

				if e % 10000 == 0:
					print("Epoch: ", ep, ", step: ", e/25, ", loss: ", smooth_loss)
					output = self.generate_text(X[24], res["hs"][24], 100)
					st = ''.join([self.processed_data.int_to_char[idx] for idx in output])
					print(st, "\n")
			# 	# Update the hidden state
			# 	hprev = res["hs"][len(inputs) - 1]
				
			# 	# Update the loss
			# 	loss += res["loss"]
			
			# # Print the loss after each epoch
			# print(f"Epoch {epoch} Loss: {loss}")


processed_data = DataProcessing()
rnn = RNN(processed_data, 100, 0.1, 25)
rnn.train()