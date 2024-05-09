""" 
book fname = ’data/Goblet.txt’;
fid = fopen(book fname,’r’);
book data = fscanf(fid,’%c’);
fclose(fid);
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import string
import re

def read_file(fname):
	with open(fname, 'r') as f:
		data = f.read()
	return data

def get_word_list(data):
	# remove all non-alphanumeric characters
	data = re.sub(r'[^a-zA-Z0-9\s]', '', data)
	# split the data into words
	words = data.split()
	return words

def create_one_hot(chars):

	one_hot = np.zeros((len(char_to_int), len(chars)))
	for i, char in enumerate(chars):
		one_hot[char_to_int[char], i] = 1
	return one_hot

def synthesise_chars(h0, x0, n):
	txt = [x0]
	h = h0
	x = x0
	
	for i in range(n):
		x = create_one_hot(x)
		h = np.tanh(np.dot(U, x) + np.dot(W, h) + b)
		o = np.dot(V, h) + c
		p = np.exp(o) / np.sum(np.exp(o))
		ix = np.random.choice(range(K), p=p.ravel())
		# ix = np.argmax(p)
		x = int_to_char[ix]
		txt += x
		# print(txt)
	return txt

def cross_entropy_loss(predicted_probs, true_labels):
    # Initialize the loss
    # loss = 0.0
    
    # # Iterate over each time step
    # for t in range(len(predicted_probs)):
    #     # Compute the cross-entropy loss at time step t
    #     loss_t = -np.dot(true_labels[t], np.log(predicted_probs[t]))
    #     # Add the loss at time step t to the total loss
    #     loss += loss_t
    # return loss
	return -np.dot(true_labels, np.log(predicted_probs))

def forward_pass(inputs, targets, hprev):
	# Initialize the values of the forward pass
	xs, hs, os, ps = {}, {}, {}, {}
	hs[-1] = np.copy(hprev)
	loss = 0.0
	
	# Iterate over each time step
	for t in range(inputs.shape[1]):
		# Create a one-hot vector for the input at time step t
	
		# Compute the hidden state at time step t
		hs[t] = np.tanh(np.dot(U, inputs[:,t]).reshape(-1,1) + np.dot(W, hs[t - 1]) + b)
		# a_test = np.dot(U, inputs[:,t]).reshape(-1, 1)
		# b_test = np.dot(W, hs[t - 1])
		# ab_test  = a_test + b_test
		# ab_tan = np.tanh(ab_test)
		# Compute the output at time step t
		os[t] = np.dot(V, hs[t]) + c
		
		# Compute the softmax probability at time step t
		ps[t] = np.exp(os[t]) / np.sum(np.exp(os[t]))
		
		# Update the loss by adding the cross-entropy loss at time step t
		loss += cross_entropy_loss(ps[t], targets[:,t])

	return {"xs": xs, "hs": hs, "os": os, "ps": ps, "loss": loss}

def backward_pass(inputs, targets, hs, ps):
	# Initialize the gradients with zero values
	dU, dW, dV = np.zeros_like(U), np.zeros_like(W), np.zeros_like(V)
	db, dc = np.zeros_like(b), np.zeros_like(c)
	dhnext = np.zeros_like(hs[0])
	
	# Iterate over each time step in reverse order
	for t in reversed(range(inputs.shape[1])):
		# Compute the gradient of the output layer w.r.t. the cross-entropy loss
		do = np.copy(ps[t])
		do = do - targets[:,t].reshape(-1,1)
		
		# Compute the gradient of the output layer w.r.t. the output at time step t
		dV += np.dot(do, hs[t].T)
		dc += do
		
		# Compute the gradient of the hidden layer w.r.t. the output at time step t
		dh = np.dot(V.T, do) + dhnext
		
		# Compute the gradient of the hidden layer w.r.t. the hidden state at time step t
		dhraw = (1 - hs[t] ** 2) * dh
		
		# Compute the gradient of the hidden layer w.r.t. the weights and biases
		dU += np.dot(dhraw, inputs[:,t].T)
		dW += np.dot(dhraw, hs[t - 1].T)
		db += dhraw
		
		# Update the gradient of the hidden layer w.r.t. the hidden state at time step t
		dhnext = np.dot(W.T, dhraw)
	
	return dU, dW, dV, db, dc

data = read_file('assignment4/goblet_book.txt')

#creat a character to integer mapping
char_to_int = dict((c, i) for i, c in enumerate(sorted(set(data))))
int_to_char = dict((i, c) for i, c in enumerate(sorted(set(data))))

m = 100
K = len(char_to_int)

eta = 0.1
seq_length = 25

#initialize the weights
b = np.zeros((m, 1))
c = np.zeros((K, 1))

U = np.random.randn(m, K) * 0.01
W = np.random.randn(m, m) * 0.01
V = np.random.randn(K, m) * 0.01

h0 = np.zeros((m, 1))
x0 = 'H'
n = 10
print(synthesise_chars(h0, x0, n))

x_chars = data[:seq_length]
y_chars = data[1:seq_length + 1]


# Convert the characters in the input and target sequences to integers
inputs = create_one_hot(x_chars)
targets = create_one_hot(y_chars)

res = forward_pass(inputs, targets, h0)

dU, dW, dV, db, dc = backward_pass(inputs, targets, res["hs"], res["ps"])

