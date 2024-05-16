import numpy as np


book_fname = 'Assignment 4/data/goblet_book.txt'


with open(book_fname, 'r') as fid:
    book_data = fid.read()

book_chars = sorted(set(book_data))
char_to_ind = {char: ind for ind, char in enumerate(book_chars)}
ind_to_char = {ind: char for ind, char in enumerate(book_chars)}

print('{} unique characters'.format(len(book_chars)))
print('vocab:', book_chars)
print('char_to_ind mapping:', char_to_ind)
print('ind_to_char mapping:', ind_to_char)

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seq_length=25, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # Initialize weights
        sig = 0.01
        self.U = np.random.randn(hidden_size, input_size) * sig
        self.W = np.random.randn(hidden_size, hidden_size) * sig
        self.V = np.random.randn(output_size, hidden_size) * sig
        
        # Initialize biases
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((output_size, 1))
        
        # Initialize AdaGrad memory parameters
        self.mU = np.zeros_like(self.U)
        self.mW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.mb = np.zeros_like(self.b)
        self.mc = np.zeros_like(self.c)
        
        # Small constant for numerical stability
        self.eps = 1e-8
        
    def forward(self, x, h_prev):
        # Compute activation
        a = np.dot(self.W, h_prev) + np.dot(self.U, x) + self.b
        # Compute hidden state
        h_next = np.tanh(a)
        # Compute output
        o = np.dot(self.V, h_next) + self.c
        # Compute probabilities using softmax
        p = np.exp(o) / np.sum(np.exp(o), axis=0)
        return p, h_next
    
    def synthesize_sequence(self, h0, x0, n, ind_to_char):
        # Initialize variables
        x_next = x0
        h_prev = h0
        synthesized_sequence = []
        
        for t in range(n):
            # Forward pass
            a = np.dot(self.W, h_prev) + np.dot(self.U, x_next) + self.b
            h_next = np.tanh(a)
            o = np.dot(self.V, h_next) + self.c
            p = np.exp(o) / np.sum(np.exp(o))
            
            # Sample a label based on the output probability scores
            sampled_index = np.random.choice(range(self.output_size), p=p.ravel())
            sampled_char = ind_to_char[sampled_index]
            
            # Store sampled character in synthesized sequence
            synthesized_sequence.append(sampled_char)
            
            # Update x_next for next iteration
            x_next = np.zeros((self.input_size, 1))
            x_next[sampled_index] = 1
            
            # Update h_prev for next iteration
            h_prev = h_next
            
        return synthesized_sequence
    
    def train(self, x_train, y_train, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(x_train, y_train):
                h_prev = np.zeros((self.hidden_size, 1))
                loss = 0
                for t in range(self.seq_length):
                    # Forward pass
                    p, h_prev = self.forward(x[t], h_prev)
                    # Compute loss
                    loss += -np.sum(y[t] * np.log(p))
                    # Backward pass
                    dW, dU, dV, db, dc = self.backward(x[t], p, y[t], h_prev, h_prev)
                    
                    # Update AdaGrad memory
                    self.mW += dW**2
                    self.mU += dU**2
                    self.mV += dV**2
                    self.mb += db**2
                    self.mc += dc**2
                    
                    # Update weights and biases with AdaGrad
                    self.W -= (self.learning_rate / np.sqrt(self.mW + self.eps)) * dW
                    self.U -= (self.learning_rate / np.sqrt(self.mU + self.eps)) * dU
                    self.V -= (self.learning_rate / np.sqrt(self.mV + self.eps)) * dV
                    self.b -= (self.learning_rate / np.sqrt(self.mb + self.eps)) * db
                    self.c -= (self.learning_rate / np.sqrt(self.mc + self.eps)) * dc
                
                # Print average loss per sequence
                total_loss += loss / self.seq_length
            # Print average loss per epoch
            print(f'Epoch {epoch+1}/{epochs}, Average Loss: {total_loss / len(x_train)}')

# Example usage:
# Define input sequences and labels
x_train = [...]  # List of input sequences
y_train = [...]  # List of corresponding labels

input_size = 80
hidden_size = 100
output_size = 80
seq_length = 25
learning_rate = 0.1

rnn = SimpleRNN(input_size, hidden_size, output_size, seq_length, learning_rate)
#rnn.train(x_train, y_train)

# Synthesize a sequence
h0 = np.zeros((hidden_size, 1))
x0 = np.zeros((input_size, 1))
x0[0] = 1  # Assuming the first input is a full-stop
n = 100  # Length of the sequence to generate
synthesized_sequence = rnn.synthesize_sequence
print(synthesized_sequence)