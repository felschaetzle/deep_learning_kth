import numpy as np


book_fname = 'Assignment 4/data/goblet_book.txt'

class DataGenerator:
    
    def __init__(self, path):
        self.path = path
        
        # Read in data from file and convert to lowercase
        with open(path, 'r') as f:
            data = f.read()
        
        # Create list of unique characters in the data
        self.chars = list(set(data))
        
        # Create dictionaries mapping characters to and from their index in the list of unique characters
        self.char_to_idx = {ch: i for (i, ch) in enumerate(self.chars)}
        self.idx_to_char = {i: ch for (i, ch) in enumerate(self.chars)}
        
        # Set the size of the vocabulary (i.e. number of unique characters)
        self.vocab_size = len(self.chars)
        
        # Read in examples from file and convert to lowercase, removing leading/trailing white space
        with open(path) as f:
            examples = f.readlines()
        self.examples = [x.lower().strip() for x in examples]
 
    def generate_example(self, idx):

        example_chars = self.examples[idx]
        
        # Convert the characters in the example to their corresponding indices in the list of unique characters
        example_char_idx = [self.char_to_idx[char] for char in example_chars]
        
        # Add newline character as the first character in the input array, and as the last character in the output array
        X = [self.char_to_idx['\n']] + example_char_idx
        Y = example_char_idx + [self.char_to_idx['\n']]
        
        return np.array(X), np.array(Y)

class RNN:
   
    def __init__(self, hidden_size, data_generator, sequence_length, learning_rate):
        
        # hyper parameters
        self.hidden_size = hidden_size
        self.data_generator = data_generator
        self.vocab_size = self.data_generator.vocab_size
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.X = None

        # model parameters
        sig = 0.01
        self.Wax = np.random.randn(hidden_size, self.vocab_size) * sig
        self.Waa = np.random.randn(hidden_size, hidden_size) * sig
        self.Wya = np.random.randn(self.vocab_size, hidden_size) * sig

        self.ba = np.zeros((hidden_size, 1))  
        self.by = np.zeros((self.vocab_size, 1))
        
        # Initialize gradients
        self.dWax, self.dWaa, self.dWya = np.zeros_like(self.Wax), np.zeros_like(self.Waa), np.zeros_like(self.Wya)
        self.dba, self.dby = np.zeros_like(self.ba), np.zeros_like(self.by)
        
        # parameter update with AdamW
        self.mWax = np.zeros_like(self.Wax)
        self.vWax = np.zeros_like(self.Wax)
        self.mWaa = np.zeros_like(self.Waa)
        self.vWaa = np.zeros_like(self.Waa)
        self.mWya = np.zeros_like(self.Wya)
        self.vWya = np.zeros_like(self.Wya)
        self.mba = np.zeros_like(self.ba)
        self.vba = np.zeros_like(self.ba)
        self.mby = np.zeros_like(self.by)
        self.vby = np.zeros_like(self.by)

    def softmax(self, x):
        # shift the input to prevent overflow when computing the exponentials
        x = x - np.max(x)
        # compute the exponentials of the shifted input
        p = np.exp(x)
        # normalize the exponentials by dividing by their sum
        return p / np.sum(p)

    def forward(self, X, a_prev):
        # Initialize dictionaries to store activations and output probabilities.
        x, a, y_pred = {}, {}, {}

        # Store the input data in the class variable for later use in the backward pass.
        self.X = X

        # Set the initial activation to the previous activation.
        a[-1] = np.copy(a_prev)
        # iterate over each time step in the input sequence
        for t in range(len(self.X)): 
            # get the input at the current time step
            x[t] = np.zeros((self.vocab_size,1)) 
            if (self.X[t] != None):
                x[t][self.X[t]] = 1
            # compute the hidden activation at the current time step
            a[t] = np.tanh(np.dot(self.Wax, x[t]) + np.dot(self.Waa, a[t - 1]) + self.ba)
            # compute the output probabilities at the current time step
            y_pred[t] = self.softmax(np.dot(self.Wya, a[t]) + self.by)
            # add an extra dimension to X to make it compatible with the shape of the input to the backward pass
         # return the input, hidden activations, and output probabilities at each time step
        return x, a, y_pred 
    
    def backward(self,x, a, y_preds, targets):
        # Initialize derivative of hidden state for the last time-step
        da_next = np.zeros_like(a[0])

        # Loop through the input sequence backwards
        for t in reversed(range(len(self.X))):
            # Calculate derivative of output probability vector
            dy_preds = np.copy(y_preds[t])
            dy_preds[targets[t]] -= 1

            # Calculate derivative of hidden state
            da = np.dot(self.Waa.T, da_next) + np.dot(self.Wya.T, dy_preds)
            dtanh = (1 - np.power(a[t], 2))
            da_unactivated = dtanh * da

            # Calculate gradients
            self.dba += da_unactivated
            self.dWax += np.dot(da_unactivated, x[t].T)
            self.dWaa += np.dot(da_unactivated, a[t - 1].T)

            # Update derivative of hidden state for the next iteration
            da_next = da_unactivated

            # Calculate gradient for output weight matrix
            self.dWya += np.dot(dy_preds, a[t].T)

            # clip gradients to avoid exploding gradients
            for grad in [self.dWax, self.dWaa, self.dWya, self.dba, self.dby]:
                np.clip(grad, -1, 1, out=grad)
 
    def loss(self, y_preds, targets):
        # calculate cross-entropy loss
        return sum(-np.log(y_preds[t][targets[t], 0]) for t in range(len(self.X)))
    
    def adamw(self, beta1=0.9, beta2=0.999, epsilon=1e-8, L2_reg=1e-4):

        # AdamW update for Wax
        self.mWax = beta1 * self.mWax + (1 - beta1) * self.dWax
        self.vWax = beta2 * self.vWax + (1 - beta2) * np.square(self.dWax)
        m_hat = self.mWax / (1 - beta1)
        v_hat = self.vWax / (1 - beta2)
        self.Wax -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wax)

        # AdamW update for Waa
        self.mWaa = beta1 * self.mWaa + (1 - beta1) * self.dWaa
        self.vWaa = beta2 * self.vWaa + (1 - beta2) * np.square(self.dWaa)
        m_hat = self.mWaa / (1 - beta1)
        v_hat = self.vWaa / (1 - beta2)
        self.Waa -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Waa)

        # AdamW update for Wya
        self.mWya = beta1 * self.mWya + (1 - beta1) * self.dWya
        self.vWya = beta2 * self.vWya + (1 - beta2) * np.square(self.dWya)
        m_hat = self.mWya / (1 - beta1)
        v_hat = self.vWya / (1 - beta2)
        self.Wya -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.Wya)

        # AdamW update for ba
        self.mba = beta1 * self.mba + (1 - beta1) * self.dba
        self.vba = beta2 * self.vba + (1 - beta2) * np.square(self.dba)
        m_hat = self.mba / (1 - beta1)
        v_hat = self.vba / (1 - beta2)
        self.ba -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + L2_reg * self.ba)

        # AdamW update for by
        self.mby = beta1 * self.mby + (1 - beta1) * self.dby
        self.vby = beta2 * self.vby + (1 - beta2) * np.square(self.dby)
    
    def sample(self):

        # initialize input and hidden state
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        # create an empty list to store the generated character indices
        indices = []

        # idx is a flag to detect a newline character, initialize it to -1
        idx = -1

        # generate sequence of characters
        counter = 0
        max_chars = 50 # maximum number of characters to generate
        newline_character = self.data_generator.char_to_idx['\n'] # the newline character

        while (idx != newline_character and counter != max_chars):
            # compute the hidden state
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)

            # compute the output probabilities
            y = self.softmax(np.dot(self.Wya, a) + self.by)

            # sample the next character from the output probabilities
            idx = np.random.choice(list(range(self.vocab_size)), p=y.ravel())

            # set the input for the next time step
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1

            # store the sampled character index in the list
            indices.append(idx)

            # update the previous hidden state
            a_prev = a

            # increment the counter
            counter += 1

        # return the list of sampled character indices
        return indices

        
    def train(self, generated_names=5):

        iter_num = 0
        threshold = 5 # stopping criterion for training
        smooth_loss = -np.log(1.0 / self.data_generator.vocab_size) * self.sequence_length  # initialize loss

        while (smooth_loss > threshold):
            a_prev = np.zeros((self.hidden_size, 1))
            idx = iter_num % self.vocab_size
            # get a batch of inputs and targets
            inputs, targets = self.data_generator.generate_example(idx)

            # forward pass
            x, a, y_pred  = self.forward(inputs, a_prev)

            # backward pass
            self.backward(x, a, y_pred, targets)

            # calculate and update loss
            loss = self.loss(y_pred, targets)
            self.adamw()
            smooth_loss = smooth_loss * 0.999 + loss * 0.001

            # update previous hidden state for the next batch
            a_prev = a[len(self.X) - 1]
            # print progress every 500 iterations
            if iter_num % 500 == 0:
                print("\n\niter :%d, loss:%f\n" % (iter_num, smooth_loss))
                for i in range(generated_names):
                    sample_idx = self.sample()
                    txt = ''.join(self.data_generator.idx_to_char[idx] for idx in sample_idx)
                    txt = txt.title()  # capitalize first character 
                    print ('%s' % (txt, ), end='')
            iter_num += 1
    
    def predict(self, start):

        # Initialize input vector and previous hidden state
        x = np.zeros((self.vocab_size, 1))
        a_prev = np.zeros((self.hidden_size, 1))

        # Convert start sequence to indices
        chars = [ch for ch in start]
        idxes = []
        for i in range(len(chars)):
            idx = self.data_generator.char_to_idx[chars[i]]
            x[idx] = 1
            idxes.append(idx)

        # Generate sequence
        max_chars = 50  # maximum number of characters to generate
        newline_character = self.data_generator.char_to_idx['\n']  # the newline character
        counter = 0
        while (idx != newline_character and counter != max_chars):
            # Compute next hidden state and predicted character
            a = np.tanh(np.dot(self.Wax, x) + np.dot(self.Waa, a_prev) + self.ba)
            y_pred = self.softmax(np.dot(self.Wya, a) + self.by)
            idx = np.random.choice(range(self.vocab_size), p=y_pred.ravel())

            # Update input vector, previous hidden state, and indices
            x = np.zeros((self.vocab_size, 1))
            x[idx] = 1
            a_prev = a
            idxes.append(idx)
            counter += 1

        # Convert indices to characters and concatenate into a string
        txt = ''.join(self.data_generator.idx_to_char[i] for i in idxes)

        # Remove newline character if it exists at the end of the generated sequence
        if txt[-1] == '\n':
            txt = txt[:-1]

        return txt



data_generator = DataGenerator('Assignment 4/data/goblet_book.txt')
rnn = RNN(hidden_size=100,data_generator=data_generator, sequence_length=25, learning_rate=0.1)
rnn.train()

rnn.predict("Sommer")

rnn.predict("GE")
