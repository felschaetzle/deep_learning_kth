# -*- coding: utf-8 -*-
"""FinalProject_temp_nucleus

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1XjrligP4mzxxj49eDrZ0AY1827BSRVEa
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from collections import defaultdict

tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)

############################################################################################################
# Preparing Data
############################################################################################################

def read_data(lowercase_only=False):
    fname = 'project/shakespeare.txt'
    with open(fname, 'r') as f:
        file_data = f.read()
        
    corpus = file_data

    # for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(corpus)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1
    # print(word_freqs)


    alphabet = []

    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()

    vocab = alphabet.copy()

    splits = {word: [c for c in word] for word in word_freqs.keys()}

    def compute_pair_freqs(splits):
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            split = splits[word]
            if len(split) == 1:
                continue
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += freq
        return pair_freqs

    pair_freqs = compute_pair_freqs(splits)

    for i, key in enumerate(pair_freqs.keys()):
        if i >= 5:
            break

    best_pair = ""
    max_freq = None

    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq

    merges = {("Ġ", "t"): "Ġt"}
    vocab.append("Ġt")

    def merge_pair(a, b, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    def merge_pair(a, b, splits):
        for word in word_freqs:
            split = splits[word]
            if len(split) == 1:
                continue

            i = 0
            while i < len(split) - 1:
                if split[i] == a and split[i + 1] == b:
                    split = split[:i] + [a + b] + split[i + 2 :]
                else:
                    i += 1
            splits[word] = split
        return splits

    vocab_size = 201

    while len(vocab) < vocab_size:
        pair_freqs = compute_pair_freqs(splits)
        best_pair = ""
        max_freq = None
        for pair, freq in pair_freqs.items():
            if max_freq is None or max_freq < freq:
                best_pair = pair
                max_freq = freq
        splits = merge_pair(*best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        vocab.append(best_pair[0] + best_pair[1])

    vocab = list(set(vocab))
    vocab.sort(key=len, reverse=True)

    tokens_pre = [token.replace("Ġ", " ") for token in vocab]
    tokens = [token.replace("Ċ", "\n") for token in tokens_pre]

    ind_to_char = dict((i, c) for i, c in enumerate(tokens))
    char_to_ind = dict((c, i) for i, c in enumerate(tokens))

    def custom_tokenize(text, vocab):
        tokens = []
        i = 0
        while i < len(text):
            for j in range(len(vocab)):
                token = vocab[j]
                if text[i:i+len(token)] == token:
                    tokens.append(token)
                    i += len(token) - 1
                    break
            i += 1
        return tokens

    # Tokenize the corpus using the custom vocabulary
    custom_tokens = custom_tokenize(corpus, tokens)

    encoded_tokens = [char_to_ind[token] for token in custom_tokens]

    int_data = np.array(encoded_tokens)

    return int_data, ind_to_char, char_to_ind

def split_data(int_data, seq_length, val_split=0.1, test_split=0.1):
    data_length = len(int_data)
    test_size = int(data_length * test_split)
    val_size = int(data_length * val_split)

    train_data, temp_data = train_test_split(int_data, test_size=test_size + val_size, shuffle=False)
    val_data, test_data = train_test_split(temp_data, test_size=test_size, shuffle=False)

    return train_data, val_data, test_data

def create_dataloaders(train_data, val_data, seq_length, batch_size):
    def create_dataset(data):
        X, Y = [], []
        for i in range(0, len(data) - seq_length, seq_length):
            X_chars = data[i:i + seq_length]
            Y_chars = data[i + 1:i + seq_length + 1]
            if len(Y_chars) == seq_length:  # Ensure Y has the correct length
                X.append(X_chars)
                Y.append(Y_chars)
        X = torch.tensor(X, dtype=torch.long)
        Y = torch.tensor(Y, dtype=torch.long)
        return TensorDataset(X, Y)

    train_dataset = create_dataset(train_data)
    val_dataset = create_dataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

############################################################################################################
# RNN Model Definition
############################################################################################################

class RNNModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

############################################################################################################
# LSTM Definition
############################################################################################################

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # Forward propagate LSTM
        out, hidden = self.lstm(x, hidden)
        # Decode hidden state of last time step
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return h0, c0

############################################################################################################
# Helper Functions
############################################################################################################

def one_hot_encode(indices, vocab_size):
    shape = indices.shape + (vocab_size,)
    one_hot = np.zeros(shape, dtype=np.float32)
    for i in range(indices.shape[0]):
        for j in range(indices.shape[1]):
            one_hot[i, j, indices[i, j]] = 1.0
    return one_hot

def temperature_scale(logits, temperature):
    return logits / temperature

def nucleus_sampling(logits, p):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    if sorted_indices_to_remove[0]:
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
    sorted_indices_to_remove[0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = -float('Inf')
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)

def synthesize_text(model, char_to_ind, ind_to_char, start_char, hidden, n_samples, temperature, nucleus_p, use_temp_nuc):
    model.eval()
    input = torch.zeros(1, 1, len(char_to_ind))
    input[0, 0, char_to_ind[start_char]] = 1
    result = [start_char]

    with torch.no_grad():
        for _ in range(n_samples):
            output, hidden = model(input, hidden)

            if use_temp_nuc is True:
                logits = output.data.view(-1)
                logits = temperature_scale(logits, temperature)
                char_ind = nucleus_sampling(logits, nucleus_p).item()
                char = ind_to_char[char_ind]
                result.append(char)
                input.zero_()
                input[0, 0, char_ind] = 1
            else: 
                output_dist = output.data.view(-1).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                char = ind_to_char[top_i.item()]
                result.append(char)
                input.zero_()
                input[0, 0, top_i] = 1

            
    return ''.join(result)

def synthesize_text_lstm(model, char_to_ind, ind_to_char, start_char, n_samples, temperature, nucleus_p, use_temp_nuc):
    model.eval()
    input = torch.zeros(1, 1, len(char_to_ind))
    input[0, 0, char_to_ind[start_char]] = 1
    result = [start_char]

    hidden = model.init_hidden(1)

    with torch.no_grad():
        for _ in range(n_samples):
            output, hidden = model(input, hidden)

            if use_temp_nuc is True:
                logits = output.data.view(-1)
                logits = temperature_scale(logits, temperature)
                char_ind = nucleus_sampling(logits, nucleus_p).item()
                char = ind_to_char[char_ind]
                result.append(char)
                input.zero_()
                input[0, 0, char_ind] = 1
            else:
                output_dist = output.data.view(-1).exp()
                top_i = torch.multinomial(output_dist, 1)[0]
                char = ind_to_char[top_i.item()]
                result.append(char)
                input.zero_()
                input[0, 0, top_i] = 1

    return ''.join(result)

############################################################################################################
# Training Functions
############################################################################################################

def train_rnn(model, train_loader, val_loader, char_to_ind, num_epochs, learning_rate, use_temp_nuc):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    smooth_loss = None
    smooth_losses = []
    losses = []
    val_losses = []
    vocab_size = len(char_to_ind)

    for epoch in range(num_epochs):
        hidden = None
        for X, Y in train_loader:
            X_one_hot = one_hot_encode(X.numpy(), vocab_size)
            X_one_hot = torch.tensor(X_one_hot, dtype=torch.float32)

            if hidden is None or hidden.size(1) != X.size(0):
                hidden = torch.zeros(1, X.size(0), model.hidden_size)

            model.train()
            optimizer.zero_grad()
            output, hidden = model(X_one_hot, hidden.detach())
            loss = criterion(output.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            optimizer.step()

            if smooth_loss is None:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss.item()

        val_loss = evaluate_model_rnn(model, val_loader, char_to_ind)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}, Smooth Loss: {smooth_loss}, Loss: {loss.item()}, Validation Loss: {val_loss}")

        start_char = ind_to_char[np.random.choice(X[:, 0].cpu().numpy())]
        sample_text = synthesize_text(model, char_to_ind, ind_to_char, start_char, hidden[:, :1, :], 100, 1.0, 0.9, use_temp_nuc)
        print(f"Sample Text at epoch {epoch+1}:\n{sample_text}\n")

        smooth_losses.append(smooth_loss)
        losses.append(loss.item())

    start_char = ind_to_char[np.random.choice(X[:, 0].cpu().numpy())]
    sample_text = synthesize_text(model, char_to_ind, ind_to_char, start_char, hidden[:, :1, :], 500, 1.0, 0.9, use_temp_nuc)
    print(f"Sample Text at end of training:\n{sample_text}\n")

    return smooth_losses, losses, val_losses

def train_lstm(model, train_loader, val_loader, char_to_ind, num_epochs, learning_rate, use_temp_nuc):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    smooth_loss = None
    smooth_losses = []
    losses = []
    val_losses = []
    vocab_size = len(char_to_ind)

    for epoch in range(num_epochs):
        for X, Y in train_loader:
            X_one_hot = one_hot_encode(X.numpy(), vocab_size)
            X_one_hot = torch.tensor(X_one_hot, dtype=torch.float32)

            hidden = model.init_hidden(X.size(0))
            model.train()
            optimizer.zero_grad()
            output, hidden = model(X_one_hot, hidden)
            loss = criterion(output.view(-1, vocab_size), Y.view(-1))
            loss.backward()
            optimizer.step()

            if smooth_loss is None:
                smooth_loss = loss.item()
            else:
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss.item()

        val_loss = evaluate_model_lstm(model, val_loader, char_to_ind)
        val_losses.append(val_loss)


        print(f"Epoch {epoch+1}, Smooth Loss: {smooth_loss}, Loss: {loss.item()}, Validation Loss: {val_loss}")

        start_char = ind_to_char[np.random.choice(X[:, 0].cpu().numpy())]
        sample_text = synthesize_text_lstm(model, char_to_ind, ind_to_char, start_char, 100, 1.0, 0.9, use_temp_nuc)
        print(f"Sample Text at epoch {epoch+1}:\n{sample_text}\n")

        smooth_losses.append(smooth_loss)
        losses.append(loss.item())

    start_char = ind_to_char[np.random.choice(X[:, 0].cpu().numpy())]
    sample_text = synthesize_text_lstm(model, char_to_ind, ind_to_char, start_char, 500, 1.0, 0.9, use_temp_nuc)
    print(f"Sample Text at end of training:\n{sample_text}\n")

    return smooth_losses, losses, val_losses

def plot_loss(smooth_losses, losses, val_losses):
    plt.plot(smooth_losses, label='Smooth Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss vs Training Step")
    plt.legend()
    plt.savefig('smooth_and_val_loss_rnn.png')  # Adjust path if needed
    plt.show()

    plt.plot(losses, label='Training Loss')
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Training Step")
    plt.legend()
    plt.savefig('loss_rnn.png')  # Adjust path if needed
    plt.show()

############################################################################################################
# Evaluation Function
############################################################################################################

def evaluate_model_rnn(model, val_loader, char_to_ind):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    vocab_size = len(char_to_ind)

    with torch.no_grad():
        hidden = None
        for X, Y in val_loader:
            X_one_hot = one_hot_encode(X.numpy(), vocab_size)
            X_one_hot = torch.tensor(X_one_hot, dtype=torch.float32)

            if hidden is None or hidden.size(1) != X.size(0):
                hidden = torch.zeros(1, X.size(0), model.hidden_size)

            output, hidden = model(X_one_hot, hidden)
            loss = criterion(output.view(-1, vocab_size), Y.view(-1))
            losses.append(loss.item())

    average_loss = np.mean(losses)
    return average_loss

def evaluate_model_lstm(model, val_loader, char_to_ind):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    losses = []
    vocab_size = len(char_to_ind)

    with torch.no_grad():
        for X, Y in val_loader:
            X_one_hot = one_hot_encode(X.numpy(), vocab_size)
            X_one_hot = torch.tensor(X_one_hot, dtype=torch.float32)

            hidden = model.init_hidden(X.size(0))
            output, hidden = model(X_one_hot, hidden)
            loss = criterion(output.view(-1, vocab_size), Y.view(-1))
            losses.append(loss.item())

    average_loss = np.mean(losses)
    return average_loss

############################################################################################################
# Main Execution
############################################################################################################

if __name__ == "__main__":

    # Parameters
    seq_length = 25
    num_epochs = 100
    learning_rate = 0.001
    hidden_size = 128
    num_layers = 1
    batch_size = 64
    use_temp_nuc = False

    model_type = 'LSTM' #'RNN

    # Read and process data
    int_data, ind_to_char, char_to_ind = read_data()
    train_data, val_data, test_data = split_data(int_data, seq_length)

    vocab_size = len(char_to_ind)

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(train_data, val_data, seq_length, batch_size)

    if model_type == "RNN":
        model = RNNModel(vocab_size, hidden_size)
        smooth_losses, losses, val_losses = train_rnn(model, train_loader, val_loader, char_to_ind, num_epochs, learning_rate, use_temp_nuc)
        val_loss = evaluate_model_rnn(model, val_loader, char_to_ind)
        test_loss = evaluate_model_rnn(model, val_loader, char_to_ind)
    else: 
        model = LSTMNet(vocab_size, hidden_size, vocab_size, num_layers)
        smooth_losses, losses, val_losses = train_lstm(model, train_loader, val_loader, char_to_ind, num_epochs, learning_rate, use_temp_nuc)
        val_loss = evaluate_model_lstm(model, val_loader, char_to_ind)
        test_loss = evaluate_model_lstm(model, val_loader, char_to_ind)
  

    print(f"Validation Loss: {val_loss}")
    print(f"Test Loss: {test_loss}")

    # Plot the training losses
    plot_loss(smooth_losses, losses, val_losses)