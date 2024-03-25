import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open(filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def CreateOneHot(labels):
    one_hot_labels = np.zeros((len(labels), 10))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels

def visualize(data):
    # reshape the data
    data = data.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    # plot the data
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(data[random.randint(0, 9999)])
    plt.show()

def EvaluateClassifier(X, W, b):
    s = W@X + b
    p = np.exp(s) / np.sum(np.exp(s), axis=0)
    return p

# calculate cross entropy loss with regularization
def CalculateCost(X, Y, W, b, lamda):
    N = X.shape[1]
    P = EvaluateClassifier(X, W, b)
    loss = -np.log(np.diag(Y.T @ P))
    reg = lamda * np.sum(W**2)
    return np.sum(loss) / N + reg

def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b)
    predictions = np.argmax(P, axis=0)
    return np.sum(predictions == y) / len(y)

def ComputeGradients(X, Y, P, W, lamda):
    N = X.shape[1]
    G = -(Y - P)
    grad_W = G @ X.T / N + 2 * lamda * W
    grad_b = np.sum(G, axis=1).reshape(10, 1) / N
    return grad_W, grad_b

file = 'assignment1/Datasets/data_batch_1'
lambda_ = 0.1

data = LoadBatch(file)

X = np.array(data[b"data"]).T

labels = np.array(data[b"labels"])
Y = CreateOneHot(labels).T

x_mean = np.mean(X, axis=1).reshape(3072, 1)
X = X - x_mean

x_std = np.std(X, axis=1).reshape(3072, 1)
X = X / x_std

# initialize the weights and bias with 0 mean and 0.01 standard deviation
np.random.seed(0)
W = np.random.normal(0, 0.01, (10, 3072))
b = np.random.normal(0, 0.01, (10, 1))

P = EvaluateClassifier(X, W, b)

cost = CalculateCost(X, Y, W, b, lambda_)

accuracy = ComputeAccuracy(X, labels, W, b)

grad_W, grad_b = ComputeGradients(X, Y, P, W, lambda_)

print("grad_W shape: ", grad_W.shape, "grad_b shape: ", grad_b.shape)
print("grad_b: ", grad_b)