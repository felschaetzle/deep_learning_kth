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

def ComputeGradients(X, Y, W, lamda):
    P = EvaluateClassifier(X, W, b)
    N = X.shape[1]
    G = -(Y - P)
    grad_W = G @ X.T / N + 2 * lamda * W
    grad_b = np.sum(G, axis=1).reshape(10, 1) / N
    return grad_W, grad_b

def ComputeGradsNum(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))

	c = CalculateCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = CalculateCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = CalculateCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape)
	grad_b = np.zeros((no, 1))
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = CalculateCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = CalculateCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = CalculateCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = CalculateCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]


file = 'assignment1/Datasets/data_batch_1'
lambda_ = 0.1
size = 20

data = LoadBatch(file)

X = np.array(data[b"data"]).T[:, :size]

labels = np.array(data[b"labels"])[:size]
Y = CreateOneHot(labels).T

x_mean = np.mean(X, axis=1).reshape(3072, 1)
X = X - x_mean

x_std = np.std(X, axis=1).reshape(3072, 1)
X = X / x_std

np.random.seed(0)
W = np.random.normal(0, 0.01, (10, 3072))
b = np.random.normal(0, 0.01, (10, 1))

P = EvaluateClassifier(X, W, b)

cost = CalculateCost(X, Y, W, b, lambda_)

accuracy = ComputeAccuracy(X, labels, W, b)

grad_W, grad_b = ComputeGradients(X, Y, W, lambda_)
# print("grad_W: ", grad_W, "grad_b: ", grad_b)

num_grad_W, num_grad_b = ComputeGradsNum(X, Y, P, W, b, lambda_, 1e-6)
# print("num_grad_W: ", num_grad_W, "num_grad_b: ", num_grad_b)

slow_num_grad_W, slow_num_grad_b = ComputeGradsNumSlow(X, Y, P, W, b, lambda_, 1e-6)
# print("slow_num_grad_W: ", slow_num_grad_W, "slow_num_grad_b: ", slow_num_grad_b)

print("###################")
print("Results:")

# print(grad_W - num_grad_W, grad_b - num_grad_b)
# print(grad_W - slow_num_grad_W, grad_b - slow_num_grad_b)

#check if abs difference is less than 1e-6
print(np.allclose(grad_W, num_grad_W, atol=1e-6), np.allclose(grad_b, num_grad_b, atol=1e-6))
print(np.allclose(grad_W, slow_num_grad_W, atol=1e-6), np.allclose(grad_b, slow_num_grad_b, atol=1e-6))
