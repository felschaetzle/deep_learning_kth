import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open(filename, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def Preprocess(data):
	X = np.array(data[b"data"]).T[:, :size]

	labels = np.array(data[b"labels"])[:size]
	Y = CreateOneHot(labels).T
	print(X.shape, "X shape")
	x_mean = np.mean(X, axis=1).reshape(3072, 1)
	X = X - x_mean

	x_std = np.std(X, axis=1).reshape(3072, 1)
	X = X / x_std

	return X, Y, labels

def CreateOneHot(labels):
    one_hot_labels = np.zeros((len(labels), 10))
    one_hot_labels[np.arange(len(labels)), labels] = 1
    return one_hot_labels
	
def EvaluateClassifier(X, W, b):
    s = W@X + b
    p = np.exp(s) / np.sum(np.exp(s), axis=0)
    return p

def CalculateCost(X, Y, W, b, lamda):
    N = X.shape[1]
    P = EvaluateClassifier(X, W, b)
    loss = -np.log(np.diag(Y.T @ P))
    reg = lamda * np.sum(W**2)
    return np.sum(loss) / N + reg, np.sum(loss) / N

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

	c,_ = CalculateCost(X, Y, W, b, lamda)
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2,_ = CalculateCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2,_ = CalculateCost(X, Y, W_try, b, lamda)
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
		c1,_ = CalculateCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2,_ = CalculateCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1,_ = CalculateCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2,_ = CalculateCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def MiniBatchGD(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch, eta, n_epochs):
	n = X_train.shape[1]
	print(n)
	costs_train = []
	costs_val = []
	losses_train = []
	losses_val = []
	accuracies_train = []
	accuracies_val = []
	for epoch in range(n_epochs):
		for j in range(1, int(n/n_batch)):
			j_start = (j-1)*n_batch + 1
			j_end = j*n_batch
			X_batch = X_train[:, j_start:j_end]
			Y_batch = Y_train[:, j_start:j_end]
			grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W, lambda_)
			W = W - eta * grad_W
			b = b - eta * grad_b

		cost_train, loss_train = CalculateCost(X_train, Y_train, W, b, lambda_)
		costs_train.append(cost_train)
		losses_train.append(loss_train)
		accuracy_train = ComputeAccuracy(X_train, labels_train, W, b)
		accuracies_train.append(accuracy_train)

		cost_val, loss_val = CalculateCost(X_val, Y_val, W, b, lambda_)
		costs_val.append(cost_val)
		losses_val.append(loss_val)
		accuracy_val = ComputeAccuracy(X_val, labels_val, W, b)
		accuracies_val.append(accuracy_val)

		if epoch % 10 == 0:
			print("Epoch: ", epoch, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
	return {"W": W, "b": b, "costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val}	

def Visualize(data):
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

def Montage(W):
	""" Display the image for each label in W """
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')

training_data_file = 'assignment1/Datasets/data_batch_1'
validation_data_file = 'assignment1/Datasets/data_batch_2'
test_data_file = 'assignment1/Datasets/test_batch'

np.random.seed(0)
size = 10000
lambda_ = 1 #0, 0, 0.1, 1
eta = 0.001 #0.1, 0.001, 0.001, 0.001
n_batch = 100
n_epochs = 40

training_data = LoadBatch(training_data_file)
validation_data = LoadBatch(validation_data_file)
test_data = LoadBatch(test_data_file)

X_train, Y_train, labels_train = Preprocess(training_data)
X_val, Y_val, labels_val = Preprocess(validation_data)
X_test, Y_test, labels_test = Preprocess(test_data)

W = np.random.normal(0, 0.01, (10, 3072))
b = np.random.normal(0, 0.01, (10, 1))

res_dict = MiniBatchGD(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch, eta, n_epochs)

test_accuracy = ComputeAccuracy(X_test, labels_test, res_dict["W"], res_dict["b"])
print("Test accuracy: ", test_accuracy)

Montage(res_dict["W"])

#plot the cost to a new plot
plt.figure()
plt.plot(res_dict["costs_train"], label="Training cost")
plt.plot(res_dict["costs_val"], label="Validation cost")
plt.plot(res_dict["losses_train"], label="Training loss")
plt.plot(res_dict["losses_val"], label="Validation loss")
plt.title("Training cost vs Validation cost")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()



##### dev stuff
# P = EvaluateClassifier(X, W, b)

# cost, loss = CalculateCost(X, Y, W, b, lambda_)

# accuracy = ComputeAccuracy(X, labels, W, b)

# grad_W, grad_b = ComputeGradients(X, Y, W, lambda_)
# # print("grad_W: ", grad_W, "grad_b: ", grad_b)

# num_grad_W, num_grad_b = ComputeGradsNum(X, Y, P, W, b, lambda_, 1e-6)
# # print("num_grad_W: ", num_grad_W, "num_grad_b: ", num_grad_b)

# slow_num_grad_W, slow_num_grad_b = ComputeGradsNumSlow(X, Y, P, W, b, lambda_, 1e-6)
# # print("slow_num_grad_W: ", slow_num_grad_W, "slow_num_grad_b: ", slow_num_grad_b)

# print("###################")
# print("Results:")

# # print(grad_W - num_grad_W, grad_b - num_grad_b)
# # print(grad_W - slow_num_grad_W, grad_b - slow_num_grad_b)

# #check if abs difference is less than 1e-6
# print(np.allclose(grad_W, num_grad_W, atol=1e-6), np.allclose(grad_b, num_grad_b, atol=1e-6))
# print(np.allclose(grad_W, slow_num_grad_W, atol=1e-6), np.allclose(grad_b, slow_num_grad_b, atol=1e-6))