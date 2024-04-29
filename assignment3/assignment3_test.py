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
	
def InitializeParams(X_train, Y_train, layers):
	W = []
	b = []
	for i, elm in enumerate(layers):
		if i == 0:
			np.random.seed(0)
			W.append(np.random.normal(0, 0.1/np.sqrt(3072), (layers[i], 3072)))
			b.append(np.zeros((layers[i], 1)))
		else:
			np.random.seed(0)
			W.append(np.random.normal(0, 1/np.sqrt(50), (layers[i], layers[i-1])))
			b.append(np.zeros((layers[i], 1)))
	print("W shape", len(W))
	print("b shape", len(b))
	# W_1 = np.random.normal(0, 1/np.sqrt(3072), (50, 3072))
	# b_1 = np.zeros((50, 1))

	# W_2 = np.random.normal(0, 1/np.sqrt(50), (10, 50))
	# b_2 = np.zeros((10, 1))
	# W.append(W_1)
	# W.append(W_2)
	# b.append(b_1)
	# b.append(b_2)
	return W, b

def EvaluateClassifier(X, W, b):
    layer_input = X
    for W_elm, b_elm in zip(W, b):
        s = W_elm @ layer_input + b_elm
        h = np.maximum(0, s)  # ReLU activation
        layer_input = h
    return np.exp(s) / np.sum(np.exp(s), axis=0)

"""
def EvaluateClassifier(X, W, b):
	W_1 = W[0]
	W_2 = W[1]
	b_1 = b[0]
	b_2 = b[1]
	s_1 = W_1@X + b_1
	h = np.maximum(0, s_1)
	s = W_2@h + b_2
	p = np.exp(s) / np.sum(np.exp(s), axis=0)
	return p
"""
def CalculateCost(X, Y, W_1_i, b_1_i, W_2_i, b_2_i, lamda):

	W_el = [W_1_i, W_2_i]
	b_el = [b_1_i, b_2_i]
	N = X.shape[1]
	P = EvaluateClassifier(X, W_el, b_el)
	loss = -np.log(np.diag(Y.T @ P))
	reg_loss = 0
	for elm in W_el:
		reg_loss += np.sum(elm**2)
	reg = lamda * reg_loss

	sum_loss = np.sum(loss)
	return sum_loss / N + reg, sum_loss / N

"""
def CalculateCost(X, Y, W_1, b_1, W_2, b_2, lamda):
	N = X.shape[1]
	W_t = [W_1, W_2]
	b_t = [b_1, b_2]
	P = EvaluateClassifier(X, W_t, b_t)
	loss = -np.log(np.diag(Y.T @ P))
	reg = lamda * (np.sum(W_1**2) + np.sum(W_2**2))
	sum_loss = np.sum(loss)
	return sum_loss / N + reg, sum_loss / N
"""

def ComputeAccuracy(X, y, W, b):
	P = EvaluateClassifier(X, W, b)
	predictions = np.argmax(P, axis=0)
	return np.sum(predictions == y) / len(y)

"""
def ComputeGradients(X, Y, W, b, lamda):
	W_1 = W[0]
	W_2 = W[1]
	b_1 = b[0]
	b_2 = b[1]

	P = EvaluateClassifier(X, W, b)
	N = X.shape[1]
	# print("P shape", P.shape, "Y shape", Y.shape)
	G = -(Y - P)
	# print(G.shape, "G shape")
	forward = W_1@X + b_1
	grad_W_2 = G @ np.maximum(0, forward).T / N + 2 * lamda * W_2
	grad_b_2 = np.sum(G, axis=1).reshape(10, 1) / N
	G = W_2.T @ G
	G = G * (forward > 0)
	grad_W_1 = G @ X.T / N + 2 * lamda * W_1
	grad_b_1 = np.sum(G, axis=1).reshape(50, 1) / N

	g_W = [grad_W_1, grad_W_2]
	g_b = [grad_b_1, grad_b_2]
	return g_W, g_b
"""

def ComputeGradients(X, Y, W, b, lamda):
	P = EvaluateClassifier(X, W, b)
	N = X.shape[1]
	#calculate all gradients for the list of W and b of lenght L
	grad_W = []
	grad_b = []

	# Forward pass
	layer_input = X
	layer_inputs = [X]  # Store inputs for each layer
	layer_outputs = []  # Store outputs (activations) for each layer
	for W_elm, b_elm in zip(W, b):
		s = W_elm @ layer_input + b_elm
		h = np.maximum(0, s)  # ReLU activation
		layer_outputs.append(h)
		layer_input = h
		layer_inputs.append(layer_input)
	# print("layer_inputs", len(layer_inputs))
	# print("layer_outputs", len(layer_outputs))
	# Backward pass
	G = -(Y - P)
	for i in range(len(W) - 1, -1, -1):
		grad_W.append(G @ layer_inputs[i].T / N + 2 * lamda * W[i])
		grad_b.append(np.sum(G, axis=1).reshape(-1, 1) / N)
		G = W[i].T @ G
		G = G * (layer_inputs[i] > 0)  # ReLU derivative


	grad_W = grad_W[::-1]
	grad_b = grad_b[::-1]

	return grad_W, grad_b

"""
	G = -(Y - P)
	for i in range(len(W)-1, -1, -1):
		if i == len(W) - 1:
			s = s_cache[i-1]
			# print("G shape", G.shape)
			# print("s shape", s.shape)
			grad_W.append(G @ np.maximum(0, s).T / N + 2 * lamda * W[i])
			grad_b.append(np.sum(G, axis=1).reshape(W[i].shape[0], 1) / N)
			G = W[i].T @ G
			G = G * (s > 0)

		else:
			h = h_cache[i]
			# print("G shape", G.shape)
			# print("h shape", h.shape)
			grad_W.append(G @ h.T / N + 2 * lamda * W[i])
			grad_b.append(np.sum(G, axis=1).reshape(W[i].shape[0], 1) / N)
			G = W[i].T @ G
			G = G * (h > 0)
	
	grad_W = grad_W[::-1]
	grad_b = grad_b[::-1]
	return grad_W, grad_b

	G = -(Y - P)
	forward = W_1@X + b_1
	grad_W_2 = G @ np.maximum(0, forward).T / N + 2 * lamda * W_2
	grad_b_2 = np.sum(G, axis=1).reshape(10, 1) / N
	G = W_2.T @ G
	G = G * (forward > 0)
	grad_W_1 = G @ X.T / N + 2 * lamda * W_1
	grad_b_1 = np.sum(G, axis=1).reshape(50, 1) / N
	return grad_W_1, grad_b_1, grad_W_2, grad_b_2
"""

def ComputeGradsNum(X, Y, W, b, lamda, h_delta):
	""" Converted from matlab code """
	# W_1 = W[0]
	# W_2 = W[1]
	# b_1 = b[0]
	# b_2 = b[1]

	no 	= 	W[-1].shape[0]
	print("no", no)
	d 	= 	X.shape[0]

	grad_W_1 = np.zeros(W_1.shape)
	grad_b_1 = np.zeros((50, 1))
	grad_W_2 = np.zeros(W_2.shape)
	grad_b_2 = np.zeros((no, 1))


	c,_ = CalculateCost(X, Y, W[0], b[0], W[1], b[1], lamda)
	print(c)
	print("c")


	for i in range(len(b[1])):
		b_try = np.array(b[1])
		b_try[i] += h_delta
		c2,_ = CalculateCost(X, Y, W[0], b[0], W[1], b_try, lamda)
		grad_b_2[i] = (c2-c) / h_delta

	for i in range(W[1].shape[0]):
		for j in range(W[1].shape[1]):
			W_try = np.array(W[1])
			W_try[i,j] += h_delta
			c2,_ = CalculateCost(X, Y, W[0], b[0], W_try, b[1], lamda)
			grad_W_2[i,j] = (c2-c) / h_delta
	
	for i in range(len(b[0])):
		b_try = np.array(b[0])
		b_try[i] += h_delta
		c2,_ = CalculateCost(X, Y, W[0], b_try, W[1], b[1], lamda)
		grad_b_1[i] = (c2-c) / h_delta

	for i in range(W[0].shape[0]):
		for j in range(W[0].shape[1]):
			W_try = np.array(W[0])
			W_try[i,j] += h_delta
			c2,_ = CalculateCost(X, Y, W_try, b[0], W[1], b[1], lamda)
			grad_W_1[i,j] = (c2-c) / h_delta
	
	g_W = [grad_W_1, grad_W_2]
	g_b = [grad_b_1, grad_b_2]

	return g_W, g_b

# def ComputeGradsNum(X, Y, W, b, lambda_, h_size=1e-5):
#     """
#     Computes gradients using finite differences for testing purposes.

#     Arguments:
#     X -- input data of shape (input size, number of examples)
#     Y -- one-hot encoded labels of shape (output size, number of examples)
#     W_list -- list of weight matrices
#     b_list -- list of bias vectors
#     lambda_ -- regularization parameter
#     h_size -- small value for computing finite differences (default 1e-5)

#     Returns:
#     grad_W_list -- list of gradients of the cost function with respect to the weights
#     grad_b_list -- list of gradients of the cost function with respect to the biases
#     """
#     # Initialize gradient lists
#     grad_W_list = [np.zeros_like(W_elm) for W_elm in W]
#     grad_b_list = [np.zeros_like(b_elm) for b_elm in b]

#     c, _ = CalculateCost(X, Y, W[0], b[0], W[1], b[1], lambda_)
#     print("c", c)

#     print("done!")


#     # Compute gradients using finite differences
#     for layer in range(len(W)):
#         W_elm = W[layer]
#         b_elm = b[layer]

#         for i in range(W[layer].shape[0]):
#             for j in range(W[layer].shape[1]):
#                 # Compute gradient for W[i, j]
#                 W_try = W.copy()
#                 W_try[layer][i, j] += h_size
#                 c2, _ = CalculateCost(X, Y, W_try[0], b[0],W_try[1], b[1], lambda_)
#                 # print(c2, c)
#                 # W_try = W.copy()
#                 # W_try[layer][i, j] -= h_size
#                 # c1, _ = CalculateCost(X, Y, W_try[0], b[0],W_try[1], b[1], lambda_)

#                 grad_W_list[layer][i, j] = (c2 - c) / (h_size)

#         for i in range(b[layer].shape[0]):
#             # Compute gradient for b[i]
#             b_try = b.copy()
#             b_try[layer][i] += h_size
#             c2, _ = CalculateCost(X, Y, W[0], b_try[0],W[1], b_try[1], lambda_)

#             # b_try = b.copy()
#             # b_try[layer][i] -= h_size
#             # c1, _ = CalculateCost(X, Y, W, b_try, lambda_)

#             grad_b_list[layer][i] = (c2 - c) / (h_size)

#     return grad_W_list, grad_b_list


# def ComputeGradsNum(X, Y, W, b, lamda, delta_h):
# 	""" Converted from matlab code """
# 	no 	=  10
# 	d 	=  X.shape[0]

# 	grad_W = []
# 	grad_b = []

# 	for i in range(len(W)):
# 		grad_W.append(np.zeros(W[i].shape))
# 		grad_b.append(np.zeros((W[i].shape[0], 1)))

# 	c,_ = CalculateCost(X, Y, W, b, lamda)

# 	for i in range(len(b)):
# 		for j in range(len(b[i])):
# 			b_try = b.copy()
# 			b_try[i][j] += delta_h
# 			c2,_ = CalculateCost(X, Y, W, b_try, lamda)
# 			grad_b[i][j] = (c2-c) / delta_h

# 	for i in range(len(W)):
# 		for j in range(W[i].shape[0]):
# 			for k in range(W[i].shape[1]):
# 				W_try = W.copy()
# 				W_try[i][j,k] += delta_h
# 				c2,_ = CalculateCost(X, Y, W_try, b, lamda)
# 				grad_W[i][j,k] = (c2-c) / delta_h

# 	return grad_W, grad_b


	# no 	= 	W_2.shape[0]
	# d 	= 	X.shape[0]

	# grad_W_1 = np.zeros(W_1.shape)
	# grad_b_1 = np.zeros((50, 1))
	# grad_W_2 = np.zeros(W_2.shape)
	# grad_b_2 = np.zeros((no, 1))

	# c,_ = CalculateCost(X, Y, W_1, b_1, W_2, b_2, lamda)

	# for i in range(len(b_2)):
	# 	b_try = np.array(b_2)
	# 	b_try[i] += h
	# 	c2,_ = CalculateCost(X, Y, W_1, b_1, W_2, b_try, lamda)
	# 	grad_b_2[i] = (c2-c) / h

	# for i in range(W_2.shape[0]):
	# 	for j in range(W_2.shape[1]):
	# 		W_try = np.array(W_2)
	# 		W_try[i,j] += h
	# 		c2,_ = CalculateCost(X, Y, W_1, b_1, W_try, b_2, lamda)
	# 		grad_W_2[i,j] = (c2-c) / h
	
	# for i in range(len(b_1)):
	# 	b_try = np.array(b_1)
	# 	b_try[i] += h
	# 	c2,_ = CalculateCost(X, Y, W_1, b_try, W_2, b_2, lamda)
	# 	grad_b_1[i] = (c2-c) / h

	# for i in range(W_1.shape[0]):
	# 	for j in range(W_1.shape[1]):
	# 		W_try = np.array(W_1)
	# 		W_try[i,j] += h
	# 		c2,_ = CalculateCost(X, Y, W_try, b_1, W_2, b_2, lamda)
	# 		grad_W_1[i,j] = (c2-c) / h
	
	# return [grad_W_1, grad_b_1, grad_W_2, grad_b_2]

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


def MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch):
	eta_s = int(5 * (10000 / n_batch))
	print(eta_s)
	eta_min = 1e-5
	eta_max = 1e-1
	cycles = 1
	n = X_train.shape[1]
	print(n)
	costs_train = []
	costs_val = []
	losses_train = []
	losses_val = []
	accuracies_train = []
	accuracies_val = []

	update_list = []
	eta_list = []

	gamma = []
	beta = []
	#initialize gamma and beta
	for i, elm in enumerate(W):
		if i < len(W) - 1:
			gamma.append(np.ones((elm.shape[0], 1)))
			beta.append(np.zeros((elm.shape[0], 1)))

	for cycle in range(cycles):
		for step in range(2*eta_s):
			if step <= eta_s:
				eta = eta_min + step/eta_s * (eta_max - eta_min)
			else:
				eta = eta_max - (step - eta_s)/eta_s * (eta_max - eta_min)
			# print(step)
			eta_list.append(eta)

			j_start = (step * n_batch) % n
			j_end = ((step + 1) * n_batch) % n
			if ((step + 1) * n_batch)%n == 0:
				j_end = n

			X_batch = X_train[:, j_start:j_end]
			Y_batch = Y_train[:, j_start:j_end]

			grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W, b, lambda_)

			num_grad_W, num_grad_b = ComputeGradsNum(X_batch, Y_batch, W, b, lambda_, 1e-5)

			print("grad_W", grad_W[0].shape, grad_W[1].shape)
			print("num_grad_W", num_grad_W[0].shape, num_grad_W[1].shape)

			print(np.allclose(grad_W[0], num_grad_W[0], rtol=1e-3, atol=1e-3))
			print(np.allclose(grad_W[1], num_grad_W[1], rtol=1e-3, atol=1e-3))
			# print(np.allclose(grad_W[2], num_grad_W[2], rtol=1e-3, atol=1e-3))
			print(np.allclose(grad_b[0], num_grad_b[0], rtol=1e-3, atol=1e-3))
			print(np.allclose(grad_b[1], num_grad_b[1], rtol=1e-3, atol=1e-3))
			# print(np.allclose(grad_b[2], num_grad_b[2], rtol=1e-3, atol=1e-3))
			print("done!")

			for i in range(len(W)):
				W[i] = W[i] - eta * grad_W[i]
				b[i] = b[i] - eta * grad_b[i]

			# for i in range(len(gamma)):
			# 	gamma[i] = gamma[i] - eta * grad_gamma[i]
			# 	beta[i] = beta[i] - eta * grad_beta[i]

			if step % (eta_s/5) == 0:
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
				update_list.append(step + (2*eta_s)*(cycle+1))

				print("Step: ", step, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
	
	return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W": W, "b": b, "update_list": update_list}	

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

training_data_1 = 'assignment1/Datasets/data_batch_1'
training_data_2 = 'assignment1/Datasets/data_batch_2'
training_data_3 = 'assignment1/Datasets/data_batch_3'
training_data_4 = 'assignment1/Datasets/data_batch_4'
training_data_5 = 'assignment1/Datasets/data_batch_5'

test_data_file = 'assignment1/Datasets/test_batch'

# np.random.seed(0)
size = 10000
lambda_ = 0.01 #0.016 beste! 0, 0, 0.1, 1
# eta = 0.001 #0.1, 0.001, 0.001, 0.001
n_batch = 5
layers = [50, 10] #[50, 30, 20, 20, 10, 10, 10, 10] 

training_data_1 = LoadBatch(training_data_1)
# training_data_2 = LoadBatch(training_data_2)
# training_data_3 = LoadBatch(training_data_3)
# training_data_4 = LoadBatch(training_data_4)
# training_data_5 = LoadBatch(training_data_5)

test_data = LoadBatch(test_data_file)

X_train_1, Y_train_1, labels_train_1 = Preprocess(training_data_1)
# X_train_2, Y_train_2, labels_train_2 = Preprocess(training_data_2)
# X_train_3, Y_train_3, labels_train_3 = Preprocess(training_data_3)
# X_train_4, Y_train_4, labels_train_4 = Preprocess(training_data_4)
# X_train_5, Y_train_5, labels_train_5 = Preprocess(training_data_5)

X_test, Y_test, labels_test = Preprocess(test_data)

data_size = 5
X_train, Y_train, labels_train = X_train_1[:,:data_size], Y_train_1[:,:data_size], labels_train_1[:data_size]

W_1 = np.random.normal(0, 1/np.sqrt(3072), (50, 3072))
b_1 = np.zeros((50, 1))

W_2 = np.random.normal(0, 1/np.sqrt(50), (10, 50))
b_2 = np.zeros((10, 1))

W = [W_1, W_2]
b = [b_1, b_2]


grad_W, grad_b = ComputeGradients(X_train, Y_train, W, b, lambda_)

num_grad_W, num_grad_b = ComputeGradsNum(X_train, Y_train, W, b, lambda_, 1e-6)

print("grad_W", grad_W[0].shape, grad_W[1].shape)
print("num_grad_W", num_grad_W[0].shape, num_grad_W[1].shape)

print(np.allclose(grad_W[0], num_grad_W[0], rtol=1e-3, atol=1e-3))
print(np.allclose(grad_W[1], num_grad_W[1], rtol=1e-3, atol=1e-3))
# print(np.allclose(grad_W[2], num_grad_W[2], rtol=1e-3, atol=1e-3))
print(np.allclose(grad_b[0], num_grad_b[0], rtol=1e-3, atol=1e-3))
print(np.allclose(grad_b[1], num_grad_b[1], rtol=1e-3, atol=1e-3))
# print(np.allclose(grad_b[2], num_grad_b[2], rtol=1e-3, atol=1e-3))
print("done!")



# X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
# Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
# labels_train = np.concatenate((labels_train_1, labels_train_2, labels_train_3, labels_train_4, labels_train_5))

# print("train", X_train.shape, Y_train.shape, labels_train.shape)

#Randomly cut out 1000 samples for validation
# val_size = 5
# np.random.seed(0)
# indices = np.random.permutation(X_train.shape[1])
# X_val = X_train[:, indices[:val_size]]
# Y_val = Y_train[:, indices[:val_size]]
# labels_val = labels_train[indices[:val_size]]

# X_train = X_train[:, indices[val_size:]]
# Y_train = Y_train[:, indices[val_size:]]
# labels_train = labels_train[indices[val_size:]]

# print("train", X_train.shape, Y_train.shape, labels_train.shape)
# print("val", X_val.shape, Y_val.shape, labels_val.shape)


# # lambdas_ = [0.01, 0.013, 0.016, 0.02, 0.03]
# # for lambda_ in lambdas_:
# # 	print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
# # 	print("lambda: ", lambda_)

# # W, b = InitializeParams(X_train, Y_train, layers)
# W_1 = np.random.normal(0, 1/np.sqrt(3072), (50, 3072))
# b_1 = np.zeros((50, 1))

# W_2 = np.random.normal(0, 1/np.sqrt(50), (10, 50))
# b_2 = np.zeros((10, 1))

# W = [W_1, W_2]
# b = [b_1, b_2]

# res_dict = MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch)

# test_accuracy = ComputeAccuracy(X_test, labels_test, res_dict["W"], res_dict["b"])

# print("Test accuracy: ", test_accuracy)

# # # # # Montage(res_dict["W_1"])
# # # # # Montage(res_dict["W_2"])

# #plot the cost to a new plot
# plt.figure()
# # plot the update list on the x-axis
# plt.plot(res_dict["update_list"], res_dict["costs_train"], label="Training cost")
# plt.plot(res_dict["update_list"], res_dict["costs_val"], label="Validation cost")
# plt.plot(res_dict["update_list"], res_dict["losses_train"], label="Training loss")
# plt.plot(res_dict["update_list"], res_dict["losses_val"], label="Validation loss")
# plt.title("Training cost vs Validation cost")
# plt.legend()
# plt.xlabel("Updates")
# plt.ylabel("Cost")


# #plot the accuracy to a new plot
# plt.figure()
# plt.plot(res_dict["update_list"], res_dict["accuracies_train"], label="Training accuracy")
# plt.plot(res_dict["update_list"], res_dict["accuracies_val"], label="Validation accuracy")
# plt.title("Training accuracy vs Validation accuracy")
# plt.legend()
# plt.xlabel("Updates")
# plt.ylabel("Accuracy")
# plt.show()
