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
			W.append(np.random.normal(0, 0.01, (layers[i], 3072)))
			b.append(np.zeros((layers[i], 1)))
		else:
			W.append(np.random.normal(0, 0.01, (layers[i], layers[i-1])))
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
	s_cache = []
	h_cache = []
	h_cache.append(X)
	for i in range(len(W)):
		if i == 0:
			s = W[i]@X + b[i]
			h = np.maximum(0, s)

		elif i < len(W) - 1:
			s = W[i]@h + b[i]
			h = np.maximum(0, s)
		else:
			s = W[i]@h + b[i]
			p = np.exp(s) / np.sum(np.exp(s), axis=0)
			# print("eval done! p shape", p.shape)
			# print("s len", len(s_cache))
			# print("s", s_cache[0].shape, s_cache[1].shape)
			# print("h len", len(h_cache))
			# print("h", h_cache[0].shape, h_cache[1].shape, h_cache[2].shape)
			return p, s_cache, h_cache
		s_cache.append(s)
		h_cache.append(h)

def CalculateCost(X, Y, W, b, lamda):
	N = X.shape[1]
	P, _, _ = EvaluateClassifier(X, W, b)
	loss = -np.log(np.diag(Y.T @ P))
	reg_loss = 0
	for elm in W:
		reg_loss += np.sum(elm**2)
	reg = lamda * reg_loss

	sum_loss = np.sum(loss)
	return sum_loss / N + reg, sum_loss / N

def ComputeAccuracy(X, y, W, b):
	P, _, _ = EvaluateClassifier(X, W, b)
	predictions = np.argmax(P, axis=0)
	return np.sum(predictions == y) / len(y)

def ComputeGradients(X, Y, W, b, lamda):
	P, s_cache, h_cache = EvaluateClassifier(X, W, b)
	N = X.shape[1]
	#calculate all gradients for the list of W and b of lenght L
	grad_W = []
	grad_b = []
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

	# G = -(Y - P)
	# forward = W_1@X + b_1
	# grad_W_2 = G @ np.maximum(0, forward).T / N + 2 * lamda * W_2
	# grad_b_2 = np.sum(G, axis=1).reshape(10, 1) / N
	# G = W_2.T @ G
	# G = G * (forward > 0)
	# grad_W_1 = G @ X.T / N + 2 * lamda * W_1
	# grad_b_1 = np.sum(G, axis=1).reshape(50, 1) / N
	# return grad_W_1, grad_b_1, grad_W_2, grad_b_2

def ComputeGradsNum(X, Y, W_1, b_1, W_2, b_2, lamda, h):
	""" Converted from matlab code """
	no 	= 	W_2.shape[0]
	d 	= 	X.shape[0]

	grad_W_1 = np.zeros(W_1.shape)
	grad_b_1 = np.zeros((50, 1))
	grad_W_2 = np.zeros(W_2.shape)
	grad_b_2 = np.zeros((no, 1))

	c,_ = CalculateCost(X, Y, W_1, b_1, W_2, b_2, lamda)

	for i in range(len(b_2)):
		b_try = np.array(b_2)
		b_try[i] += h
		c2,_ = CalculateCost(X, Y, W_1, b_1, W_2, b_try, lamda)
		grad_b_2[i] = (c2-c) / h

	for i in range(W_2.shape[0]):
		for j in range(W_2.shape[1]):
			W_try = np.array(W_2)
			W_try[i,j] += h
			c2,_ = CalculateCost(X, Y, W_1, b_1, W_try, b_2, lamda)
			grad_W_2[i,j] = (c2-c) / h
	
	for i in range(len(b_1)):
		b_try = np.array(b_1)
		b_try[i] += h
		c2,_ = CalculateCost(X, Y, W_1, b_try, W_2, b_2, lamda)
		grad_b_1[i] = (c2-c) / h

	for i in range(W_1.shape[0]):
		for j in range(W_1.shape[1]):
			W_try = np.array(W_1)
			W_try[i,j] += h
			c2,_ = CalculateCost(X, Y, W_try, b_1, W_2, b_2, lamda)
			grad_W_1[i,j] = (c2-c) / h
	
	return [grad_W_1, grad_b_1, grad_W_2, grad_b_2]

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

def MiniBatchGD(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W_1, b_1, W_2, b_2, lambda_, n_batch, eta, n_epochs):
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
			grad_W_1, grad_b_1, grad_W_2, grad_b_2 = ComputeGradients(X_batch, Y_batch, W_1, b_1, W_2, b_2, lambda_)
			W_1 = W_1 - eta * grad_W_1
			b_1 = b_1 - eta * grad_b_1

			W_2 = W_2 - eta * grad_W_2
			b_2 = b_2 - eta * grad_b_2

		cost_train, loss_train = CalculateCost(X_train, Y_train, W_1, b_1, W_2, b_2, lambda_)
		costs_train.append(cost_train)
		losses_train.append(loss_train)
		accuracy_train = ComputeAccuracy(X_train, labels_train, W_1, b_1, W_2, b_2)
		accuracies_train.append(accuracy_train)

		cost_val, loss_val = CalculateCost(X_val, Y_val, W_1, b_1, W_2, b_2, lambda_)
		costs_val.append(cost_val)
		losses_val.append(loss_val)
		accuracy_val = ComputeAccuracy(X_val, labels_val, W_1, b_1, W_2, b_2)
		accuracies_val.append(accuracy_val)

		if epoch % 10 == 0:
			print("Epoch: ", epoch, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
	return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W_1": W_1, "b_1": b_1, "W_2": W_2, "b_2": b_2}	

def MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch):
	eta_s = int(5 * (45000 / n_batch))
	print(eta_s)
	eta_min = 1e-5
	eta_max = 1e-1
	cycles = 2
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

			for i in range(len(W)):
				W[i] = W[i] - eta * grad_W[i]
				b[i] = b[i] - eta * grad_b[i]

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
lambda_ = 0.005 #0.016 beste! 0, 0, 0.1, 1
# eta = 0.001 #0.1, 0.001, 0.001, 0.001
n_batch = 100
# n_epochs = 10
layers = [50, 30, 20, 20, 10, 10, 10, 10] # [50, 50, 10]

training_data_1 = LoadBatch(training_data_1)
training_data_2 = LoadBatch(training_data_2)
training_data_3 = LoadBatch(training_data_3)
training_data_4 = LoadBatch(training_data_4)
training_data_5 = LoadBatch(training_data_5)

test_data = LoadBatch(test_data_file)

X_train_1, Y_train_1, labels_train_1 = Preprocess(training_data_1)
X_train_2, Y_train_2, labels_train_2 = Preprocess(training_data_2)
X_train_3, Y_train_3, labels_train_3 = Preprocess(training_data_3)
X_train_4, Y_train_4, labels_train_4 = Preprocess(training_data_4)
X_train_5, Y_train_5, labels_train_5 = Preprocess(training_data_5)

X_test, Y_test, labels_test = Preprocess(test_data)

# X_train, Y_train, labels_train = X_train_1, Y_train_1, labels_train_1


X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
labels_train = np.concatenate((labels_train_1, labels_train_2, labels_train_3, labels_train_4, labels_train_5))

# print("train", X_train.shape, Y_train.shape, labels_train.shape)

#Randomly cut out 1000 samples for validation
indices = np.random.permutation(X_train.shape[1])
X_val = X_train[:, indices[:5000]]
Y_val = Y_train[:, indices[:5000]]
labels_val = labels_train[indices[:5000]]

X_train = X_train[:, indices[5000:]]
Y_train = Y_train[:, indices[5000:]]
labels_train = labels_train[indices[5000:]]

print("train", X_train.shape, Y_train.shape, labels_train.shape)
print("val", X_val.shape, Y_val.shape, labels_val.shape)


# lambdas_ = [0.01, 0.013, 0.016, 0.02, 0.03]
# for lambda_ in lambdas_:
# 	print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
# 	print("lambda: ", lambda_)



W, b = InitializeParams(X_train, Y_train, layers)

# _ = EvaluateClassifier(X_train, W, b)

# grad_W, grad_b = ComputeGradients(X_train, Y_train, W, b, lambda_)

res_dict = MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch)


test_accuracy = ComputeAccuracy(X_test, labels_test, res_dict["W"], res_dict["b"])

print("Test accuracy: ", test_accuracy)

# # # # Montage(res_dict["W_1"])
# # # # Montage(res_dict["W_2"])

#plot the cost to a new plot
plt.figure()
# plot the update list on the x-axis
plt.plot(res_dict["update_list"], res_dict["costs_train"], label="Training cost")
plt.plot(res_dict["update_list"], res_dict["costs_val"], label="Validation cost")
plt.plot(res_dict["update_list"], res_dict["losses_train"], label="Training loss")
plt.plot(res_dict["update_list"], res_dict["losses_val"], label="Validation loss")
plt.title("Training cost vs Validation cost")
plt.legend()
plt.xlabel("Updates")
plt.ylabel("Cost")


#plot the accuracy to a new plot
plt.figure()
plt.plot(res_dict["update_list"], res_dict["accuracies_train"], label="Training accuracy")
plt.plot(res_dict["update_list"], res_dict["accuracies_val"], label="Validation accuracy")
plt.title("Training accuracy vs Validation accuracy")
plt.legend()
plt.xlabel("Updates")
plt.ylabel("Accuracy")
plt.show()
