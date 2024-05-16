import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random

def load_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data
    

def preprocess_data(data):
	
	X = np.array(data[b"data"]).T[:, :10000]

	labels = np.array(data[b"labels"])[:10000]
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
        for i, layer in enumerate(layers):
            if i == 0:
                    # Adjust the shape of W for the first layer to match the reduced input size
                    W.append(np.random.normal(0, 1e-4, (layer, 3072)))
                    #W.append(np.random.normal(0, 0.1 / np.sqrt(X_train.shape[0]), (layer, X_train.shape[0])))
                    b.append(np.zeros((layer, 1)))
            else:
                    # Initialize the remaining layers as before
                    W.append(np.random.normal(0, 1e-4, (layer, layers[i-1])))
                    #W.append(np.random.normal(0, 1 / np.sqrt(layers[i - 1]), (layer, layers[i - 1])))
                    b.append(np.zeros((layer, 1)))
        
        return W, b

def EvaluateClassifier(X, W, b):
    layer_input = X
    for W_elm, b_elm in zip(W, b):
        s = W_elm @ layer_input + b_elm
        h = np.maximum(0, s)  # ReLU activation
        layer_input = h
    return np.exp(s) / np.sum(np.exp(s), axis=0)

def CalculateCost(X, Y, W_el, b_el, lamda):

	N = X.shape[1]
	P = EvaluateClassifier(X, W_el, b_el)
	loss = - np.sum(Y * np.log(P), axis=0)
	reg_loss = 0
	for elm in W_el:
		reg_loss += np.sum(elm**2)
	reg = lamda * reg_loss

	sum_loss = np.sum(loss)
	return sum_loss / N + reg, sum_loss / N

def ComputeAccuracy(X, y, W, b):
	P = EvaluateClassifier(X, W, b)
	predictions = np.argmax(P, axis=0)
	return np.sum(predictions == y) / len(y)

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

def ComputeGradsNum(X, Y, W, b, lamda, h_delta):
	""" Converted from matlab code """

	gra_W = []
	gra_b = []
	for i in range(len(W)):
		gra_W.append(np.zeros(W[i].shape))
		gra_b.append(np.zeros((W[i].shape[0], 1)))

	c,_ = CalculateCost(X, Y, W, b, lamda)
	print(c)
	print("c")

	for i in range(len(W)):
		for j in range(W[i].shape[0]):
			for k in range(W[i].shape[1]):
				W_elm = np.array(W[i])
				W_elm[j,k] = W_elm[j,k] + h_delta
				W_try = W.copy()
				W_try[i] = W_elm
				c2,_ = CalculateCost(X, Y, W_try, b, lamda)
				gra_W[i][j,k] = (c2-c) / h_delta

		for j in range(len(b[i])):
			b_elm = np.array(b[i])
			b_elm[j] = b_elm[j] + h_delta
			b_try = b.copy()
			b_try[i] = b_elm
			c2,_ = CalculateCost(X, Y, W, b_try, lamda)
			gra_b[i][j] = (c2-c) / h_delta

	return gra_W, gra_b

def MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch):
	eta_s = int(5 * (45000/ n_batch))
	print(eta_s)
	eta_min = 1e-5
	eta_max = 1e-1
	cycles = 3
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

			res_grad_W, res_grad_b = ComputeGradients(X_batch, Y_batch, W, b, lambda_)

			for i in range(len(W)):
				W[i] = W[i] - eta * res_grad_W[i]
				b[i] = b[i] - eta * res_grad_b[i]

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

training_data_1 = 'Assigment 1/Datasets/cifar-10-batches-py/data_batch_1'
training_data_2 = 'Assigment 1/Datasets/cifar-10-batches-py/data_batch_2'
training_data_3 = 'Assigment 1/Datasets/cifar-10-batches-py/data_batch_3'
training_data_4 = 'Assigment 1/Datasets/cifar-10-batches-py/data_batch_4'
training_data_5 = 'Assigment 1/Datasets/cifar-10-batches-py/data_batch_5'

test_data_file = 'Assigment 1/Datasets/cifar-10-batches-py/test_batch'

# np.random.seed(0)
#size = 10000
lambda_ = 0.005 
# eta = 0.001 #0.1, 0.001, 0.001, 0.001
n_batch = 100
layers = [50, 50, 10] #[50, 30, 20, 20, 10, 10, 10, 10] 

training_data_1 = load_batch(training_data_1)
training_data_2 = load_batch(training_data_2)
training_data_3 = load_batch(training_data_3)
training_data_4 = load_batch(training_data_4)
training_data_5 = load_batch(training_data_5)

test_data = load_batch(test_data_file)

X_train_1, Y_train_1, labels_train_1 = preprocess_data(training_data_1)
X_train_2, Y_train_2, labels_train_2 = preprocess_data(training_data_2)
X_train_3, Y_train_3, labels_train_3 = preprocess_data(training_data_3)
X_train_4, Y_train_4, labels_train_4 = preprocess_data(training_data_4)
X_train_5, Y_train_5, labels_train_5 = preprocess_data(training_data_5)

X_test, Y_test, labels_test = preprocess_data(test_data)

#data_size = 10000
#X_train, Y_train, labels_train = X_train_1[:,:data_size], Y_train_1[:,:data_size], labels_train_1[:data_size]

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
labels_train = np.concatenate((labels_train_1, labels_train_2, labels_train_3, labels_train_4, labels_train_5))

# print("train", X_train.shape, Y_train.shape, labels_train.shape)

#Randomly cut out 1000 samples for validation
val_size = 5000
indices = np.random.permutation(X_train.shape[1])
X_val = X_train[:, indices[:val_size]]
Y_val = Y_train[:, indices[:val_size]]
labels_val = labels_train[indices[:val_size]]

X_train = X_train[:, indices[val_size:]]
Y_train = Y_train[:, indices[val_size:]]
labels_train = labels_train[indices[val_size:]]



# # lambdas_ = [0.01, 0.013, 0.016, 0.02, 0.03]
# # for lambda_ in lambdas_:
# # 	print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
# # 	print("lambda: ", lambda_)

W, b = InitializeParams(X_train, Y_train, layers)

res_dict = MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch)

test_accuracy = ComputeAccuracy(X_test, labels_test, res_dict["W"], res_dict["b"])

print("Test accuracy: ", test_accuracy)

plt.figure()
plt.plot(res_dict["losses_train"], label="Training loss")
plt.plot(res_dict["losses_val"], label="Validation loss")
plt.legend()
plt.xlabel("Updates")
plt.ylabel("Loss")
plt.show()