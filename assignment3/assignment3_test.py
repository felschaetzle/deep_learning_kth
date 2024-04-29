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
	return W, b

def EvaluateClassifier(X, W, b, gamma, beta, mu=None, va=None):
	layer_input = X
	if mu == None and va == None:
		s_hat_cache = []
		s_cache = []
		x_batch_cache = []
		x_batch_cache.append(X)
		mean_cache = []
		var_cache = []
		for i in range(len(W)):
			W_elm = W[i]
			b_elm = b[i]

			if i == len(W) - 1:
				s = W_elm @ layer_input + b_elm
				return np.exp(s) / np.sum(np.exp(s), axis=0), s_hat_cache, s_cache, x_batch_cache, mean_cache, var_cache
			
			gamma_elm = gamma[i]
			beta_elm = beta[i]
					
			s = W_elm @ layer_input + b_elm
			s_cache.append(s)

			#add batch normalization here
			mean = np.mean(s, axis=1).reshape(-1, 1)
			var = np.var(s, axis=1).reshape(-1, 1)

			mean_cache.append(mean)
			var_cache.append(var)

			s_hat = (s - mean) / np.sqrt(var + 1e-8)
			s_hat_cache.append(s_hat)

			s_tilde = s_hat * gamma_elm + beta_elm

			h = np.maximum(0, s_tilde)  # ReLU activation
			x_batch_cache.append(h)

			layer_input = h
	else:
		for i in range(len(W)):
			s = W[i]@layer_input + b[i]

			if i == len(W) -1:
				return np.exp(s) / np.sum(np.exp(s), axis=0)

			s_hat = (s - mu[i])/np.sqrt(va[i] + 1e-8)
			s_tilde = gamma[i] * s_hat + beta[i]
			h = np.maximum(0, s_tilde)
			layer_input = h
		
def CalculateCost(X, Y, W_el, b_el, gamma, beta, lamda, m=None, v=None):
	if m == None and v == None:
		N = X.shape[1]
		P, _, _, _, _, _ = EvaluateClassifier(X, W_el, b_el, gamma, beta)
		loss = -np.log(np.diag(Y.T @ P))
		reg_loss = 0
		for elm in W_el:
			reg_loss += np.sum(elm**2)
		reg = lamda * reg_loss

		sum_loss = np.sum(loss)
		return sum_loss / N + reg, sum_loss / N
	else:
		N = X.shape[1]
		P = EvaluateClassifier(X, W_el, b_el, gamma, beta, mu=m, va=v)
		loss = -np.log(np.diag(Y.T @ P))
		reg_loss = 0
		for elm in W_el:
			reg_loss += np.sum(elm**2)
		reg = lamda * reg_loss

		sum_loss = np.sum(loss)
		return sum_loss / N + reg, sum_loss / N

def ComputeAccuracy(X, y, W, b, gamma, beta, m=None, v=None):
	if m == None and v == None:
		P, _, _, _, _, _ = EvaluateClassifier(X, W, b, gamma, beta)
		predictions = np.argmax(P, axis=0)
		return np.sum(predictions == y) / len(y)
	else:
		P= EvaluateClassifier(X, W, b, gamma, beta, mu=m, va=v)
		predictions = np.argmax(P, axis=0)
		return np.sum(predictions == y) / len(y)
	
def ComputeGradients(X, Y, W, b, gamma, beta, lamda):
	P, s_hat_list, s_list, x_batch_list, mean_list, var_list = EvaluateClassifier(X, W, b, gamma, beta)
	N = X.shape[1]
	L = len(W)

	grad_W = []
	grad_b = []
	grad_gamma = []
	grad_beta = []

	G = -(Y - P)
	for i in range(L - 1, -1, -1):

		if i == L - 1:
			grad_W.append(G @ x_batch_list[i].T / N + 2 * lamda * W[i])
			grad_b.append(np.sum(G, axis=1).reshape(-1, 1) / N)
			G = W[i].T @ G
			G = G * (x_batch_list[i] > 0)  # ReLU derivative
		
		else:
			grad_gamma.append(np.sum(G * s_hat_list[i], axis=1).reshape(-1, 1) / N)
			grad_beta.append(np.sum(G, axis=1).reshape(-1, 1) / N)
			G = G * gamma[i]
			
			G_1 = G * (1/np.sqrt(var_list[i] + 1e-8))
			G_2 = G * (var_list[i] + 1e-8) ** (-3/2)
			D = s_list[i] - mean_list[i]
			c_elm = np.sum(G_2 * D, axis=1).reshape(-1, 1)
			G = G_1 - np.sum(G_1, axis=1).reshape(-1, 1) / N - D * c_elm / N

			grad_W.append(G @ x_batch_list[i].T / N + 2 * lamda * W[i])
			grad_b.append(np.sum(G, axis=1).reshape(-1, 1) / N)
			G = W[i].T @ G
			G = G * (x_batch_list[i] > 0) 

	grad_W = grad_W[::-1]
	grad_b = grad_b[::-1]
	grad_gamma = grad_gamma[::-1]
	grad_beta = grad_beta[::-1]

	return grad_W, grad_b, grad_gamma, grad_beta, mean_list, var_list

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
	mean_global = []
	variance_global = []
	#initialize gamma and beta
	for i, elm in enumerate(W):
		if i < len(W) - 1:
			gamma.append(np.ones((elm.shape[0], n_batch)))
			beta.append(np.zeros((elm.shape[0], 1)))
			mean_global.append(np.zeros((elm.shape[0], 1)))
			variance_global.append(np.zeros((elm.shape[0], 1)))

	for cycle in range(cycles):
		for step in range(2*eta_s):
			if step <= eta_s:
				eta = eta_min + step/eta_s * (eta_max - eta_min)
			else:
				eta = eta_max - (step - eta_s)/eta_s * (eta_max - eta_min)
			eta_list.append(eta)

			j_start = (step * n_batch) % n
			j_end = ((step + 1) * n_batch) % n
			if ((step + 1) * n_batch)%n == 0:
				j_end = n

			X_batch = X_train[:, j_start:j_end]
			Y_batch = Y_train[:, j_start:j_end]

			res_grad_W, res_grad_b, res_grad_gamma, res_grad_beta, mean_update, variance_update = ComputeGradients(X_batch, Y_batch, W, b, gamma, beta, lambda_)


			for i in range(len(W)):
				W[i] = W[i] - eta * res_grad_W[i]
				b[i] = b[i] - eta * res_grad_b[i]

			for i in range(len(gamma)):
				gamma[i] = gamma[i] - eta * res_grad_gamma[i]
				beta[i] = beta[i] - eta * res_grad_beta[i]

			for i in range(len(mean_global)):
				mean_global[i] = 0.9 * mean_global[i] + 0.1 * mean_update[i]
				variance_global[i] = 0.9 * variance_global[i] + 0.1 * variance_update[i]

			if step % (eta_s/5) == 0:
				cost_train, loss_train = CalculateCost(X_train, Y_train, W, b, gamma, beta, lambda_, m=mean_global, v=variance_global)
				costs_train.append(cost_train)
				losses_train.append(loss_train)
				accuracy_train = ComputeAccuracy(X_train, labels_train, W, b, m=mean_global, v=variance_global)
				accuracies_train.append(accuracy_train)

				cost_val, loss_val = CalculateCost(X_val, Y_val, W, b, gamma, beta, lambda_, m=mean_global, v=variance_global)
				costs_val.append(cost_val)
				losses_val.append(loss_val)
				accuracy_val = ComputeAccuracy(X_val, labels_val, W, b, m=mean_global, v=variance_global)
				accuracies_val.append(accuracy_val)
				update_list.append(step + (2*eta_s)*(cycle+1))

				print("Step: ", step, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
	
	return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W": W, "b": b, "update_list": update_list, "mean": mean_global, "variance": variance_global}	

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
n_batch = 100
layers = [50, 30, 10] #[50, 30, 20, 20, 10, 10, 10, 10] 

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

data_size = 10000
X_train, Y_train, labels_train = X_train_1[:,:data_size], Y_train_1[:,:data_size], labels_train_1[:data_size]

# X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
# Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
# labels_train = np.concatenate((labels_train_1, labels_train_2, labels_train_3, labels_train_4, labels_train_5))

# print("train", X_train.shape, Y_train.shape, labels_train.shape)

#Randomly cut out 1000 samples for validation
val_size = 500
np.random.seed(0)
indices = np.random.permutation(X_train.shape[1])
X_val = X_train[:, indices[:val_size]]
Y_val = Y_train[:, indices[:val_size]]
labels_val = labels_train[indices[:val_size]]

X_train = X_train[:, indices[val_size:]]
Y_train = Y_train[:, indices[val_size:]]
labels_train = labels_train[indices[val_size:]]

print("train", X_train.shape, Y_train.shape, labels_train.shape)
print("val", X_val.shape, Y_val.shape, labels_val.shape)


# # lambdas_ = [0.01, 0.013, 0.016, 0.02, 0.03]
# # for lambda_ in lambdas_:
# # 	print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
# # 	print("lambda: ", lambda_)

W, b = InitializeParams(X_train, Y_train, layers)

res_dict = MiniBatchGDCyclicLR(X_train, Y_train, labels_train, X_val, Y_val, labels_val, W, b, lambda_, n_batch)

test_accuracy = ComputeAccuracy(X_test, labels_test, res_dict["W"], res_dict["b"], m=res_dict["mean"], v=res_dict["variance"])

print("Test accuracy: ", test_accuracy)

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
