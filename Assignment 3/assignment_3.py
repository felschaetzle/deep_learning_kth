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
    
        X = np.array(data[b"data"]).T[:, :size]

        y = np.array(data[b"labels"]) + 1
        Y = np.eye(10, dtype=np.double)[np.array(data[b'labels'])].T 

        x_mean = np.mean(X, axis=1).reshape(3072, 1)
        X = X - x_mean

        x_std = np.std(X, axis=1).reshape(3072, 1)
        X = X / x_std

        return X, Y, y

def initialize_parameters(X_train, Y_train, layers):

        W = []
        b = []
        for i, layer in enumerate(layers):
            if i == 0:
               
                #W.append(np.random.normal(0, 1e-4, (layer, 3072)))
                W.append(np.random.normal(0, 0.1 / np.sqrt(X_train.shape[0]), (layer, X_train.shape[0])))
                b.append(np.zeros((layer, 1)))
            else:
                
                #W.append(np.random.normal(0, 1e-4, (layer, layers[i-1])))
                W.append(np.random.normal(0, 1 / np.sqrt(layers[i - 1]), (layer, layers[i - 1])))
                b.append(np.zeros((layer, 1)))
        
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
                W_l = W[i]
                b_l = b[i]

                if i == len(W) - 1:
                    s = np.dot(W_l, layer_input) + b_l
                    return np.exp(s) / np.sum(np.exp(s), axis=0), s_hat_cache, s_cache, x_batch_cache, mean_cache, var_cache
                
                gamma_l = gamma[i]
                beta_l = beta[i]
                        
                s = np.dot(W_l, layer_input) + b_l
                s_cache.append(s)
                    
                mean = np.mean(s, axis=1).reshape(-1, 1)
                var = np.var(s, axis=1).reshape(-1, 1)

                mean_cache.append(mean)
                var_cache.append(var)

                s_hat = (s - mean) / np.sqrt(var + 1e-8)
                s_hat_cache.append(s_hat)

                s_t = s_hat * gamma_l + beta_l

                h = np.maximum(0, s_t) 
                x_batch_cache.append(h)

                layer_input = h
        else:
            for i in range(len(W)):
                s = np.dot(W[i],layer_input) + b[i]

                if i == len(W) -1:
                    return np.exp(s) / np.sum(np.exp(s), axis=0)

                s_hat = (s - mu[i])/np.sqrt(va[i] + 1e-8)
                s_t = gamma[i] * s_hat + beta[i]
                h = np.maximum(0, s_t)
                layer_input = h

def ComputeCost(X, Y, W_el, b_el, gamma, beta, lambda_, m=None, v=None):
        if m == None and v == None:
            N = X.shape[1]
            P, _, _, _, _, _ = EvaluateClassifier(X, W_el, b_el, gamma, beta)
            loss = - np.sum(Y * np.log(P), axis=0)
            reg_loss = 0
            for l in W_el:
                reg_loss += np.sum(l**2)
            reg = lambda_ * reg_loss

            J_cost = (1 / N) * np.sum(loss) + reg
            J_loss = (1 / N) * np.sum(loss)
            return J_cost, J_loss
        else:
            N = X.shape[1]
            P = EvaluateClassifier(X, W_el, b_el, gamma, beta, mu=m, va=v)
            loss = - np.sum(Y * np.log(P), axis=0)
            reg_loss = 0
            for l in W_el:
                reg_loss += np.sum(l**2)
            reg = lambda_ * reg_loss

            J_cost = (1 / N) * np.sum(loss) + reg
            J_loss = (1 / N) * np.sum(loss)
            return J_cost, J_loss


def ComputeAccuracy(X, y, W, b, gamma, beta):
    
        P, _, _, _, _, _, _, _ = EvaluateClassifier(X, W, b, gamma, beta)
        y_pred= np.argmax(P, axis=0) + 1
        acc = accuracy_score(y, y_pred)
        
        return acc

def ComputeAccuracy(X, y, W, b, gamma, beta, m=None, v=None):
        if m == None and v == None:
            P, _, _, _, _, _ = EvaluateClassifier(X, W, b, gamma, beta)
            y_pred= np.argmax(P, axis=0) + 1
            acc = accuracy_score(y, y_pred)
            return acc
        else:
            P= EvaluateClassifier(X, W, b, gamma, beta, mu=m, va=v)
            y_pred= np.argmax(P, axis=0) + 1
            acc = accuracy_score(y, y_pred)
            return acc

def ComputeGradients(X, Y, W, b, gamma, beta, lambda_):
        grad_W = []
        grad_b = []
        grad_gamma = []
        grad_beta = []
        
        P, s_hat_list, s_list, x_batch_list, mean_list, var_list = EvaluateClassifier(X, W, b, gamma, beta)
        N = X.shape[1]
        L = len(W)

        G = -(Y - P)
        for i in range(L - 1, -1, -1):
            if i == L - 1:
                grad_W.append(np.dot(G, x_batch_list[i].T) / N + 2 * lambda_ * W[i])
                grad_b.append(np.sum(G, axis=1).reshape(-1, 1) / N)
                G = np.dot(W[i].T , G)
                G = G * (x_batch_list[i] > 0)
            
            else:
                grad_gamma.append(np.sum(G * s_hat_list[i], axis=1).reshape(-1, 1) / N)
                grad_beta.append(np.sum(G, axis=1).reshape(-1, 1) / N)
                G = G * gamma[i]
                G_1 = G * (1/np.sqrt(var_list[i] + 1e-8))
                G_2 = G * (var_list[i] + 1e-8) ** (-3/2)
                D = s_list[i] - mean_list[i]
                c_l = np.sum(G_2 * D, axis=1).reshape(-1, 1)
                G = G_1 - np.sum(G_1, axis=1).reshape(-1, 1) / N - D * c_l / N
                grad_W.append(np.dot(G, x_batch_list[i].T) / N + 2 * lambda_ * W[i])
                grad_b.append(np.sum(G, axis=1).reshape(-1, 1) / N)
                G = np.dot(W[i].T , G)
                G = G * (x_batch_list[i] > 0) 

        grad_W = grad_W[::-1]
        grad_b = grad_b[::-1]
        grad_gamma = grad_gamma[::-1]
        grad_beta = grad_beta[::-1]

        return grad_W, grad_b, grad_gamma, grad_beta, mean_list, var_list



def ComputeGradsNum(X, Y, W, b, lambda_, h_delta):
	""" Converted from matlab code """

	grad_num_W = []
	grad_num_b = []
	for i in range(len(W)):
		grad_num_W.append(np.zeros(W[i].shape))
		grad_num_b.append(np.zeros((W[i].shape[0], 1)))

	c,_ = ComputeCost(X, Y, W, b, lambda_)

	for i in range(len(W)):
		for j in range(W[i].shape[0]):
			for k in range(W[i].shape[1]):
				W_l = np.array(W[i])
				W_l[j,k] = W_l[j,k] + h_delta
				W_try = W.copy()
				W_try[i] = W_l
                    
				c2,_ = ComputeCost(X, Y, W_try, b, lambda_)
                    
				grad_num_W[i][j,k] = (c2-c) / h_delta

		for j in range(len(b[i])):
			b_l = np.array(b[i])
			b_l[j] = b_l[j] + h_delta
			b_try = b.copy()
			b_try[i] = b_l
               
			c2,_ = ComputeCost(X, Y, W, b_try, lambda_)
               
			grad_num_b[i][j] = (c2-c) / h_delta

	return grad_num_W, grad_num_b

def MiniBatchGDCyclicLR(X_train, Y_train, y_train, X_val, Y_val, y_val, W, b, lambda_, n_batch):
        
        eta_s = int(5 * (45000 / n_batch))
        eta_min = 1e-5
        eta_max = 1e-1
        cycles = 2
        n = X_train.shape[1]
        mean = []
        variance = []
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
        
        for i, l in enumerate(W):
            if i < len(W) - 1:
                gamma.append(np.ones((l.shape[0], 1)))
                beta.append(np.zeros((l.shape[0], 1)))
                mean.append(np.zeros((l.shape[0], 1)))
                variance.append(np.zeros((l.shape[0], 1)))

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

                for i in range(len(mean)):
                    mean[i] = 0.9 * mean[i] + 0.1 * mean_update[i]
                    variance[i] = 0.9 * variance[i] + 0.1 * variance_update[i]

                if step % (eta_s/5) == 0:
                    cost_train, loss_train = ComputeCost(X_train, Y_train, W, b, gamma, beta, lambda_, m=mean, v=variance)
                    costs_train.append(cost_train)
                    losses_train.append(loss_train)
                    accuracy_train = ComputeAccuracy(X_train, y_train, W, b, gamma, beta, m=mean, v=variance)
                    accuracies_train.append(accuracy_train)

                    cost_val, loss_val = ComputeCost(X_val, Y_val, W, b, gamma, beta, lambda_, m=mean, v=variance)
                    costs_val.append(cost_val)
                    losses_val.append(loss_val)
                    accuracy_val = ComputeAccuracy(X_val, y_val, W, b, gamma, beta, m=mean, v=variance)
                    accuracies_val.append(accuracy_val)
                    update_list.append(step + (2*eta_s)*(cycle+1))

                    print("Step: ", step, "Cost: ", cost_train, "Accuracy Training: ", accuracy_train,"Accuracy Validation: ", accuracy_val)
        
        return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W": W, "b": b, "update_list": update_list, "mean": mean, "variance": variance, "gamma": gamma, "beta": beta}	


data_1 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_1')
data_2 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_2')
data_3 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_3')
data_4 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_4')
data_5 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_5')

test_data = load_batch('Assigment 1/Datasets/cifar-10-batches-py/test_batch')

size = 10000
#lambda_ = [0.001, 0.005, 0.01, 0.02, 0.05]
# 0.001 = 51,46
# 0.005 = 53,14
# 0.01 = 52,67
# 0.02 = 51,82
# 0.05 = 51,06


# 0.006 = 53,38 (after 3 cylces: 52,69 )
# 0.007 = 52,98 
# 0.004 = 52,65

lambda_ = 0.005
n_batch = 100
layers = [50, 50, 10] 

#X_train, Y_train, y_train = preprocess_data(data_1)
X_val, Y_val, y_val = preprocess_data(data_2)
X_train_1, Y_train_1, y_1 = preprocess_data(data_1)
X_train_2, Y_train_2, y_2 = preprocess_data(data_2)
X_train_3, Y_train_3, y_3 = preprocess_data(data_3)
X_train_4, Y_train_4, y_4 = preprocess_data(data_4)
X_train_5, Y_train_5, y_5 = preprocess_data(data_5)

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
y_train= np.concatenate((y_1, y_2, y_3, y_4, y_5))

indices = np.random.permutation(X_train.shape[1])
X_val = X_train[:, indices[:5000]]
Y_val = Y_train[:, indices[:5000]]
y_val = y_train[indices[:5000]]

X_train = X_train[:, indices[5000:]]
Y_train = Y_train[:, indices[5000:]]
y_train = y_train[indices[5000:]]

X_test, Y_test, y_test = preprocess_data(test_data)


W, b = initialize_parameters(X_train,Y_train, layers)

results_dict = MiniBatchGDCyclicLR(X_train, Y_train, y_train, X_val, Y_val, y_val, W, b, lambda_, n_batch)

test_accuracy = ComputeAccuracy(X_test, y_test, results_dict["W"], results_dict["b"], results_dict["gamma"], results_dict["beta"], m=results_dict["mean"], v=results_dict["variance"])


print("Test accuracy: ", test_accuracy)


#Compare gradient methods
#grad_W1, grad_b1 = ComputeGradients(X_train[0:20, 1:2],Y_train[:, 1:2], W, b, 0.0)
#grad_W1_num, grad_b1_num = ComputeGradsNum(X_train[0:20, 1:2], Y_train[:, 1:2], W, b, 0.0, 1e-5)

#check for absolut difference < 1e-6
#print(np.allclose(grad_W1[0], grad_W1_num[0], atol=1e-6), np.allclose(grad_b1[0], grad_b1_num[0], atol=1e-6))


#loss plot
plt.figure()
plt.plot(results_dict["losses_train"], label="Training loss")
plt.plot(results_dict["losses_val"], label="Validation loss")
plt.legend()
plt.xlabel("Updates")
plt.ylabel("Loss")
plt.show()

# costs plot
#plt.figure()
#plt.plot(results_dict["costs_train"], label="Training cost")
#plt.plot(results_dict["costs_val"], label="Validation cost")
#plt.legend()
#plt.xlabel("Epoch")
#plt.ylabel("Cost")
#plt.show()
