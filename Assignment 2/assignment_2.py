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

        y = np.array(data[b"labels"]) + 1
        Y = np.eye(10, dtype=np.double)[np.array(data[b'labels'])].T 

        x_mean = np.mean(X, axis=1).reshape(3072, 1)
        X = X - x_mean

        x_std = np.std(X, axis=1).reshape(3072, 1)
        X = X / x_std

        return X, Y, y

def initialize_parameters(K, d, m = 50):

    W_1 = np.random.normal(0, 1/np.sqrt(d), (m, d))
    W_2 = np.random.normal(0, 1/np.sqrt(m), (K, m))
    b_1 = np.zeros((m, 1))
    b_2 = np.zeros((K, 1))

    return W_1, b_1, W_2, b_2

def EvaluateClassifier(X, W1, b1, W2, b2):

    s1 = np.dot(W1, X) + b1
    h = np.maximum(0, s1)  
    s = np.dot(W2, h) + b2
    p = np.exp(s) / np.sum(np.exp(s), axis=0)
   
    return p, s



def ComputeCost(X, Y, W1, b1, W2, b2, lambda_):

    P, _ = EvaluateClassifier(X, W1, b1, W2, b2)
    n = X.shape[1]
    
    loss = - np.sum(Y * np.log(P), axis=0)
    
    regularization_term = lambda_ * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    
    J_cost = (1 / n) * np.sum(loss) + regularization_term
    J_loss = (1 / n) * np.sum(loss)
    
    return J_cost, J_loss



def ComputeAccuracy(X, y, W1, b1, W2, b2):

    p, _ = EvaluateClassifier(X, W1, b1, W2, b2)
    y_pred = np.argmax(p, axis=0) + 1
    acc = accuracy_score(y, y_pred)

    return acc


def ComputeGradients(X, Y, W1, b1, W2, b2, lambda_):

    n = X.shape[1]
    p, _ = EvaluateClassifier(X, W1, b1, W2, b2) 
   
    G = - (Y - p) 
    forward = np.dot(W1,X)+ b1
    grad_W2 = G @ np.maximum(0, forward).T / n + 2 * lambda_ * W2
    grad_b2 = np.sum(G, axis=1).reshape(10, 1) / n

    G = np.dot(W2.T, G)
    G = G * (forward > 0)
   
    grad_W1 = (1 / n) * np.dot(G, X.T) + 2 * lambda_ * W1
    grad_b1 = (1 / n) * np.sum(G, axis=1, keepdims=True) 
    
    return grad_W1, grad_b1, grad_W2, grad_b2

def ComputeGradsNum(X, Y, W1, b1, W2, b2, lambda_, h):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    c,_ = ComputeCost(X, Y, W1, b1, W2, b2, lambda_)

    for i in range(b1.shape[0]):
        b1_try = np.array(b1)
        b1_try[i] = b1_try[i] + h
        c2_1, _ = ComputeCost(X, Y, W1, b1_try, W2, b2, lambda_)

        grad_b1[i] = (c2_1 - c) / h

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] = W1_try[i, j] + h
            c2_1,_ = ComputeCost(X, Y, W1_try, b1, W2, b2, lambda_)

            grad_W1[i, j] = (c2_1 - c) / h

    # Repeat for W2 and b2
    for i in range(b2.shape[0]):
        b2_try = np.array(b2)
        b2_try[i] = b2_try[i] + h
        c2_2,_ = ComputeCost(X, Y, W1, b1, W2, b2_try, lambda_)

        grad_b2[i] = (c2_2 - c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] = W2_try[i, j] + h
            c2_2,_ = ComputeCost(X, Y, W1, b1, W2_try, b2, lambda_)

            grad_W2[i, j] = (c2_2 - c) / h

    return grad_W1, grad_b1, grad_W2, grad_b2


def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, W_1, b_1, W_2, b_2, lambda_, n_batch, eta, n_epochs):
	
    n = X_train.shape[1]
    costs_train = []
    costs_val = []
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []
     
    for epoch in range(n_epochs):
        for j in range(1, n // n_batch):
            j_start = (j-1)*n_batch
            j_end = j*n_batch
            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]
        
            grad_W_1, grad_b_1, grad_W_2, grad_b_2 = ComputeGradients(X_batch, Y_batch, W_1, b_1, W_2, b_2, lambda_)
                
            W_1 = W_1 - eta * grad_W_1
            b_1 = b_1 - eta * grad_b_1

            W_2 = W_2 - eta * grad_W_2
            b_2 = b_2 - eta * grad_b_2

        cost_train, loss_train = ComputeCost(X_train, Y_train, W_1, b_1, W_2, b_2, lambda_)
        costs_train.append(cost_train)
        losses_train.append(loss_train)
        accuracy_train = ComputeAccuracy(X_train, y_train, W_1, b_1, W_2, b_2)
        accuracies_train.append(accuracy_train)

        cost_val, loss_val = ComputeCost(X_val, Y_val, W_1, b_1, W_2, b_2, lambda_)
        costs_val.append(cost_val)
        losses_val.append(loss_val)
        accuracy_val = ComputeAccuracy(X_val, y_val, W_1, b_1, W_2, b_2)
        accuracies_val.append(accuracy_val)

        #if epoch % 10 == 0:
        print("Epoch: ", epoch, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
    return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W_1": W_1, "b_1": b_1, "W_2": W_2, "b_2": b_2}	

def MiniBatchGDCyclic(X_train, Y_train, y_train, X_val, Y_val, y_val, W_1, b_1, W_2, b_2, lambda_, n_batch, eta, n_epochs):
	
    n_s = 900
    eta_min = 1e-5
    eta_max = 1e-1
    cycles = 3
    n = X_train.shape[1]

    costs_train = []
    costs_val = []
    losses_train = []
    losses_val = []
    accuracies_train = []
    accuracies_val = []

    update_list = []


    for cycle in range(cycles):
        for step in range(2*n_s):
            if step <= n_s:
                eta = eta_min + step/n_s * (eta_max - eta_min)
            else:
                eta = eta_max - ((step - n_s)/n_s) * (eta_max - eta_min)

            j_start = (step % (n // n_batch)) * n_batch
            j_end = ((step % (n // n_batch)) + 1) * n_batch
  

            X_batch = X_train[:, j_start:j_end]
            Y_batch = Y_train[:, j_start:j_end]


            grad_W_1, grad_b_1, grad_W_2, grad_b_2 = ComputeGradients(X_batch, Y_batch, W_1, b_1, W_2, b_2, lambda_)

            W_1 = W_1 - eta * grad_W_1
            b_1 = b_1 - eta * grad_b_1

            W_2 = W_2 - eta * grad_W_2
            b_2 = b_2 - eta * grad_b_2


            if step % (n_s/5) == 0:
                cost_train, loss_train = ComputeCost(X_train, Y_train, W_1, b_1, W_2, b_2, lambda_)
                costs_train.append(cost_train)
                losses_train.append(loss_train)
                accuracy_train = ComputeAccuracy(X_train, y_train, W_1, b_1, W_2, b_2)
                accuracies_train.append(accuracy_train)

                cost_val, loss_val = ComputeCost(X_val, Y_val, W_1, b_1, W_2, b_2, lambda_)
                costs_val.append(cost_val)
                losses_val.append(loss_val)
                accuracy_val = ComputeAccuracy(X_val, y_val, W_1, b_1, W_2, b_2)
                accuracies_val.append(accuracy_val)
                update_list.append(step + (2*n_s)*(cycle+1))

                print("Step: ", step, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
    return {"costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val, "W_1": W_1, "b_1": b_1, "W_2": W_2, "b_2": b_2, "update_list": update_list}	


data_1 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_1')
data_2 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_2')
data_3 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_3')
data_4 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_4')
data_5 = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_5')

test_data = load_batch('Assigment 1/Datasets/cifar-10-batches-py/test_batch')

#X_train, Y_train, y_train = preprocess_data(data_1)
#X_val, Y_val, y_val = preprocess_data(data_2)
X_train_1, Y_train_1, y_1 = preprocess_data(data_1)
X_train_2, Y_train_2, y_2 = preprocess_data(data_2)
X_train_3, Y_train_3, y_3 = preprocess_data(data_3)
X_train_4, Y_train_4, y_4 = preprocess_data(data_4)
X_train_5, Y_train_5, y_5 = preprocess_data(data_5)

X_train = np.concatenate((X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1)
Y_train = np.concatenate((Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1)
y_train= np.concatenate((y_1, y_2, y_3, y_4, y_5))

indices = np.random.permutation(X_train.shape[1])
X_val = X_train[:, indices[:1000]]
Y_val = Y_train[:, indices[:1000]]
y_val = y_train[indices[:1000]]

X_train = X_train[:, indices[1000:]]
Y_train = Y_train[:, indices[1000:]]
y_train = y_train[indices[1000:]]

X_test, Y_test, y_test = preprocess_data(test_data)


#n = 10000
K = 10
d = X_train.shape[0]
m = 50
W1, b1, W2, b2 = initialize_parameters(K, d, m)

#l = -5 + (-1 - (-5)) * np.random.rand(1, 1)
#lambda_ = 10**l
#print(lambda_)

# coarse search
#For lambda search over [5e-05, 0.0002, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1 ]
# lambda_ = 5e-05 --> Accuracy= 0.5043
# lambda_ = 0.0002 --> Accuracy= 0.505
# lambda_ = 0.001 --> Accuracy= 0.5125
# lambda_ = 0.005 --> Accuracy= 0.5133
# lambda_ = 0.01 --> Accuracy= 0.51
# lambda_ = 0.02 --> Accuracy= 0.4908
# lambda_ = 0.05 --> Accuracy= 0.4378


# narrower search over [0.007, 0.009, 0.012, 0.015, 0.018]
# lambda_ = 0.007 --> Accuracy= 0.5055
# lambda_ = 0.009 --> Accuracy= 0.5154
# lambda_ = 0.012 --> Accuracy= 0.5059
# lambda_ = 0.015 --> Accuracy= 0.4965
# lambda_ = 0.018 --> Accuracy= 0.4934

lambda_ =  0.009

results_dict = MiniBatchGDCyclic(X_train, Y_train, y_train, X_val, Y_val, y_val, W1, b1, W2, b2, lambda_, 100, 0.01, 10)
test_accuracy = ComputeAccuracy(X_test, y_test, results_dict["W_1"], results_dict["b_1"], results_dict["W_2"], results_dict["b_2"])
print("Test accuracy: ", test_accuracy)


#Compare gradient methods
#grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGradients(X_train[0:20, 1:2],Y_train[:, 1:2], W1, b1, W2,b2, 0.01)
#grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = ComputeGradsNum(X_train[0:20, 1:2], Y_train[:, 1:2], W1, b1, W2, b2, 0.01, 1e-5)

#check for absolut difference < 1e-6
#print(np.allclose(grad_W1, grad_W1_num, atol=1e-6), np.allclose(grad_b1, grad_b1_num, atol=1e-6))
#print(np.allclose(grad_W2, grad_W2_num, atol=1e-6), np.allclose(grad_b2, grad_b2_num, atol=1e-6))

#Check if overfitting is possible
#results_dict = MiniBatchGD(X_train[:,0:100], Y_train[:,0:100], y_train[0:100], X_val[:,0:100], Y_val[:,0:100], y_val[0:100], W1, b1, W2, b2, 0, 10, 0.01, 200)
#test_accuracy = ComputeAccuracy(X_train[:,0:100], y_train[0:100], results_dict["W_1"], results_dict["b_1"], results_dict["W_2"], results_dict["b_2"])
#print("Test accuracy: ", test_accuracy)

#loss plot
plt.figure()
plt.plot(results_dict["losses_train"], label="Training loss")
plt.plot(results_dict["losses_val"], label="Validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# costs plot
plt.figure()
plt.plot(results_dict["costs_train"], label="Training cost")
plt.plot(results_dict["costs_val"], label="Validation cost")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()
