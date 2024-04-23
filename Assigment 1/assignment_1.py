import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        
    X = np.array(data[b'data'], dtype=np.double) / 255  
    X = X.T  

    Y = np.eye(10, dtype=np.double)[np.array(data[b'labels'])].T  
    y = np.array(data[b'labels']) + 1  
    
    return X, Y, y

def preprocess_data(data):
    
    mean_ = np.mean(data, axis=1, keepdims=True)
    std_ = np.std(data, axis=1, keepdims=True)
    
    
    norm_data = (data - mean_) / std_
   
    
    return norm_data


X_train, Y_train, y_train = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_1')
X_val, Y_val, y_val = load_batch('Assigment 1/Datasets/cifar-10-batches-py/data_batch_2')
X_test, Y_test, y_test = load_batch('Assigment 1/Datasets/cifar-10-batches-py/test_batch')

X_train = preprocess_data(X_train)
X_val = preprocess_data(X_val)
X_test = preprocess_data(X_test)


W = np.random.normal(loc=0, scale=0.01, size=(Y_train.shape[0], X_train.shape[0]))
b = np.random.normal(loc=0, scale=0.01, size=(Y_train.shape[0], 1))


def EvaluateClassifier(X, W, b):
    
    S = np.dot(W, X) + b
    
    exp_scores = np.exp(S)
    P = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    
    return P


def ComputeCost(X, Y, W, b, lambda_):

    P = EvaluateClassifier(X, W, b)
    
    d = X.shape[1]
    cross_entropy_loss = np.sum(Y * (np.log(P)*(-1))) / d
    
    regularization_term = lambda_ * np.sum(W * W)
    
    J = cross_entropy_loss + regularization_term
    
    return J, cross_entropy_loss


def ComputeAccuracy(X, y, W, b):

    P = EvaluateClassifier(X, W, b)
    
    y_pred = np.argmax(P, axis=0) + 1
    acc = accuracy_score(y, y_pred)
    
    return acc


def ComputeGradients(X, Y, W, b, lambda_):

    P = EvaluateClassifier(X, W, b)

    n = X.shape[1]
    
    # Compute gradients of the cost function w.r.t. W
    grad_W = (np.dot(-(Y - P), X.T)/n) + lambda_ * W
    
    # Compute gradient of the cost function w.r.t. b
    grad_b = np.mean(-(Y - P), axis=1, keepdims=True)
    
    return grad_b, grad_W

#grad_W, grad_b = ComputeGradients(X_train[:, :100], Y_train[:, 1],P, W, 0.1)

def ComputeGradsNumSlow(X, Y, W, b, lambda_, h):
    # Initialize gradients
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    
    # Compute gradients for b
    for i in range(len(b)):
        b_try = b.copy()
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lambda_)
        
        b_try = b.copy()
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lambda_)
        
        grad_b[i] = (c2 - c1) / (2 * h)
    
    # Compute gradients for W
    for i in range(W.size):
        W_try = W.copy()
        W_try.flat[i] -= h
        c1 = ComputeCost(X, Y, W_try, b, lambda_)
        
        W_try = W.copy()
        W_try.flat[i] += h
        c2 = ComputeCost(X, Y, W_try, b, lambda_)
        
        grad_W.flat[i] = (c2 - c1) / (2 * h)
    
    return grad_b, grad_W

def ComputeGradsNum(X, Y, W, b, lamda, h):
    # Number of samples
    n = X.shape[1]
    
    # Number of classes
    no = W.shape[0]
    
    # Number of features
    d = X.shape[0]
    
    # Initialize gradients
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    
    # Compute cost for current parameters
    c = ComputeCost(X, Y, W, b, lamda)
    
    # Compute gradients with respect to b
    for i in range(len(b)):
        b_try = np.copy(b)
        b_try[i] = b_try[i] + h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h
    
    # Compute gradients with respect to W
    for i in range(W.size):
        W_try = np.copy(W)
        W_try.flat[i] = W_try.flat[i] + h
        c2 = ComputeCost(X, Y, W_try, b, lamda)
        grad_W.flat[i] = (c2 - c) / h
    
    return grad_b, grad_W


#grad_b_1, grad_W_1 = ComputeGradsNum(X_train[:, :20], Y_train[:, :20], W, b, 0, 1e-6)

#grad_W_2, grad_b_2 = ComputeGradients(X_train[:, :20], Y_train[:, :20],P[:, :20], W, 0)


#check for absolut difference < 1e-6
#print(np.allclose(grad_W_1, grad_W_2, atol=1e-6), np.allclose(grad_b_1, grad_b_2, atol=1e-6))

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val,  n_batch, eta, n_epochs, W, b, lambda_):

        costs_train = []
        costs_val = []
        losses_train = []
        losses_val = []
        accuracies_train = []
        accuracies_val = []
        
        for epoch in range(n_epochs):
            for j in range(1, X_train.shape[1] // n_batch):
                start = (j - 1) * n_batch
                end = j * n_batch
                X_batch = X_train[:, start:end]
                Y_batch = Y_train[:, start:end]
                
                # Forward pass
                P = EvaluateClassifier(X_batch, W, b)
                
                # Compute gradients
                grad_b, grad_W = ComputeGradients(X_batch, Y_batch, W, b, lambda_)
                
                # Update parameters
                W -= eta * grad_W
                b -= eta * grad_b

            cost_train, loss_train = ComputeCost(X_train, Y_train, W, b, lambda_)
            costs_train.append(cost_train)
            losses_train.append(loss_train)
            accuracy_train = ComputeAccuracy(X_train, y_train, W, b)
            accuracies_train.append(accuracy_train)

            cost_val, loss_val = ComputeCost(X_val, Y_val, W, b, lambda_)
            costs_val.append(cost_val)
            losses_val.append(loss_val)
            accuracy_val = ComputeAccuracy(X_val, y_val, W, b)
            accuracies_val.append(accuracy_val)
            
            #if epoch % 10 == 0:
            print("Epoch: ", epoch, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
    
        return {"W": W, "b": b, "costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val}

def visualize_weights(W):
    # Reshape each row of W into a set of images
    images = []
    for i in range(W.shape[0]):
        im = np.reshape(W[i, :], (32, 32, 3), order='F')
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        im = np.transpose(im, (1, 0, 2))
        images.append(im)
    
    return images



results_dict = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, 100, 0.001, 40, W, b, 1)

test_accuracy = ComputeAccuracy(X_test, y_test, results_dict["W"], results_dict["b"])
print("Test accuracy: ", test_accuracy)

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


images = visualize_weights(results_dict["W"])
for im in images:
     plt.imshow(im)
     plt.show()