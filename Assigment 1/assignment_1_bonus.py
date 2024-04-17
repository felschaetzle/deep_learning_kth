import numpy as np
import pickle
from sklearn.metrics import accuracy_score
import albumentations as albu

def load_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data
    

def preprocess_data(data):
    
        X = np.array(data[b"data"])

        flip_images = albu.HorizontalFlip(p=0.5)
        X = X.reshape(10000, 3, 32, 32)
        X = X.transpose(0,2,3,1).astype("uint8")
        X = [flip_images(image=x)["image"] for x in X]

        X = np.array(X).transpose(0,3,1,2).astype("uint8").reshape(10000, 3*32*32).T

        y = np.array(data[b"labels"])+1
        Y = np.eye(10, dtype=np.double)[np.array(data[b'labels'])].T 

        x_mean = np.mean(X, axis=1).reshape(3072, 1)
        X = X - x_mean

        x_std = np.std(X, axis=1).reshape(3072, 1)
        X = X / x_std

        return X, Y, y


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
    
    y_pred = np.argmax(P, axis=0)+1
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

def MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val,  n_batch, eta, n_epochs, W, b, lambda_):

        costs_train = []
        costs_val = []
        losses_train = []
        losses_val = []
        accuracies_train = []
        accuracies_val = []
        
        for epoch in range(n_epochs):
            
            eta = eta * (0.99**epoch)
            print("Current learning rate: ", eta)

            for j in range(1, X_train.shape[1] // n_batch):
                start = (j - 1) * n_batch
                end = j * n_batch
                X_batch = X_train[:, start:end]
                Y_batch = Y_train[:, start:end]
            
                
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
            
           
            print("Epoch: ", epoch, "Cost: ", cost_train, "Accuracy: ", accuracy_train)
    
        return {"W": W, "b": b, "costs_train": costs_train, "accuracies_train": accuracies_train, "costs_val": costs_val, "losses_train": losses_train, "losses_val": losses_val, "accuracies_val": accuracies_val}

data_1 = load_batch('Datasets/cifar-10-batches-py/data_batch_1')
data_2 = load_batch('Datasets/cifar-10-batches-py/data_batch_2')
data_3 = load_batch('Datasets/cifar-10-batches-py/data_batch_3')
data_4 = load_batch('Datasets/cifar-10-batches-py/data_batch_4')
data_5 = load_batch('Datasets/cifar-10-batches-py/data_batch_5')

test_data = load_batch('Datasets/cifar-10-batches-py/test_batch')

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

W = np.random.normal(loc=0, scale=0.01, size=(10, 3072))
b = np.random.normal(loc=0, scale=0.01, size=(10, 1))

results_dict = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, 100, 0.1, 40, W, b, 0.06)

test_accuracy = ComputeAccuracy(X_test, y_test, results_dict["W"], results_dict["b"])
print("Test accuracy: ", test_accuracy)

