import numpy as np
import matplotlib.pyplot as plt
import pickle

############################################################################################################
# Train and test a two layer network with multiple outputs to classify images
# Train the network using mini-batch gradient descent
# Cost function that computes the cross-entropy loss of the classifier applied to the labelled training data
# L2 regularization term on the weight matrix
############################################################################################################

############################################################################################################
# Data Preprocessing
############################################################################################################

# Load CIFAR-10 data
def load_batch(filename):
    """ Copied from the dataset website """
    with open('Datasets/cifar-10-batches-py/' + filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Use permutation to shuffle the data
def shuffle_data(images, labels):  # images and labels are numpy arrays
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    return shuffled_images, shuffled_labels

def random_permutation(images, labels):
    indices = np.random.permutation(images.shape[0])
    shuffled_images = images[indices]
    shuffled_labels = labels[indices]
    return shuffled_images, shuffled_labels

# Function to one-hot encode the labels
def one_hot_encode(labels):
    # Convert class labels from scalars to one-hot vectors
    one_hot_labels = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

# Function to normalize the images for each pixel
# Changed from Assignment 1
def normalize_images(train_images, val_images, test_images):
    train_mean = np.mean(train_images, axis=0)  # axis = 0 means the mean is computed along the columns, which are the pixels
    train_std = np.std(train_images, axis=0)
    train_norm_images = (train_images - train_mean) / train_std
    val_norm_images = (val_images - train_mean) / train_std
    test_norm_images = (test_images - train_mean) / train_std
    return train_norm_images, val_norm_images, test_norm_images


############################################################################################################
# Preparation for classifier
############################################################################################################

# Initialize the weights and bias
# Use cell arrays to store matrices and vectors
def initialize_params(K_num_classes, d_num_features, m_num_nodes = 50):
    # W1 has size m x d, b1 has size m x 1 and W2 has size K x m, b2 has size K x 1
    # Gaussian distribution with mean 0 and standard deviation 1/sqrt(d) for layer 1 and 1/sqrt(m) for layer 2
    W1 = np.random.normal(0, 1/np.sqrt(d_num_features), (m_num_nodes, d_num_features))
    b1 = np.zeros((m_num_nodes, 1))
    W2 = np.random.normal(0, 1/np.sqrt(m_num_nodes), (K_num_classes, m_num_nodes))
    b2 = np.zeros((K_num_classes, 1))
    return W1, b1, W2, b2

# Softmax function
def softmax(x):  # x here is the scores
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# Compute Network function that returns the final probabilities and the intermediary activation values
def evaluate_classifier(images, W1, b1, W2, b2):
    # From Task: W1 has size m x d and images has size d x n
    #print('Fctn: evaluate_classifier, W1 shape: ', W1.shape, 'images shape: ', images.shape, 'b1 shape: ', b1.shape)
    s1 = np.dot(W1, images.T) + b1 # Shape W1: m x n, images: d x n, b1: m x 1 and s1: m x d
    #print('Fctn: evaluate_classifier, s1 shape: ', s1.shape)
    h = np.maximum(0, s1)  # ReLU activation function
    #print('Fctn: evaluate_classifier, W2 shape: ', W2.shape, 'h shape: ', h.shape, 'b2 shape: ', b2.shape)
    s = np.dot(W2, h) + b2
    #print('Fctn: evaluate_classifier, s shape: ', s.shape)
    probabilities = softmax(s)
    #print('Fctn: evaluate_classifier, probabilities shape: ', probabilities.shape, 'h shape: ', h.shape)
    return probabilities, h

# Compute cross-entropy loss function
def cross_entropy_loss(probabilities, Y_labels_one_hot):
    # Compute the cross-entropy loss
    #print('Fctn: cross_entropy_loss, probabilities shape: ', probabilities.shape, 'Y_labels_one_hot shape: ', Y_labels_one_hot.shape)
    loss = - np.sum(Y_labels_one_hot.T * np.log(probabilities), axis=0)
    # loss = -np.log(np.dot(Y_labels_one_hot, probabilities))
    #print('Fctn: cross_entropy_loss, loss shape: ', loss.shape)
    return loss # Shape loss: n x 1 (n,)

# Compute the cost function
def compute_cost(X_images, Y_labels_one_hot, W1, b1, W2, b2, lamda):
    n_dataset = X_images.shape[0]
    probabilities, _ = evaluate_classifier(X_images, W1, b1, W2, b2)
    #print('Fctn: compute_cost, probabilities shape: ', probabilities.shape, 'Y_labels_one_hot shape: ', Y_labels_one_hot.shape)
    loss = cross_entropy_loss(probabilities, Y_labels_one_hot)
    #print('Fctn: compute_cost, loss shape: ', loss.shape, 'W1 shape: ', W1.shape, 'W2 shape: ', W2.shape)
    reg_term = lamda * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    #print('Fctn: compute_cost, reg_term: ', reg_term)
    # Compute the total cost
    J_total_cost = (1 / n_dataset) * np.sum(loss) + reg_term
    return J_total_cost

# Compute the loss function (cost function without the regularization term)
def compute_loss(X_images, Y_labels_one_hot, W1, b1, W2, b2):
    n_dataset = X_images.shape[0]
    probabilities, _ = evaluate_classifier(X_images, W1, b1, W2, b2)
    loss = cross_entropy_loss(probabilities, Y_labels_one_hot)
    J_loss = (1 / n_dataset) * np.sum(loss)
    return J_loss

# Compute accuracy
def compute_accuracy(X_images, Y_labels_one_hot, W1, b1, W2, b2):
    probabilities, _ = evaluate_classifier(X_images, W1, b1, W2, b2)
    predictions = np.argmax(probabilities, axis=0)
    labels = np.argmax(Y_labels_one_hot, axis=1) # Convert one-hot encoded labels to integers
    accuracy = np.sum(predictions == labels) / len(labels)
    return accuracy




############################################################################################################
# Training two-layer network using mini-batch gradient descent
# Applied to a cost function that computes the cross-entropy loss of the classifier applied to the labelled training data
# L2 regularization term on the weight matrix
############################################################################################################

# Compute the gradients of the cost function for a mini-batch of data given the values computed from the forward pass
def compute_gradients(images, labels_one_hot, W1, b1, W2, b2, lamda):
    n_dataset = images.shape[0]
    #print('n_dataset: ', n_dataset)
    probabilities, h = evaluate_classifier(images, W1, b1, W2, b2) # Shape probabilities: K x n, h: m x n
    #print('Fctn: compute_gradients, probabilities shape: ', probabilities.shape, 'labels_one_hot shape: ', labels_one_hot.shape)
    G = - (labels_one_hot - probabilities.T) # Shape G: d x K
    #print('Fctn: compute_gradients, G shape: ', G.shape, 'h shape: ', h.shape, 'W2 shape: ', W2.shape)
    grad_W2 = (1 / n_dataset) * np.dot(G.T, h.T) + 2 * lamda * W2 # Shape grad_W2: K x m (same dimensions as initialized W2)
    grad_b2 = (1 / n_dataset) * np.sum(G, axis=0, keepdims=True).reshape(-1,1) # Shape grad_b2: K x 1 (same dimensions as initialized b2)
    #print('Fctn: compute_gradients, grad_W2 shape: ', grad_W2.shape, 'G shape: ', G.shape)
    G = np.dot(W2.T, G.T) # Shape G: m x d
    G = G * np.greater(h, 0)  # ReLU activation function
    #print('Fctn: compute_gradients, G shape: ', G.shape, 'images shape: ', images.shape, 'W1 shape: ', W1.shape)
    grad_W1 = (1 / n_dataset) * np.dot(G, images) + 2 * lamda * W1
    grad_b1 = (1 / n_dataset) * np.sum(G, axis=1, keepdims=True) # Shape grad_b1: m x 1 (same dimensions as initialized b1)
    #print('Fctn: compute_gradients, grad_W1 shape: ', grad_W1.shape, 'grad_b1 shape: ', grad_b1.shape, 'grad_W2 shape: ', grad_W2.shape, 'grad_b2 shape: ', grad_b2.shape)
    return grad_W1, grad_b1, grad_W2, grad_b2

# Mini-batch gradient descent
# Generate mini-batches by running through the images sequentially
def generate_mini_batches(images, labels_one_hot, batch_size):
    X_batches = []
    Y_batches = []
    n_dataset = images.shape[0]
    for i in range(0, n_dataset, batch_size):
        X_batch = images[i:i + batch_size, :]
        Y_batch = labels_one_hot[i:i + batch_size, :]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
    return X_batches, Y_batches

# Cyclic eta
def cyclic_eta(t, eta_min, eta_max, n_s): # n_s is the stepsize
    l_val = 4
    for l in range(l_val):
        if 2*l*n_s <= t < (2*l + 1)*n_s:
            eta = eta_min + (t - 2*l*n_s) / n_s * (eta_max - eta_min)
        elif (2*l + 1)*n_s <= t < 2*(l + 1)*n_s:
            eta = eta_max - (t - (2*l + 1)*n_s) / n_s * (eta_max - eta_min)
    return eta


# Mini-batch gradient descent
def mini_batch_gd(train_images, train_labels_one_hot, val_images, val_labels_one_hot, W1, b1, W2, b2, lamda, batch_size, n_epochs, eta_min, eta_max, n_s, cyclic = True, eta = 0.01):
    # Initialize the parameters
    train_cost = []
    val_cost = []
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    eta_val = []
    X_batches, Y_batches = generate_mini_batches(train_images, train_labels_one_hot, batch_size)
    t = 0
    for epoch in range(n_epochs):
        for i in range(len(X_batches)):
            t += 1
            if cyclic:
                eta = cyclic_eta(t, eta_min, eta_max, n_s)
            grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(X_batches[i], Y_batches[i], W1, b1, W2, b2, lamda)
            W1 = W1 - eta * grad_W1
            b1 = b1 - eta * grad_b1
            W2 = W2 - eta * grad_W2
            b2 = b2 - eta * grad_b2
        train_cost.append(compute_cost(train_images, train_labels_one_hot, W1, b1, W2, b2, lamda))
        val_cost.append(compute_cost(val_images, val_labels_one_hot, W1, b1, W2, b2, lamda))
        train_loss.append(compute_loss(train_images, train_labels_one_hot, W1, b1, W2, b2))
        val_loss.append(compute_loss(val_images, val_labels_one_hot, W1, b1, W2, b2))
        train_accuracy.append(compute_accuracy(train_images, train_labels_one_hot, W1, b1, W2, b2))
        val_accuracy.append(compute_accuracy(val_images, val_labels_one_hot, W1, b1, W2, b2))
        eta_val.append(eta)
        if epoch % 10 == 0:
            print("Epoch: ", epoch, "Training Cost: ", train_cost[epoch], "Validation Cost: ", val_cost[epoch])
    return train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy, eta_val, W1, b1, W2, b2


############################################################################################################
# Comparison of Analytical and Numerical Gradients
############################################################################################################

def compute_grads_num(images, labels_one_hot, W1, b1, W2, b2, lamda, h):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    c = compute_cost(images, labels_one_hot, W1, b1, W2, b2, lamda)

    for i in range(b1.shape[0]):
        b1_try = np.array(b1)
        b1_try[i] = b1_try[i] + h
        c2_1 = compute_cost(images, labels_one_hot, W1, b1_try, W2, b2, lamda)

        grad_b1[i] = (c2_1 - c) / h

    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] = W1_try[i, j] + h
            c2_1 = compute_cost(images, labels_one_hot, W1_try, b1, W2, b2, lamda)

            grad_W1[i, j] = (c2_1 - c) / h

    # Repeat for W2 and b2
    for i in range(b2.shape[0]):
        b2_try = np.array(b2)
        b2_try[i] = b2_try[i] + h
        c2_2 = compute_cost(images, labels_one_hot, W1, b1, W2, b2_try, lamda)

        grad_b2[i] = (c2_2 - c) / h

    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] = W2_try[i, j] + h
            c2_2 = compute_cost(images, labels_one_hot, W1, b1, W2_try, b2, lamda)

            grad_W2[i, j] = (c2_2 - c) / h

    return grad_W1, grad_b1, grad_W2, grad_b2


# compute the relative error between a numerically computed gradient and an analytically computed gradient
def relative_error(grad_analytical, grad_numerical):
    return np.abs(grad_analytical - grad_numerical) / max(1e-6, np.linalg.norm(grad_analytical) + np.linalg.norm(
        grad_numerical))  # linalg.norm is the Euclidean norm which is the square root of the sum of the squared values


# Comparison Function
def compare_gradients(X_images_norm, Y_labels_one_hot, W1, b1, W2, b2, lamda, h):
    # Compute the analytical gradients
    grad_W1_anal, grad_b1_anal, grad_W2_anal, grad_b2_anal = compute_gradients(X_images_norm[0:10, :],
                                                                               Y_labels_one_hot[0:10, :], W1, b1, W2,
                                                                               b2, lamda)
    print("Analytical gradients computed")
    # Compute the numerical gradients
    grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = compute_grads_num(X_images_norm[0:10, :],
                                                                                         Y_labels_one_hot[0:10, :],
                                                                                         W1, b1, W2, b2, lamda, h)
    print("Numerical gradients computed")
    # Compare gradient vectors by their absolute difference
    diff_W1 = np.abs(grad_W1_anal - grad_W1_num)
    diff_b1 = np.abs(grad_b1_anal - grad_b1_num)
    diff_W2 = np.abs(grad_W2_anal - grad_W2_num)
    diff_b2 = np.abs(grad_b2_anal - grad_b2_num)
    # If the difference is less than 1e-5, the gradient computation is likely correct
    threshold = 1e-5
    print("Comparing the gradients (numerically and analytically) based on the absolute difference")
    if np.all(diff_W1 < threshold):
        print("The gradient of W1 is correct, max absolute difference of W1", np.max(diff_W1))
    else:
        print("The gradient of W1 is incorrect, max absolute difference of W1", np.max(diff_W1))
    if np.all(diff_b1 < threshold):
        print("The gradient of b1 is correct, max absolute difference of b1", np.max(diff_b1))
    else:
        print("The gradient of b1 is incorrect, max absolute difference of b1", np.max(diff_b1))
    if np.all(diff_W2 < threshold):
        print("The gradient of W2 is correct, max absolute difference of W2", np.max(diff_W2))
    else:
        print("The gradient of W2 is incorrect, max absolute difference of W2", np.max(diff_W2))
    if np.all(diff_b2 < threshold):
        print("The gradient of b2 is correct, max absolute difference of b2", np.max(diff_b2))
    else:
        print("The gradient of b2 is incorrect, max absolute difference of b2", np.max(diff_b2))

    rel_error_W1 = relative_error(grad_W1_anal, grad_W1_num)
    rel_error_b1 = relative_error(grad_b1_anal, grad_b1_num)
    rel_error_W2 = relative_error(grad_W2_anal, grad_W2_num)
    rel_error_b2 = relative_error(grad_b2_anal, grad_b2_num)

    print("Comparing the gradients (numerically and analytically) based on the relative error")
    if np.all(rel_error_W1 < threshold):
        print("The gradient of W1 is correct")
    else:
        print("The gradient of W1 is incorrect")
    if np.all(rel_error_b1 < threshold):
        print("The gradient of b1 is correct")
    else:
        print("The gradient of b1 is incorrect")
    if np.all(rel_error_W2 < threshold):
        print("The gradient of W2 is correct")
    else:
        print("The gradient of W2 is incorrect")
    if np.all(rel_error_b2 < threshold):
        print("The gradient of b2 is correct")
    else:
        print("The gradient of b2 is incorrect")

############################################################################################################
# Display Functions and Plots
############################################################################################################
def montage(W):
    """ Display the image for each label in W """
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 5)  # Create 2x5 subplots
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()


# Plot the training and validation loss
def plot_loss(training_loss, validation_loss):
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

# Plot the cost function
def plot_cost(training_cost, validation_cost):
    plt.plot(training_cost, label='Training cost')
    plt.plot(validation_cost, label='Validation cost')
    plt.xlabel('Epochs')
    plt.ylabel('cost')
    plt.title('Training and Validation cost')
    plt.legend()
    plt.show()

# Plot the training and validation accuracy
def plot_accuracy(training_accuracy, validation_accuracy):
    plt.plot(training_accuracy, label='Training Accuracy')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()

# Plot the learning rate
def plot_learning_rate(eta_val):
    plt.plot(eta_val)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate')
    plt.show()

if __name__ == "__main__":
    ############################################################################################################
    # Data Preprocessing
    ############################################################################################################
    random_seed = 123
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    # Load CIFAR-10 data LoadData function
    train_data = load_batch('data_batch_1')
    val_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    # Extract image data
    train_images = train_data[b'data']  # this command extracts the image data from the dictionary
    train_labels = np.array(train_data[b'labels'])  # this command extracts the labels from the dictionary
    val_images = val_data[b'data']
    val_labels = np.array(val_data[b'labels'])
    test_images = test_data[b'data']
    test_labels = np.array(test_data[b'labels'])

    # Input vector needs to have dimensions d x 1
    print('Input vector dimensions (dxn)', train_images.shape)
    print('Labels vector dimensions (nx1)', train_labels.shape)

    # Shuffle the data
    train_images, train_labels = shuffle_data(train_images, train_labels)
    val_images, val_labels = shuffle_data(val_images, val_labels)
    test_images, test_labels = shuffle_data(test_images, test_labels)
    # Permutation could be used but shuffle_data lead to sufficient results

    # Normalize the IMAGES, changed from assignment 1
    train_norm_images, val_norm_images, test_norm_images = normalize_images(train_images, val_images, test_images)

    # One-hot encode the LABELS
    train_labels_one_hot = one_hot_encode(train_labels)
    val_labels_one_hot = one_hot_encode(val_labels)
    test_labels_one_hot = one_hot_encode(test_labels)
    print('One-hot encoded label dimensions (Kxn)', train_labels_one_hot.shape)


    ############################################################################################################
    # Preparation for multi-linear classifier
    ############################################################################################################

    # Initialize the parameters W (weights) and b (bias)
    K_num_classes = 10
    d_num_features = train_norm_images.shape[1]
    m_num_nodes = 50 # nodes in the hidden layer
    W1, b1, W2, b2 = initialize_params(K_num_classes, d_num_features, m_num_nodes)
    lamda = 0.01
    print('W1 dimensions (mxd)', W1.shape)
    print('b1 dimensions (mx1)', b1.shape)
    print('W2 dimensions (Kxm)', W2.shape)
    print('b2 dimensions (Kx1)', b2.shape)

    ############################################################################################################
    # Comparison of Analytical and Numerical Gradients
    ############################################################################################################

    #h_comp = 1e-5 # h≈1e-5 gave the best precision for the numerical gradients
    #compare_gradients(train_norm_images, train_labels_one_hot, W1, b1, W2, b2, lamda, h_comp)

    ############################################################################################################
    # Training multi-linear classifier using mini-batch gradient descent
    ############################################################################################################

    # Try and train the network on a small amount of the training data (100) without regularization
    batch_size = 100
    #k = 4 # integer between 2 and 8
    #n_s = k * (train_norm_images.shape[0] / batch_size)
    n_s = 800 # n_s is the stepsize for the cyclic learning rate
    num_cycles = 3
    total_iterations = n_s * 2 * num_cycles
    print('Total iterations: ', total_iterations)
    n_epochs = int(total_iterations / (len(train_norm_images) / batch_size))
    print('Number of epochs: ', n_epochs)
    #n_epochs = 200
    eta_min = 1e-5
    eta_max = 1e-1
    eta = 0.01
    cyclic = True
    # only 100 samples are used for training
    # train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy, eta_val, W1_trained, b1_trained, W2_trained, b2_trained = mini_batch_gd(train_norm_images[0:100, :], train_labels_one_hot[0:100, :], val_norm_images[0:100, :], val_labels_one_hot[0:100, :], W1, b1, W2, b2, lamda, batch_size, n_epochs, eta_min, eta_max, n_s, cyclic, eta)
    train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy, eta_val, W1_trained, b1_trained, W2_trained, b2_trained = mini_batch_gd(train_norm_images, train_labels_one_hot, val_norm_images, val_labels_one_hot, W1, b1, W2, b2, lamda, batch_size, n_epochs, eta_min, eta_max, n_s, cyclic, eta)


    ############################################################################################################
    # Display Functions and Plots
    ############################################################################################################

    # Display the original Images
    montage(train_images)

    # Display normalized images
    montage(train_norm_images)

    # Plot the training and validation cost
    plot_cost(train_cost, val_cost)

    # Plot the training and validation loss
    plot_loss(train_loss, val_loss)

    # Plot the training and validation accuracy
    plot_accuracy(train_accuracy, val_accuracy)

    # Plot the learning rate
    plot_learning_rate(eta_val)

    # Compute the accuracy of the trained model
    test_accuracy = compute_accuracy(test_norm_images, test_labels_one_hot, W1_trained, b1_trained, W2_trained, b2_trained)
    print("Test accuracy: ", test_accuracy)
