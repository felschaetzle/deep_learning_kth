import numpy as np
import matplotlib.pyplot as plt
import pickle


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
def normalize_images(images):
    mean = np.mean(images, axis=0)  # axis = 0 means the mean is computed along the columns, which are the pixels
    std = np.std(images, axis=0)
    norm_images = (images - mean) / std
    return norm_images


############################################################################################################
# Preparation for multi-linear classifier
############################################################################################################

# Initialize the parameters W (weight) and b (bias) randomly
# W has size K×d and b is K×1
def initialize_params(K_num_classes, d_num_features):  # K is the number of classes and d is the number of features (pixels)
    W = np.random.normal(0, 0.01, (K_num_classes, d_num_features))  # Initialize each entry to have Gaussian random values with zero mean and standard deviation .01
    b = np.zeros((K_num_classes, 1))
    return W, b


# Softmax function
def softmax(x):  # x here is the scores
    return np.exp(x) / np.sum(np.exp(x), axis=0) # https://deepnotes.io/softmax-crossentropy


# Function to evaluate the network function
def evaluate_classifier(X_images, W, b):
    # Compute the softmax
    scores = np.dot(W, X_images.T) + b
    probabilities = softmax(scores)
    return probabilities  # Matrix K x N (K is the number of classes and N is the number of images)


# Create cross-entropy loss function
def cross_entropy_loss(probabilities, Y_labels_one_hot):
    # Compute the cross-entropy loss
    loss = -np.log(np.diag(np.dot(Y_labels_one_hot, probabilities)))
    # loss = -np.log(np.dot(Y_labels_one_hot, probabilities))
    return loss

# Compute loss which is cost without the regularization, needed for the report
def compute_loss(X_images, Y_labels_one_hot, W, b):
    n_dataset = X_images.shape[0]
    # Compute the softmax probabilities
    probabilities = evaluate_classifier(X_images, W, b)
    # Compute the cross-entropy loss
    loss = cross_entropy_loss(probabilities, Y_labels_one_hot)
    # Compute the total loss without regularization term
    J_loss = (1 / n_dataset) * np.sum(loss)
    return J_loss

# Function to compute the cost function
def compute_cost(X_images, Y_labels_one_hot, W, b, lamda):
    n_dataset = X_images.shape[0]
    # Compute the softmax probabilities
    probabilities = evaluate_classifier(X_images, W, b)
    # Compute the cross-entropy loss
    loss = cross_entropy_loss(probabilities, Y_labels_one_hot)
    # Compute the regularization term
    reg_term = 0.5 * lamda * np.sum(W ** 2)
    # Compute the total cost
    J_total_cost = (1 / n_dataset) * np.sum(loss) + reg_term
    return J_total_cost


# Compute accuracy function of networks predictions
def compute_accuracy(X_images, Y_labels_one_hot, W, b):
    probabilities = evaluate_classifier(X_images, W, b)
    predictions = np.argmax(probabilities, axis=0)
    labels = np.argmax(Y_labels_one_hot, axis=1) # Convert one-hot encoded labels to integers
    accuracy = np.sum(predictions == labels) / len(labels)
    return accuracy


############################################################################################################
# Training multi-linear classifier using mini-batch gradient descent
############################################################################################################

def compute_gradients(X_images, Y_labels_one_hot, W, b, lamda):
    n_dataset = X_images.shape[0]
    # Compute the softmax probabilities
    probabilities = evaluate_classifier(X_images, W, b)
    # Compute the gradients
    G = -(Y_labels_one_hot.T - probabilities)
    grad_W = (1 / n_dataset) * np.dot(G, X_images) + lamda * W
    grad_b = (1 / n_dataset) * np.dot(G, np.ones((n_dataset, 1)))
    return grad_W, grad_b


# Generate mini-batches by running through the images sequentially
def generate_mini_batches(X_images, Y_labels_one_hot, batch_size):
    X_batches = []  # List to store the mini-batches of images
    Y_batches = []
    for i in range(0, X_images.shape[0], batch_size):  # range(start, stop, step)
        X_batch = X_images[i:i + batch_size, :]  # Generates an array of images from i to i+batch_size
        Y_batch = Y_labels_one_hot[i:i + batch_size, :]
        X_batches.append(X_batch)
        Y_batches.append(Y_batch)
    return X_batches, Y_batches


# Mini-batch learning function
# Including the generation of training loss and validation loss data
def mini_batch_GD(train_images, train_labels_one_hot, val_images, val_labels_one_hot, W, b, lamda, n_batch, eta, n_epochs):
    training_cost = []
    validation_cost = []
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    X_batches, Y_batches = generate_mini_batches(train_images, train_labels_one_hot, n_batch)
    for epoch in range(n_epochs):
        for i in range(len(X_batches)):
            X_batch = X_batches[i]
            Y_batch = Y_batches[i]
            # The heart of the training process
            grad_W, grad_b = compute_gradients(X_batch, Y_batch, W, b, lamda)
            W = W - eta * grad_W # WEIGHT UPDATE
            b = b - eta * grad_b # BIAS UPDATE
        # Compute the training cost
        J_train_cost = compute_cost(train_images, train_labels_one_hot, W, b, lamda)
        training_cost.append(J_train_cost)
        # Compute the validation cost
        J_val_cost = compute_cost(val_images, val_labels_one_hot, W, b, lamda)
        validation_cost.append(J_val_cost)
        # Compute the training loss
        J_train_loss = compute_loss(train_images, train_labels_one_hot, W, b)
        training_loss.append(J_train_loss)
        # Compute the validation loss
        J_val_loss = compute_loss(val_images, val_labels_one_hot, W, b)
        validation_loss.append(J_val_loss)
        # Compute the training accuracy
        acc_train = compute_accuracy(train_images, train_labels_one_hot, W, b)
        training_accuracy.append(acc_train)
        # Compute the validation accuracy
        acc_val = compute_accuracy(val_images, val_labels_one_hot, W, b)
        validation_accuracy.append(acc_val)
    return W, b, training_cost, validation_cost, training_loss, validation_loss, training_accuracy, validation_accuracy


############################################################################################################
# Comparison of Analytical and Numerical Gradients
############################################################################################################

# Compute the gradient vectors of the cost function numerically
def compute_grads_num_centered_diff(X, Y, W, b, lamda, h):  # slow
    # Converted from matlab code
    no = W.shape[0]
    d = X.shape[0]

    grad_W = np.zeros(W.shape);
    grad_b = np.zeros((no, 1));

    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = compute_cost(X, Y, W, b_try, lamda)

        b_try = np.array(b)
        b_try[i] += h
        c2 = compute_cost(X, Y, W, b_try, lamda)

        grad_b[i] = (c2 - c1) / (2 * h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.array(W)
            W_try[i, j] -= h
            c1 = compute_cost(X, Y, W_try, b, lamda)

            W_try = np.array(W)
            W_try[i, j] += h
            c2 = compute_cost(X, Y, W_try, b, lamda)

            grad_W[i, j] = (c2 - c1) / (2 * h)

    return [grad_W, grad_b]


# compute the relative error between a numerically computed gradient and an analytically computed gradient
def relative_error(grad_analytical, grad_numerical):
    return np.abs(grad_analytical - grad_numerical) / max(1e-6, np.linalg.norm(grad_analytical) + np.linalg.norm(
        grad_numerical)) # linalg.norm is the Euclidean norm which is the square root of the sum of the squared values


# Comparison Function
def compare_gradients(X_images_norm, Y_labels_one_hot, W, b, lamda, h):
    # Compute the analytical gradients
    grad_W_anal, grad_b_anal = compute_gradients(X_images_norm[0:100, :], Y_labels_one_hot[0:100, :], W, b, lamda)
    # Compute the numerical gradients
    grad_W_num, grad_b_num = compute_grads_num_centered_diff(X_images_norm[0:100, :], Y_labels_one_hot[0:100, :], W, b, lamda, h)
    # Compare gradient vectors (matrices) by their absolute differences
    diff_W = np.abs(grad_W_anal - grad_W_num)
    diff_b = np.abs(grad_b_anal - grad_b_num)
    #print("Difference in W: ", diff_W)
    #print("Difference in b: ", diff_b)
    # If the difference is less than 1e-6, the gradient computation is likely correct
    print("Comparing the gradients (numerically and analytically) based on the difference")
    if np.all(diff_W < 1e-6):
        print("Gradient of W is correct")
    else:
        print("Gradient of W is incorrect")
    if np.all(diff_b < 1e-6):
        print("Gradient of b is correct")
    else:
        print("Gradient of b is incorrect")

    rel_error_W = relative_error(grad_W_anal, grad_W_num)
    rel_error_b = relative_error(grad_b_anal, grad_b_num)
    print("Relative error in W: ", rel_error_W)
    print("Relative error in b: ", rel_error_b)

    # If the relative error is less than 1e-6, the gradient computation is likely correct
    print("Comparing the gradients (numerically and analytically) based on the relative error")
    if np.all(rel_error_W < 1e-6):
        print("Gradient of W is correct")
    else:
        print("Gradient of W is incorrect")
    if np.all(rel_error_b < 1e-6):
        print("Gradient of b is correct")
    else:
        print("Gradient of b is incorrect")


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

# Compute Histogram of the probability for the ground truth class for examples correctly and incorrectly classified
def compute_histograms(X_images, Y_labels, W, b):
    probabilities = evaluate_classifier(X_images, W, b)
    predictions = np.argmax(probabilities, axis=0)
    labels = Y_labels
    correct_indices = np.where(predictions == labels)[0]
    incorrect_indices = np.where(predictions != labels)[0]
    correct_probabilities = probabilities[labels[correct_indices], correct_indices]
    incorrect_probabilities = probabilities[labels[incorrect_indices], incorrect_indices]
    plt.hist(correct_probabilities, bins=10, alpha=0.5, label='Correctly classified')
    plt.hist(incorrect_probabilities, bins=10, alpha=0.5, label='Incorrectly classified')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.title('Histogram of the probability for the ground truth class')
    plt.suptitle('Cross-entropy loss')
    plt.legend()
    plt.show()



def visualize_test_samples(test_images_norm, test_labels, W_trained, b_trained, class_names, num_samples=20, num_columns=5):
    random_indices = np.random.choice(test_images_norm.shape[0], size=num_samples, replace=False)
    # Evaluate
    probabilities = evaluate_classifier(test_images_norm, W_trained, b_trained)
    predictions = np.argmax(probabilities, axis=0)
    confidence_percentages = np.max(probabilities, axis=0) * 100

    # Plot the test samples
    num_rows = int(np.ceil(num_samples / num_columns)) # ceil rounds up to the nearest integer
    plt.figure(figsize=(15, 5 * num_rows))
    for i, idx in enumerate(random_indices):
        image = test_images_norm[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape the image for plotting
        # Need to Clip image pixel values to be in the range [0, 1]
        image = np.clip(image, 0, 1) # https://numpy.org/doc/stable/reference/generated/numpy.clip.html

        plt.subplot(num_rows, num_columns, i + 1)
        plt.imshow(image) # imshow displays data as an image (https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html)
        plt.title(
            f'Predicted: {class_names[predictions[idx]]}\nConfidence: {confidence_percentages[idx]:.2f}%\nActual: {class_names[test_labels[idx]]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ############################################################################################################
    # Data Preprocessing
    ############################################################################################################
    random_seed = 123
    np.random.seed(random_seed)  # Set the random seed for reproducibility

    # Load CIFAR-10 data LoadData function
    # data is a dictionary with keys: 'batch_label', 'labels', 'data', 'filenames'
    # Use 'data_batch_1' for training and 'data_batch_2' for validation and 'test_batch' for testing
    train_data = load_batch('data_batch_1')
    val_data = load_batch('data_batch_2')
    test_data = load_batch('test_batch')

    # Extract image data
    train_images = train_data[b'data']  # this command extracts the image data from the dictionary
    train_labels = np.array(train_data[b'labels'])  # this command extracts the labels from the dictionary
    train_filenames = train_data[b'filenames']
    val_images = val_data[b'data']
    val_labels = np.array(val_data[b'labels'])
    val_filenames = val_data[b'filenames']
    test_images = test_data[b'data']
    test_labels = np.array(test_data[b'labels'])
    test_filenames = test_data[b'filenames']

    print('Train images shape: ', train_images.shape)
    print('Train labels shape: ', train_labels.shape)
    print('Validation images shape: ', val_images.shape)
    print('Validation labels shape: ', val_labels.shape)
    print('Test images shape: ', test_images.shape)
    print('Test labels shape: ', test_labels.shape)

    # Shuffle the data
    train_images, train_labels = shuffle_data(train_images, train_labels)
    val_images, val_labels = shuffle_data(val_images, val_labels)
    test_images, test_labels = shuffle_data(test_images, test_labels)

    # Permutation could be used but shuffle_data lead to sufficient results

    # Normalize the images
    train_images_norm = normalize_images(train_images)
    val_images_norm = normalize_images(val_images)
    test_images_norm = normalize_images(test_images)

    # One-hot encode the labels
    train_labels_one_hot = one_hot_encode(train_labels)
    val_labels_one_hot = one_hot_encode(val_labels)
    test_labels_one_hot = one_hot_encode(test_labels)

    ############################################################################################################
    # Preparation for multi-linear classifier
    ############################################################################################################

    # Initialize the parameters W (weight) and b (bias) randomly
    W, b = initialize_params(10, 3072)  # 10 classes and 3072 features (32x32x3)
    lamda = 0.1

    # Compute the cost function
    J_cost = compute_cost(train_images_norm, train_labels_one_hot, W, b, lamda)
    print("Cost function: ", J_cost)

    # Compute the accuracy
    acc = compute_accuracy(train_images_norm, train_labels_one_hot, W, b)
    print("Initial accuracy: ", acc)

    ############################################################################################################
    # Comparison of Analytical and Numerical Gradients
    ############################################################################################################

    h_comp = 1e-6
    compare_gradients(train_images_norm, train_labels_one_hot, W, b, lamda, h_comp)

    ############################################################################################################
    # Training multi-linear classifier
    ############################################################################################################

    # Set the hyperparameters
    n_batch = 64
    eta = 0.001
    n_epochs = 40

    # Train the multi-linear classifier using mini-batch gradient descent
    W_trained, b_trained, training_cost, validation_cost, training_loss, validation_loss, training_accuracy, validation_accuracy = mini_batch_GD(
        train_images_norm, train_labels_one_hot, val_images_norm, val_labels_one_hot, W, b, lamda, n_batch,
        eta, n_epochs)

    ############################################################################################################
    # Display Functions and Plots
    ############################################################################################################

    # Display the original Images
    montage(train_images)

    # Display normalized images
    montage(train_images_norm)

    # Display the learned weight matrix
    montage(W_trained)

    # Plot the training and validation cost
    plot_cost(training_cost, validation_cost)

    # Plot the training and validation loss
    plot_loss(training_loss, validation_loss)

    # Plot the training and validation accuracy
    plot_accuracy(training_accuracy, validation_accuracy)

    # Compute the accuracy of the trained model
    acc_train = compute_accuracy(train_images_norm, train_labels_one_hot, W_trained, b_trained)
    print("Final training accuracy: ", acc_train)
    acc_val = compute_accuracy(val_images_norm, val_labels_one_hot, W_trained, b_trained)
    print("Final validation accuracy: ", acc_val)
    acc_test = compute_accuracy(test_images_norm, test_labels_one_hot, W_trained, b_trained)
    print("Final test accuracy: ", acc_test)

    # CIFAR-10 class names from the dataset website (https://www.cs.toronto.edu/~kriz/cifar.html)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # Visualize a sample of test images with their predicted classification labels
    visualize_test_samples(test_images_norm, test_labels, W_trained, b_trained, class_names, num_samples=20, num_columns=5)


    # Compute Histogram of the probability for the ground truth class for examples correctly and incorrectly classified
    compute_histograms(test_images_norm, test_labels, W_trained, b_trained)
