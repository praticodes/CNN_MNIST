from keras import layers, models

from mnist_neural_net.data import prepare_mnist_data


def create_initial_sequential_model():
    """ Creates a sequential model.

    A sequentia model is good for our purposes as it allows data to simply flow forward from one
    layer to the next, and the MNIST problem can be solved with a feedforward neural network (each layer
    can learn increasingly complex features made out of features from the previous layer as the MNISt
    dataset is well-structured).
    """
    model = models.Sequential()
    return model


def add_convolutions_and_pooling(model):
    """ Add convolution layers and pooling layers.

    Each convolutional layer is responsible for detecting increasingly complex features of our data.
    The main operation in each convolutional layer is the convolution operation, which involves
    representing filters called kernels as matrices and convolving or sliding them across the input data
    matrix and computing a dot product to determine how present those features are in different areas
    of the input matrix (representing a digit). The different filters or kernels have a flexible size, represented
    by the kernel size parameter. So convolutions involve linear transformations,
    but often the relationships between features and the data will not be linear and an activation function helps us
    account for this. The activation function is applied  to each dot product result

    #TODO: How exactly do these activation functions help us uncover more complex, non-linear patterns in data?
    #TODO: How does the choice of activation function influence the type of features the CNN learns and represents?

    We use max pooling after each convolutional layer. For each region in a feature map (obtained from the convolution),
    the maximum value is selected and retained. This considerably simplifies the feature map, which means:
     - the CNN is able to focus on the most important features, thus minimizing chances of over-training)
     - translational invariance is introduced, preventing over-training as small shifts in feature location don't matter
     - enables hierarchical feature learning
     - reduces computational complexity

    #TODO: How does pooling enable hierarchical feature learning?

    """
    # First convolutional layer with 32 filters and 5x5 kernel size
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    # Max-pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))
    # Second convolutional layer with 64 filters and 5x5 kernel size
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    # Max-pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))


def manage_connections(model):
    """
    Our features maps are matrices, so they're 2D. This doesn't allow for us to fully connect layers.
    Therefore, we must flatten our feature maps for all our layers, since we want them all to be fully connected.
    Each of the 256 hidden units learns to recognize different combinations of the features extracted
    by the convolutional and pooling layers. The next 10 hidden units each corresponds to one of the 10 possible classes
    in the MNIST dataset (digits 0 to 9).

    #TODO: Why ReLU and why Softmax?
    """
    # Flatten the output for the fully connected layers
    model.add(layers.Flatten())

    # First fully connected layer with 256 hidden units and ReLU activation
    model.add(layers.Dense(256, activation='relu'))

    # Second fully connected layer with 10 hidden units (for the 10 classes) and softmax activation
    model.add(layers.Dense(10, activation='softmax'))


def compile_model(model):
    """ Compiles the model.

    Our compilation uses stochastic gradient descent to minimize loss.

    To actually calculate loss, we are using sparse Categorical Cross-Entropy (Sparse CCE),
    which is a loss function commonly used in deep learning for multi-class classification problems.
    It's a variation of the more general Categorical Cross-Entropy loss (CCE),
    but for when the target labels are provided as integers (true in our case) and not one-hot encoded vectors.
    """
    model.compile(optimizer='sgd',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def create_model():
    """Returns a suitable and trained model for a CNN for MNIST using all the functions in this file, alongside the
    testing images and testing labels so that its accuracy can be determined.
    """
    model = create_initial_sequential_model()
    add_convolutions_and_pooling(model)
    manage_connections(model)
    compile_model(model)
    training_images, training_labels, testing_images, testing_labels = prepare_mnist_data()
    model = model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))

    return model, testing_images, testing_labels


def evaluate_model(model, testing_images, testing_labels):
    """Evaluates the given model and prints the accuracy.
    """
    # Evaluate the model
    test_loss, test_acc = model.evaluate(testing_images, testing_labels)
    print(f'Test accuracy: {test_acc}')


if __name__ == "__main__":
    my_model, my_testing_images, my_testing_labels = create_model()
    evaluate_model(my_model, my_testing_images, my_testing_labels)
