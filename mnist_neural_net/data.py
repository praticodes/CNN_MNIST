import tensorflow as tf

print("TensorFlow version:", tf.__version__)


def load_mnist():
    """ Return mnist dataset
    """
    mnist = tf.keras.datasets.mnist
    return mnist


def extract_mnist():
    """ Return training images, training labels, testing images, and testing labels for the
    mnist dataset
    """
    mnist = load_mnist()
    (training_images, training_labels), (testing_images, testing_labels) = mnist.load_data()

    return training_images, training_labels, testing_images, testing_labels


def scale_mnist(training_images, testing_images):
    """ Scale the pixel values of the images, so they are between 0 and 1.

    The numbers representing the value of each of the 28 * 28 pixels for an MNIST image
     are between 0 (black) and 255 (white) by default, but it will be easier to work with values
     between 0 and 1.
    """
    training_images, testing_images = training_images / 255.0, testing_images / 255.0
    return training_images, testing_images


def add_channel(images):
    """Add a channel representing the intensity of pixels.

    Typically, we need three channels: one for each primary colour in physics: red, green, and blue.
    But here, our image is grayscale, so we just need one channel for the intensity of "black". This ensures
    convolution will work as expected.
    """
    images = images[..., tf.newaxis]
    return images


def prepare_mnist_data():
    """Combines all the functions in this file.

    Extracts the mnist data by loading it and then splitting it into training images, training labels,
    testing images, and testing labels. Then, scales image pixel values representing the intensity of
    black to be between 0 and 1 instead of 0 and 255. Then, adds an extra channel to the image datasets.
    """
    training_images, training_labels, testing_images, testing_labels = extract_mnist()
    training_images, testing_images = scale_mnist(training_images, testing_images)
    training_images = add_channel(training_images)
    testing_images = add_channel(testing_images)
    return training_images, training_labels, testing_images, testing_labels

# %%
