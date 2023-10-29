import tensorflow as tf
from keras import datasets, layers, models

# Load the full MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Select one image and its label
image = train_images[:1]
label = train_labels[:1]

print("Image")
print(image)
print("Label")
print(label)


#%%
