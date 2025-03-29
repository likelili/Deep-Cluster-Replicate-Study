import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Data augmentation function
def data_augmentation(img, label):
    # ensure the image is rgb
    img = tf.image.grayscale_to_rgb(img) if img.shape[-1] == 1 else img
    
    #randomly choose one way to augument the data:No augumentation; random crop; horizontal flip; adjust the brightness;
    choice = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    
    def no_augmentation():
        return img

    def random_crop():
        cropped_img = tf.image.random_crop(img, size=[28, 28, 3])
        return tf.image.resize(cropped_img, (32, 32))  # resize back to 32*32 as imagenet need

    def random_flip():
        return tf.image.random_flip_left_right(img)

    def random_brightness():
        return tf.image.adjust_brightness(img, delta=0.15)
    
    img = tf.switch_case(
        branch_index=choice,
        branch_fns={
            0: no_augmentation,
            1: random_crop,
            2: random_flip,
            3: random_brightness
        })
    return img, label

# Preprocess the data
def preprocess_data(X, y):
    X = tf.cast(X, tf.float32) / 255.0  # ensure X is float32 and / 255
    y = y.flatten()
    return X, y

def plot_cifar10(X, y, names, shape=(32, 32, 3), num_images=16):
    """
    Visualize images from the CIFAR-10 dataset with their corresponding class labels.

    Parameters:
    X: array-like
        Image data, shape (num_samples, height, width, channels).
    y: array-like
        Label data, shape (num_samples, 1).
    names: list
        List of class names corresponding to CIFAR-10 labels.
    shape: tuple, optional
        Shape of a single image (default is CIFAR-10 shape: (32, 32, 3)).
    num_images: int, optional
        Number of images to display (default is 16).
    """
    fig = plt.figure(figsize=(10, 10))  # Create a figure of size 10x10 inches
    for i in range(num_images):
        ax = fig.add_subplot(4, 4, i+1)  # Add a subplot in a 4x4 grid
        ax.imshow(X[i].reshape(shape) / 255.0)  # Display the image (normalized to [0, 1])
        ax.set_title(names[int(y[i][0])])  # Set the title to the class name
        ax.axis('off')  # Turn off the axis for cleaner display
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()  # Display the figure

def plot_preprocessed_data(train_ds, class_names, num_images=16):
    """
    Visualize preprocessed images with their corresponding class labels.

    Parameters:
    train_ds: tf.data.Dataset
        Training dataset containing image and label pairs.
    class_names: list
        List of class names corresponding to the labels.
    num_images: int, optional
        Number of images to display (default is 16).
    """
    # Prepare figure
    fig = plt.figure(figsize=(10, 10))
    rows = cols = int(np.sqrt(num_images))  # Create a square grid (e.g., 4x4)

    # Take a batch of data
    for images, labels in train_ds.take(1):  # Take 1 batch
        for i in range(num_images):
            ax = fig.add_subplot(rows, cols, i+1)  # Create subplots
            img = images[i].numpy()  # Convert Tensor to NumPy array
            label = labels[i].numpy()  # Get label
            
            ax.imshow(img)  # Display image
            ax.set_title(class_names[int(label)])  # Convert label to class name
            ax.axis('off')  # Remove axis for clarity

    plt.tight_layout()
    plt.show()




