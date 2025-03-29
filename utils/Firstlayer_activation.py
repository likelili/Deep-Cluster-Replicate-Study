import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
import cv2


# 1. Load real data
def load_real_images(image_size=32):
    """
    Load the CIFAR-10 dataset and preprocess images.
    Args:
        image_size: Size of the input images (default 32x32).
    Returns:
        Image data: shape (batch_size, image_size, image_size, 3)
    """
    (X_train, _), (_, _) = cifar10.load_data()  # Load only the training set
    X_train = X_train[:10000]  # Select the first 10,000 images
    X_train = np.array([tf.image.resize(img, (image_size, image_size)) for img in X_train])
    X_train = X_train / 255.0  # Normalize to [0, 1]
    return X_train


# 2. Visualize the response of all filters in the first convolutional layer
def visualize_filters_with_real_images(model, layer_name, images, save_dir, channels=3):
    """
    Visualize all filters in a specified convolutional layer using real data.
    Args:
        model: Trained Keras model.
        layer_name: Name of the convolutional layer.
        images: Input image data.
        save_dir: Directory to save the visualizations.
        channels: Number of image channels (3 for RGB).
    """
    # Get the output of the specified layer
    layer = model.get_layer(layer_name)
    activation_model = Model(inputs=model.input, outputs=layer.output)

    # Compute activations for all images
    activations = activation_model.predict(images)
    num_filters = activations.shape[-1]  # Number of filters

    print(f"Visualizing activations of {num_filters} filters in layer '{layer_name}'")

    # Plot responses for all filters
    plt.figure(figsize=(15, 15))
    for filter_index in range(num_filters):
        # Extract the activation for the specific filter
        filter_activations = activations[:, :, :, filter_index]

        # Average the activations across all images
        mean_activation = np.mean(filter_activations, axis=0)

        # Visualize the filter response
        plt.subplot(4, 8, filter_index + 1)  # Assuming a maximum of 36 filters to display
        plt.imshow(mean_activation, cmap='gray')
        plt.axis('off')
        plt.title(f"Filter {filter_index}")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist

    save_path = os.path.join(save_dir, f"layer_{layer_name}_allfilters_2.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved at: {save_path}")
    
    plt.show()


# 4. Main function: load the model, data, and visualize
def main():
    # Parameter settings
    model_path = "/home/ecbm4040/e4040-2024fall-project-SZQA-zs2699-yq2411-jz3849/model/mobilenet_cluster_model.h5"  # Model path
    save_dir = "images"  # Save directory path
    layer_name = "conv1"  # Name of the first convolutional layer
    image_size = 32  # Input image size
    channels = 3  # RGB images

    # Load the model
    model = load_model(model_path)
    print("Model loaded successfully!")

    # Load real image data
    images = load_real_images(image_size=image_size)
    print("Original images loaded.")

    # Visualize the filter responses
    visualize_filters_with_real_images(model, layer_name, images, save_dir, channels=channels)


if __name__ == "__main__":
    main()
