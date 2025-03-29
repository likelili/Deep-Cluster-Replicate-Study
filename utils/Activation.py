### Visualization - Top 9 activated pictures

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# 1. Calculate activations for the specified layer
def get_layer_activations_batched(model, layer_name, input_data, batch_size=512):
    """
    Calculate activations for the specified layer in batches.
    Args:
        model: Trained Keras model
        layer_name: Name of the layer to extract activations from
        input_data: Input data (images)
        batch_size: Number of images to process at once
    Returns:
        Activation values as a Tensor
    """
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    num_samples = input_data.shape[0]
    activations = []
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_data = input_data[start_idx:end_idx]
        batch_activations = intermediate_model.predict(batch_data, verbose=1)
        activations.append(batch_activations)
    print(len(activations))
    return np.concatenate(activations, axis=0)

# 2. Find the top 9 images with the highest activations
def find_top_activations(activations, selected_filter, top_n=9):
    """
    Find the indices of the images with the highest activations for a specific filter.
    Args:
        activations: Activation values for the target layer (batch_size, height, width, num_filters)
        selected_filter: Target filter index
        top_n: Number of images to return with the highest activations
    Returns:
        Indices of the top N images
    """
    # Calculate the mean activation values for the specified filter
    mean_activations = np.mean(activations[:, :, :, selected_filter], axis=(1, 2))
    # Sort activations and find the top N indices
    top_indices = np.argsort(mean_activations)[::-1][:top_n]
    return top_indices

# 3. Visualize the top 9 activated images
def visualize_top_activations(layer_name, input_data, top_indices, selected_filter, y_labels=None, save_dir="images"):
    """
    Visualize the top N images with the highest activations.
    Args:
        input_data: Input image data
        top_indices: Indices of the top N images
        selected_filter: Target filter index
        y_labels: Image labels (optional)
    """
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(top_indices):
        plt.subplot(3, 3, i + 1)
        plt.imshow(tf.squeeze(input_data[idx]))  # Display the original image
        title = f"Filter {selected_filter}"
        if y_labels is not None:
            title += f", Label: {y_labels[idx]}"
        plt.title(title)
        plt.axis('off')
        
    plt.tight_layout()
    
    # Save the entire 3x3 grid of images to a local directory
    grid_save_path = os.path.join(save_dir, f"conv_{layer_name}_filter_{selected_filter}_top9_grid.png")
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    plt.savefig(grid_save_path, dpi=300, bbox_inches='tight')  # Save the grid image

    plt.show()

# 4. Main function: load model, compute activations, and visualize
def main(model_path, layer_name, selected_filter, top_n=9):
    """
    Main function: load the model, compute activations, and visualize the top N images.
    Args:
        model_path: Path to the model
        layer_name: Target convolutional layer name
        selected_filter: Selected filter index
        top_n: Number of images with the highest activations to display
    """
    # Load the pre-trained model
    model = load_model(model_path)

    # Load the input dataset (CIFAR-10 example)
    (X_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train_gray = tf.image.rgb_to_grayscale(X_train)  # Convert to grayscale

    # Convert grayscale images to float32 and normalize to [0, 1]
    X_train_gray = tf.cast(X_train_gray, tf.float32) / 255.0

    # Duplicate to 3 channels to match the model input
    X_train_rgb = tf.repeat(X_train_gray, repeats=3, axis=-1)

    # Extract activations for the specified layer
    print(f"Extracting activations for layer: {layer_name} ...")
    activations = get_layer_activations_batched(model, layer_name, X_train_rgb, batch_size=2048)

    # Find the top 9 images with the highest activations for the specified filter
    print(f"Finding top {top_n} images for filter {selected_filter} ...")
    top_indices = find_top_activations(activations, selected_filter, top_n)

    # Visualize the results
    visualize_top_activations(layer_name, X_train, top_indices, selected_filter, y_labels=y_train)

# 5. Set parameters and run
if __name__ == "__main__":
    model_path = "../model/mobilenet_cluster_model.h5"  # Model path
    layer_name = "conv_pw_13"  # Target convolutional layer name
    selected_filter = 5  # Selected filter index
    top_n = 9  # Number of top images to display
    save_dir = "images"

    main(model_path, layer_name, selected_filter, top_n)
