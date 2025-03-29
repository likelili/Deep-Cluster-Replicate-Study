import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os


# 1. Load the trained model
def load_trained_model(model_path):
    """
    Load a pre-trained Keras model.
    Args:
        model_path: Path to the saved model.
    Returns:
        Loaded Keras model.
    """
    model = load_model(model_path)
    print("Model loaded successfully!")
    return model

# 2. Visualize the first layer's convolutional filters
def visualize_first_layer_filters(model, layer_name, save_dir):
    """
    Visualize all filters (weights) in the specified convolutional layer.
    Args:
        model: Loaded Keras model.
        layer_name: Name of the convolutional layer.
        save_dir: Directory to save visualizations.
    """
    # Get the weights of the first convolutional layer.
    first_conv_layer = model.get_layer(layer_name)
    weights = first_conv_layer.get_weights()

    if len(weights) == 1:
        # Only weights, no biases.
        filters = weights[0]
        print("No biases found in the specified layer.")
    elif len(weights) == 2:
        # Includes weights and biases.
        filters, biases = weights
    else:
        raise ValueError("Unexpected number of weights returned by the layer.")

    print(f"Filter shape: {filters.shape}")  # (height, width, input_channels, num_filters)
    print(f"Number of filters: {filters.shape[-1]}")

    # Normalize the filter weights to the range [0, 1].
    filters_min, filters_max = filters.min(), filters.max()
    filters = (filters - filters_min) / (filters_max - filters_min)

    # Visualize all filters.
    num_filters = filters.shape[-1]  # Number of filters.
    plt.figure(figsize=(12, 12))
    for i in range(num_filters):
        # Extract the three channels of the filter and average them (grayscale conversion).
        filter_rgb = filters[:, :, :, i]
        filter_gray = np.mean(filter_rgb, axis=-1)  # Average across channels to grayscale.
        
        plt.subplot(8, 8, i + 1)  # Assuming there are 64 filters, use an 8x8 grid.
        plt.imshow(filter_gray, cmap='gray')  # Display as a grayscale image.
        plt.axis('off')
        plt.title(f"Filter {i}")
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist.

    save_path = os.path.join(save_dir, f"layer_{layer_name}_allfilters_noactivate.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the visualization.
    print(f"Visualization saved at: {save_path}")
    
    plt.show()

# 3. Main function
def main():
    model_path = "/home/ecbm4040/e4040-2024fall-project-SZQA-zs2699-yq2411-jz3849/model/mobilenet_cluster_model.h5"  # Path to the model.
    save_dir = "images"  # Directory to save visualizations.
    layer_name = "conv1"  # Name of the first convolutional layer.

    # Load the model.
    model = load_model(model_path)

    # Visualize the filters of the first convolutional layer.
    visualize_first_layer_filters(model, layer_name, save_dir)

if __name__ == "__main__":
    main()
