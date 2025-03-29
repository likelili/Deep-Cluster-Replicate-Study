import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import cv2
import random
import os

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# 1. Generate initial input image: random noise or real data
def generate_input_image(image_size, channels=1, use_real_image=False, real_image=None):
    """
    Generate input image.
    Args:
        image_size: Size of the input image.
        channels: 1 for grayscale image, 3 for RGB image.
        use_real_image: Whether to use real data as input.
        real_image: Real data image.
    Returns:
        Initial input image tensor.
    """
    if use_real_image and real_image is not None:
        input_image = tf.cast(real_image, tf.float32) / 255.0  # Normalize to [0, 1]
        input_image = tf.image.resize(input_image, [image_size, image_size])  # Resize image

        # If the input is grayscale (single channel), replicate to 3 channels
        if input_image.shape[-1] == 1:
            input_image = tf.repeat(input_image, repeats=3, axis=-1)  # Replicate grayscale to 3 channels

        input_image = tf.expand_dims(input_image, axis=0)  # Add batch dimension
    
    else:
        # Random noise initialization
        input_image = tf.random.uniform((1, image_size, image_size, channels), minval=0.5, maxval=1.0)
        if channels == 1:
            input_image = tf.repeat(input_image, repeats=3, axis=-1)  # Replicate grayscale to 3 channels
    
    return input_image


# 2. Image denoising and standardization
def deprocess_image(img, channels):
    """
    Denoise and standardize the image.
    Args:
        img: Input image tensor.
    Returns:
        Standardized image.
    """
    img = img[0]
    img -= img.mean()
    img /= (img.std() + 1e-5)
    img *= 0.1
    
    # Clip to [0,1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # If the input is grayscale (channels=1), average three channels to ensure grayscale output
    if channels == 1 and img.shape[-1] == 3:
        img = np.mean(img, axis=-1)  # Average three channels to output a grayscale image
    
    
    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype('uint8')
    return img


# 3. Gaussian blur
def apply_gaussian_blur(image, sigma=0.5):
    img_np = image[0].numpy()
    img_np = cv2.GaussianBlur(img_np, (3, 3), sigma)
    
    return tf.convert_to_tensor(img_np, dtype=tf.float32)[tf.newaxis, ...]

    

# 4. Gradient ascent process
def gradient_ascent_with_cross_entropy(model, input_image, layer_name, filter_index, image_size=32, learning_rate=1.0, iterations=30):
    """
    Maximize activation of a specified filter using gradient ascent.
    Args:
        model: Loaded Keras model.
        layer_name: Target convolutional layer name.
        filter_index: Target filter index.
        input_image: Initial input image.
        learning_rate: Learning rate.
        iterations: Number of gradient ascent iterations.
    Returns:
        Optimized image tensor.
    """
    intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    

    # Gradient ascent optimization
    for i in range(iterations):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            # Forward pass to compute the output of the target convolutional layer
            layer_output = intermediate_model(input_image)
            
            # Extract all filter responses
            all_filters_activation = tf.reduce_mean(layer_output, axis=[1, 2])  # Average activation
            
            # Compute the activation value of the target filter
            target_activation = all_filters_activation[:, filter_index]
            
            # Compute cross-entropy loss: softmax transformation -> cross-entropy
            softmax_output = tf.nn.softmax(all_filters_activation)
            cross_entropy = -tf.math.log(softmax_output[:, filter_index] + 1e-5)  # Avoid log(0)
            
            # Define loss function: negative activation + cross-entropy
            l2_loss = tf.reduce_sum(tf.square(input_image))
            loss = -target_activation + cross_entropy + 0.01 * l2_loss

            # loss = -target_activation + cross_entropy
                    
        grads = tape.gradient(loss, input_image)
        grads = tf.math.l2_normalize(grads)
        input_image += learning_rate * grads
        
        if i % 5 == 0:
            input_image = apply_gaussian_blur(input_image)

        # Print intermediate loss values
        # if i % 10 == 0:
        #     print(f"Iteration {i}, Loss: {loss.numpy()[0]}")
    
    return input_image.numpy()


# 5. Visualize filter activation results
def visualize_filter(model_path, layer_name, filter_index, input_image=None, image_size=32, channels=1, use_real_image=False):
    """
    Visualize the activation results of a specific filter.
    Args:
        model_path: Model path.
        layer_name: Target convolutional layer name.
        filter_index: Filter index.
        input_image: Real input image (optional).
        image_size: Image size.
        channels: Input channel count (1 for grayscale, 3 for RGB).
        use_real_image: Whether to use real data as input.
        save_dir: Directory to save the visualizations (default: "images").
    """
    # Load model
    model = load_model(model_path)

    # Generate initial input image (noise or real data)
    input_image = generate_input_image(image_size, channels, use_real_image, input_image)

    # Perform gradient ascent
    optimized_image = gradient_ascent_with_cross_entropy(model, input_image, layer_name, filter_index, image_size)

    # Denoise the processed image
    processed_image = deprocess_image(optimized_image, channels)
    
    if channels == 3:
        plt.imshow(processed_image)
    else:
        plt.imshow(processed_image, cmap='gray')
    
    plt.title(f"Layer: {layer_name}, Filter: {filter_index}")
    plt.axis('off')
    
    # Save to specified folder
#     os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist

#     save_path = os.path.join(save_dir, f"layer_{layer_name}_filter_{filter_index}_GA.png")
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     print(f"Visualization saved at: {save_path}")
    
    plt.show()


# 6. Parameters and execution
if __name__ == "__main__":
    model_path = "/home/ecbm4040/e4040-2024fall-project-SZQA-zs2699-yq2411-jz3849/model/mobilenet_cluster_model.h5"  # Model path
    save_dir = "images"  # Save directory
    layer_name = "conv1"  # Target convolutional layer name
    filter_index = 5   # Filter index for visualization
    image_size = 32  # Input image size
    channels = 1  # 1 for grayscale, 3 for RGB

    # Real dataset: CIFAR-10 (e.g., use the 0th image)
    (X_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    X_train_gray = tf.image.rgb_to_grayscale(X_train)
    real_image = X_train_gray[1]  # Select the first image

    # Visualize results: use real data or random noise
    visualize_filter(model_path, layer_name, filter_index, input_image=real_image, image_size=image_size, channels=channels, use_real_image=None)
