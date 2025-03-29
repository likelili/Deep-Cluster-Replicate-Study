import tensorflow as tf

# Function to evaluate the model using true labels
def evaluate_model_with_true_labels(model, X_true, y_true, batch_size=256):
    """
    Evaluate the model using real labels and calculate accuracy.

    Parameters:
    model: Trained Keras model.
    X_true: array-like
        Input features with true labels.
    y_true: array-like
        Ground truth labels corresponding to X_true.
    batch_size: int
        Batch size for evaluation.

    Returns:
    float: Accuracy of the model on the true labels.
    """
    # Check the number of channels; convert grayscale images (1 channel) to 3 channels
    if X_true.shape[-1] == 1:
        X_true = tf.image.grayscale_to_rgb(X_true)

    # Create a TensorFlow dataset
    true_ds = tf.data.Dataset.from_tensor_slices((X_true, y_true)).batch(batch_size)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(true_ds)
    return accuracy

# Function to predict and display model results with real labels
def predict_and_show_results(model, X_true, y_true, class_names, num_images=10):
    """
    Predict and display model results with true labels.

    Parameters:
    model: Trained Keras model.
    X_true: array-like
        Input features with true labels.
    y_true: array-like
        Ground truth labels corresponding to X_true.
    class_names: list
        List of class names for the labels.
    num_images: int
        Number of images to predict and display.
    """
    # Ensure the input data is in RGB format
    if X_true.shape[-1] == 1:
        X_true = tf.image.grayscale_to_rgb(X_true)

    # Perform predictions
    predictions = model.predict(X_true[:num_images])
    predicted_labels = tf.argmax(predictions, axis=1).numpy()

    # Display the predictions alongside the true labels
    print("Predictions vs True Labels:")
    for i in range(num_images):
        print(f"Image {i+1}: Predicted: {class_names[predicted_labels[i]]}, True: {class_names[y_true[i]]}")





