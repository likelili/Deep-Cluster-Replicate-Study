import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model, Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize
y_train, y_test = tf.keras.utils.to_categorical(y_train), tf.keras.utils.to_categorical(y_test)
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

# Load trained MobileNet model
model_path = "/home/ecbm4040/e4040-2024fall-project-SZQA-zs2699-yq2411-jz3849/model/mobilenet_cluster_model.h5"
model = load_model(model_path)
print("Model loaded successfully!")

# Function to extract activations
def get_activations(model, layer_name, data, batch_size=1024):
    layer_output = model.get_layer(layer_name).output
    activation_model = Model(inputs=model.input, outputs=layer_output)

    activations_list = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        batch_activations = activation_model.predict(batch_data)
        activations_list.append(batch_activations)

    activations = np.concatenate(activations_list, axis=0)
    return activations

# Function to train a linear classifier
def train_linear_classifier(activations, labels, activations_test, labels_test, n_components=None):
    X_train = activations.reshape(activations.shape[0], -1)
    X_test = activations_test.reshape(activations_test.shape[0], -1)

    # Dimensionality reduction with PCA
    if n_components:
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # Logistic Regression
    y_train_labels = np.argmax(labels, axis=1)
    y_test_labels = np.argmax(labels_test, axis=1)

    clf = LogisticRegression(max_iter=5000, solver='saga', multi_class='multinomial')
    clf.fit(X_train, y_train_labels)
    predictions = clf.predict(X_test)

    acc = accuracy_score(y_test_labels, predictions)
    return acc


if __name__ == "__main__":
    # Layers to evaluate
    layers = ["conv1", "conv_pw_2", "conv_pw_3", "conv_pw_4", "conv_pw_5"]  # Add more as needed
    results = {}

    # Evaluate each layer
    for layer in layers:
        print(f"Processing layer: {layer}")
        train_activations = get_activations(model, layer, x_train)
        test_activations = get_activations(model, layer, x_test)

        # Train and evaluate classifier
        acc = train_linear_classifier(train_activations, y_train, test_activations, y_test, n_components=30)
        results[layer] = acc
        print(f"Accuracy on layer {layer}: {acc:.4f}")

    print("Final results:", results)
