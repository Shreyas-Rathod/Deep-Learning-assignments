# Question 2 

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

# Initialize wandb
wandb.init(project="CS24M046_DA6401_Assign1", name="NN_with_prob_dist")

# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def cross_entropy_loss(predictions, y_true):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.mean(np.sum(y_true * np.log(predictions), axis=1))


# --- Definition of Neural Network Class ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2 / self.layers[i]) 
                        for i in range(len(self.layers) - 1)]
        self.biases = [np.zeros((1, self.layers[i+1])) for i in range(len(self.layers) - 1)]

    def forward_prop(self, X):
        activations = [X]
        pre_act = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1].dot(w) + b
            pre_act.append(z)
            # Use softmax in the output layer; use relu in hidden layers.
            if i == len(self.weights) - 1:
                a = softmax(z)
            else:
                a = relu(z)
            activations.append(a)
        
        return activations, pre_act

    def backward_prop(self, activations, pre_act, y_true):
        m = y_true.shape[0]
        deltas = [None] * len(self.weights)
        deltas[-1] = activations[-1] - y_true  # derivative of softmax + cross-entropy

        for i in reversed(range(len(self.weights) - 1)):
            deltas[i] = deltas[i+1].dot(self.weights[i+1].T) * relu_derivative(pre_act[i])

        grad_weights = [activations[i].T.dot(deltas[i]) / m for i in range(len(self.weights))]
        grad_biases = [np.sum(deltas[i], axis=0, keepdims=True) / m for i in range(len(self.weights))]
        
        return grad_weights, grad_biases

    def update_weights(self, grad_weights, grad_biases, learning_rate):
        self.weights = [w - learning_rate * dw for w, dw in zip(self.weights, grad_weights)]
        self.biases = [b - learning_rate * db for b, db in zip(self.biases, grad_biases)]

    def predict(self, X):
        activations, _ = self.forward_prop(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Return the softmax probability distribution over the 10 classes."""
        activations, _ = self.forward_prop(X)
        return activations[-1]

    def train(self, X, y, epochs, learning_rate, batch_size=128, print_every=10):
        # mini-batch gradient descent.
        # X: input features, y: labels (not one-hot encoded)
        y_onehot = np.zeros((y.size, self.layers[-1]))
        y_onehot[np.arange(y.size), y] = 1

        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            X_shuffled = X[indices]
            y_shuffled = y_onehot[indices]

            for start in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[start:start + batch_size]
                y_batch = y_shuffled[start:start + batch_size]

                activations, pre_act = self.forward_prop(X_batch)
                grad_weights, grad_biases = self.backward_prop(activations, pre_act, y_batch)
                self.update_weights(grad_weights, grad_biases, learning_rate)

            if epoch % print_every == 0:
                activations, _ = self.forward_prop(X)
                loss = cross_entropy_loss(activations[-1], y_onehot)
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")


# --- Load and preprocess data ---
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
X_train = train_images.reshape(-1, 784) / 255.0
X_test = test_images.reshape(-1, 784) / 255.0

# --- Initialize and train network ---
hidden_layers = [256, 128]  # Change size or number of hidden layers as desired
network = NeuralNetwork(784, hidden_layers=hidden_layers, output_size=10)
network.train(X_train, train_labels, epochs=60, learning_rate=0.01, batch_size=128, print_every=5)

# --- Evaluate ---
predictions = network.predict(X_test)
accuracy = np.mean(predictions == test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# --- Log probability distribution graph to wandb ---
# Get the probability distribution (softmax outputs) for test data.
probabilities = network.predict_proba(X_test)  # shape: (num_samples, 10)
# Compute the average predicted probability for each class
mean_probs = np.mean(probabilities, axis=0)
class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot a bar chart of the average probabilities per class
fig, ax = plt.subplots()
ax.bar(np.arange(10), mean_probs, color='skyblue')
ax.set_xticks(np.arange(10))
ax.set_xticklabels(class_labels, rotation=45, ha='right')
ax.set_xlabel("Class")
ax.set_ylabel("Average Predicted Probability")
ax.set_title(f"Average Predicted Probability Distribution over 10 Classes (Test Set)\nwith {len(hidden_layers)} hidden layers of size {hidden_layers}, Test Accuracy: {accuracy * 100:.2f}%")

# Log the figure to wandb
wandb.log({"Probability Distribution": wandb.Image(fig)})

plt.show()
wandb.finish()
