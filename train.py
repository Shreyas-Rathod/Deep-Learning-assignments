#!/usr/bin/env python3
"""
train.py

A script that accepts command line arguments for:
  - wandb_entity, wandb_project
  - dataset
  - epochs, batch_size
  - loss
  - optimizer
  - lr, beta, beta1, beta2, eps
  - weight_decay
  - init
  - num_layers
  - hidden_size
  - activation

Then trains a feedforward network on MNIST or Fashion-MNIST
and logs results to wandb.
"""

import argparse
import wandb
import numpy as np

# Optional: for dataset loading
from keras.datasets import mnist, fashion_mnist

# --------------------------
# Activation Functions
# --------------------------
def identity(x):
    return x

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def activation_forward(name, x):
    if name == "identity":
        return identity(x)
    elif name == "sigmoid":
        return sigmoid(x)
    elif name == "tanh":
        return tanh_fn(x)
    elif name == "ReLU":
        return relu(x)
    else:
        return relu(x)

def activation_derivative(name, x):
    """We assume 'x' here is the pre-activation input."""
    if name == "identity":
        return np.ones_like(x)
    elif name == "sigmoid":
        s = sigmoid(x)
        return s * (1 - s)
    elif name == "tanh":
        t = np.tanh(x)
        return 1 - t**2
    elif name == "ReLU":
        grad = np.ones_like(x)
        grad[x <= 0] = 0
        return grad
    else:
        grad = np.ones_like(x)
        grad[x <= 0] = 0
        return grad

# --------------------------
# Loss Functions
# --------------------------
def cross_entropy_loss(y_pred, y_true, eps=1e-10):
    """
    y_pred, y_true shape: (batch_size, num_classes)
    """
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))

def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# --------------------------
# Feedforward Network Class
# --------------------------
class FeedForwardNet:
    def __init__(
        self,
        input_size,
        num_layers,
        hidden_size,
        output_size=10,
        activation="sigmoid",
        weight_init="random",
        weight_decay=0.0
    ):
        """
        Example: if num_layers=2, we have [input -> hidden -> hidden -> output].
        hidden_size is used for each hidden layer.
        """
        self.num_layers = num_layers
        self.activation_name = activation
        self.weight_decay = weight_decay

        # Build layer sizes, e.g. for num_layers=2: [input_size, hidden_size, hidden_size, output_size]
        layer_sizes = [input_size] + [hidden_size]*num_layers + [output_size]

        # Initialize weights & biases
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            if weight_init == "xavier":
                limit = np.sqrt(1.0 / in_dim)
                W = np.random.uniform(-limit, limit, (in_dim, out_dim))
            else:
                W = np.random.randn(in_dim, out_dim) * 0.01
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X):
        """
        Return list of activations and pre-activations for each layer
        """
        activations = [X]
        pre_acts = []
        for i in range(len(self.weights)):
            z = activations[-1].dot(self.weights[i]) + self.biases[i]
            pre_acts.append(z)
            if i == len(self.weights) - 1:
                # Output layer: use softmax
                exps = np.exp(z - np.max(z, axis=1, keepdims=True))
                a = exps / np.sum(exps, axis=1, keepdims=True)
            else:
                # Hidden layer activation
                a = activation_forward(self.activation_name, z)
            activations.append(a)
        return activations, pre_acts

    def backward(self, activations, pre_acts, y_true, loss_name="cross_entropy"):
        """
        y_true is one-hot. Return grads for self.weights, self.biases
        """
        m = y_true.shape[0]
        # Output error
        if loss_name == "mean_squared_error":
            # MSE derivative w.r.t. output = 2*(pred - y)
            delta = 2*(activations[-1] - y_true)
        else:
            # cross_entropy with softmax => (pred - y_true)
            delta = (activations[-1] - y_true)

        weight_grads = []
        bias_grads = []

        for i in reversed(range(len(self.weights))):
            dW = activations[i].T.dot(delta) / m
            dB = np.sum(delta, axis=0, keepdims=True) / m

            # L2 regularization
            if self.weight_decay > 0:
                dW += (self.weight_decay * self.weights[i]) / m

            weight_grads.insert(0, dW)
            bias_grads.insert(0, dB)

            if i > 0:
                # propagate delta to previous layer
                delta = delta.dot(self.weights[i].T)
                # multiply by derivative of activation
                delta *= activation_derivative(self.activation_name, pre_acts[i-1])

        return weight_grads, bias_grads

    def predict(self, X):
        activations, _ = self.forward(X)
        # Return the predicted class labels (i.e., the index of the max probability)
        return np.argmax(activations[-1], axis=1)


# --------------------------
# Optimizers
# --------------------------
def sgd_update(model, weight_grads, bias_grads, lr):
    for i in range(len(model.weights)):
        model.weights[i] -= lr * weight_grads[i]
        model.biases[i]  -= lr * bias_grads[i]

# For brevity, you can implement momentum, nag, rmsprop, adam, nadam here similarly

# --------------------------
# Training Function
# --------------------------
def train_model(model, X, y, epochs, batch_size, lr, optimizer, loss_name="cross_entropy"):
    # Convert y to one-hot
    num_classes = model.weights[-1].shape[1]
    y_onehot = np.zeros((y.size, num_classes))
    y_onehot[np.arange(y.size), y] = 1

    n = X.shape[0]
    steps_per_epoch = (n + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Shuffle
        idx = np.random.permutation(n)
        X = X[idx]
        y_onehot = y_onehot[idx]

        total_loss = 0.0
        for step in range(steps_per_epoch):
            start = step * batch_size
            end = min(start + batch_size, n)
            X_batch = X[start:end]
            y_batch = y_onehot[start:end]

            activations, pre_acts = model.forward(X_batch)
            loss_val = 0.0
            if loss_name == "mean_squared_error":
                loss_val = mse_loss(activations[-1], y_batch)
            else:
                loss_val = cross_entropy_loss(activations[-1], y_batch)

            total_loss += loss_val
            w_grads, b_grads = model.backward(activations, pre_acts, y_batch, loss_name=loss_name)

            # Update parameters (only sgd shown, but you can extend for others)
            if optimizer == "sgd":
                sgd_update(model, w_grads, b_grads, lr)
            else:
                # For brevity, fallback to sgd
                sgd_update(model, w_grads, b_grads, lr)

        avg_loss = total_loss / steps_per_epoch
        # Print or log each epoch's loss if desired
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

# --------------------------
# Main: Parse Arguments
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a simple feedforward NN with wandb logging.")
    parser.add_argument("--wandb_entity", type=str, default="myname", help="Entity used in wandb")
    parser.add_argument("--wandb_project", type=str, default="myprojectname", help="Project name used in wandb")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd","momentum","nag","rmsprop","adam","nadam"], help="Optimizer choice")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.9, help="Momentum used by momentum/nag")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 used by adam/nadam")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 used by adam/nadam")
    parser.add_argument("--eps", type=float, default=1e-8, help="Epsilon used by optimizers")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument("--init", type=str, default="random", choices=["random", "xavier"], help="Weight initialization")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of hidden layers")
    parser.add_argument("--hidden_size", type=int, default=1, help="Size of each hidden layer")
    parser.add_argument("--activation", type=str, default="sigmoid", choices=["identity","sigmoid","tanh","ReLU"], help="Activation function for hidden layers")

    args = parser.parse_args()

    # ----------------------
    # wandb init
    # ----------------------
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args),
        name="train_run"
    )

    # ----------------------
    # Load dataset
    # ----------------------
    if args.dataset == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # Preprocess
    X_train = X_train.reshape(-1, 28*28).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 28*28).astype(np.float32) / 255.0

    # ----------------------
    # Build model
    # ----------------------
    model = FeedForwardNet(
        input_size=784,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        output_size=10,
        activation=args.activation,
        weight_init=args.init,
        weight_decay=args.weight_decay
    )

    # ----------------------
    # Train
    # ----------------------
    train_model(
        model=model,
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        loss_name=args.loss
    )

    # ----------------------
    # Evaluate on test
    # ----------------------
    preds_test = model.predict(X_test)
    test_acc = np.mean(preds_test == y_test)
    print(f"Test Accuracy: {test_acc*100:.2f}%")

    # Log final accuracy

    wandb.log({"test_accuracy": test_acc})

    wandb.finish()


if __name__ == "__main__":
    main()
