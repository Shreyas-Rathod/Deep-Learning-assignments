# Question 8

import numpy as np
import matplotlib.pyplot as plt
import wandb
from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
os.environ['WANDB_SILENT'] = 'true'

# ---------------------------
# Helper Functions & Losses
# ---------------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    grad = np.ones_like(x)
    grad[x <= 0] = 0
    return grad

def sigmoid(x):
    # Add clipping to prevent overflow
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = np.tanh(x)
    return 1 - t**2

def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[0]
    eps = 1e-10
    y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
    loss_value = -np.sum(y_true * np.log(y_pred_clipped)) / m
    return loss_value

def squared_error_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# ---------------------------
# Neural Network Class
# ---------------------------
class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        optimizer="5",
        momentum=0.9,
        rmsprop_beta=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        epsilon=1e-8,
        weight_init="xavier",
        l2_lambda=0.0005,
        activation="relu",
        loss_type="cross_entropy",
        batch_size=64
    ):
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.depth = len(self.layer_sizes)
        
        self.optim_type = optimizer.lower()
        self.m_coef = momentum
        self.rms_decay = rmsprop_beta
        self.adam_decay1 = adam_beta1
        self.adam_decay2 = adam_beta2
        self.eps = epsilon
        self.time_step = 0
        
        self.l2_lambda = l2_lambda
        self.activation = activation.lower()
        self.loss_type = loss_type.lower()
        self.batch_size = batch_size
        
        self.initialize_parameters(weight_init)
        self.initialize_optimizer_memory()

    def initialize_parameters(self, weight_init):
        self.weights = []
        self.biases = []
        for idx in range(len(self.layer_sizes) - 1):
            in_dim = self.layer_sizes[idx]
            out_dim = self.layer_sizes[idx + 1]
            if weight_init == "random":
                W = np.random.randn(in_dim, out_dim) * 0.01
            elif weight_init == "xavier":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / (in_dim + out_dim))
            elif weight_init == "he":
                W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
            else:
                W = np.random.randn(in_dim, out_dim) * 0.01
            b = np.zeros((1, out_dim))
            self.weights.append(W)
            self.biases.append(b)

    def initialize_optimizer_memory(self):
        self.velocity_w = [np.zeros_like(w) for w in self.weights]
        self.velocity_b = [np.zeros_like(b) for b in self.biases]
        self.squared_grad_w = [np.zeros_like(w) for w in self.weights]
        self.squared_grad_b = [np.zeros_like(b) for b in self.biases]
        self.moment1_w = [np.zeros_like(w) for w in self.weights]
        self.moment1_b = [np.zeros_like(b) for b in self.biases]
        self.moment2_w = [np.zeros_like(w) for w in self.weights]
        self.moment2_b = [np.zeros_like(b) for b in self.biases]

    def _hidden_activation_forward(self, x):
        if self.activation == "relu":
            return relu(x)
        elif self.activation == "sigmoid":
            return sigmoid(x)
        elif self.activation == "tanh":
            return tanh(x)
        else:
            return relu(x)

    def _hidden_activation_derivative(self, x):
        if self.activation == "relu":
            return relu_derivative(x)
        elif self.activation == "sigmoid":
            return sigmoid_derivative(x)
        elif self.activation == "tanh":
            return tanh_derivative(x)
        else:
            return relu_derivative(x)

    def forward(self, X):
        layer_outputs = [X]
        layer_inputs = []
        for i in range(len(self.weights)):
            Z = np.dot(layer_outputs[-1], self.weights[i]) + self.biases[i]
            layer_inputs.append(Z)
            # Always use softmax in output layer for classification
            if i == len(self.weights) - 1:
                A = softmax(Z)
            else:
                A = self._hidden_activation_forward(Z)
            layer_outputs.append(A)
        return layer_outputs, layer_inputs

    def compute_loss(self, y_true, y_pred):
        if self.loss_type == "cross_entropy":
            return cross_entropy_loss(y_pred, y_true)
        elif self.loss_type == "squared_error":
            return squared_error_loss(y_pred, y_true)
        else:
            return cross_entropy_loss(y_pred, y_true)

    def backward(self, layer_outputs, layer_inputs, targets):
        batch_size = targets.shape[0]
        # For cross-entropy with softmax, derivative is (y_pred - targets)
        # For squared error (with our simplified approach), we multiply by 2
        if self.loss_type == "cross_entropy":
            error = layer_outputs[-1] - targets
        elif self.loss_type == "squared_error":
            error = 2 * (layer_outputs[-1] - targets)
        else:
            error = layer_outputs[-1] - targets
        
        weight_grads = []
        bias_grads = []
        for layer_idx in reversed(range(len(self.weights))):
            dW = np.dot(layer_outputs[layer_idx].T, error) / batch_size
            db = np.sum(error, axis=0, keepdims=True) / batch_size
            if self.l2_lambda > 0:
                dW += (self.l2_lambda * self.weights[layer_idx]) / batch_size
            weight_grads.insert(0, dW)
            bias_grads.insert(0, db)
            if layer_idx > 0:
                backprop_error = np.dot(error, self.weights[layer_idx].T)
                deriv = self._hidden_activation_derivative(layer_inputs[layer_idx - 1])
                error = backprop_error * deriv
        return weight_grads, bias_grads


    def update_parameters(self, weight_grads, bias_grads, learning_rate):
      self.time_step += 1
      
      for i in range(len(self.weights)):
          if self.optim_type == "sgd":
              # Standard SGD
              self.weights[i] -= learning_rate * weight_grads[i]
              self.biases[i] -= learning_rate * bias_grads[i]
              
          elif self.optim_type == "momentum":
              # Momentum update
              self.velocity_w[i] = self.m_coef * self.velocity_w[i] - learning_rate * weight_grads[i]
              self.velocity_b[i] = self.m_coef * self.velocity_b[i] - learning_rate * bias_grads[i]
              self.weights[i] += self.velocity_w[i]
              self.biases[i] += self.velocity_b[i]
              
          elif self.optim_type == "rmsprop":
              # RMSProp update
              self.squared_grad_w[i] = self.rms_decay * self.squared_grad_w[i] + (1 - self.rms_decay) * weight_grads[i]**2
              self.squared_grad_b[i] = self.rms_decay * self.squared_grad_b[i] + (1 - self.rms_decay) * bias_grads[i]**2
              
              self.weights[i] -= learning_rate * weight_grads[i] / (np.sqrt(self.squared_grad_w[i]) + self.eps)
              self.biases[i] -= learning_rate * bias_grads[i] / (np.sqrt(self.squared_grad_b[i]) + self.eps)
              
          elif self.optim_type == "adam":
              # Adam update
              self.moment1_w[i] = self.adam_decay1 * self.moment1_w[i] + (1 - self.adam_decay1) * weight_grads[i]
              self.moment1_b[i] = self.adam_decay1 * self.moment1_b[i] + (1 - self.adam_decay1) * bias_grads[i]
              
              self.moment2_w[i] = self.adam_decay2 * self.moment2_w[i] + (1 - self.adam_decay2) * weight_grads[i]**2
              self.moment2_b[i] = self.adam_decay2 * self.moment2_b[i] + (1 - self.adam_decay2) * bias_grads[i]**2
              
              # Bias correction
              moment1_w_corrected = self.moment1_w[i] / (1 - self.adam_decay1**self.time_step)
              moment1_b_corrected = self.moment1_b[i] / (1 - self.adam_decay1**self.time_step)
              moment2_w_corrected = self.moment2_w[i] / (1 - self.adam_decay2**self.time_step)
              moment2_b_corrected = self.moment2_b[i] / (1 - self.adam_decay2**self.time_step)
              
              self.weights[i] -= learning_rate * moment1_w_corrected / (np.sqrt(moment2_w_corrected) + self.eps)
              self.biases[i] -= learning_rate * moment1_b_corrected / (np.sqrt(moment2_b_corrected) + self.eps)
          else:
              # Default to SGD if unknown optimizer
              self.weights[i] -= learning_rate * weight_grads[i]
              self.biases[i] -= learning_rate * bias_grads[i]
          
    def train(self, X, y, epochs=20, learning_rate=0.01):
        y_onehot = np.zeros((y.size, self.layer_sizes[-1]))
        y_onehot[np.arange(y.size), y] = 1
        losses = []
        for epoch in range(epochs):
            layer_outputs, layer_inputs = self.forward(X)
            loss = self.compute_loss(y_onehot, layer_outputs[-1])
            losses.append(loss)
            w_grads, b_grads = self.backward(layer_outputs, layer_inputs, y_onehot)
            self.update_parameters(w_grads, b_grads, learning_rate)
            print(f"Epoch {epoch+1}/{epochs}, Loss ({self.loss_type}): {loss:.4f}")
        return losses

    def predict(self, X):
        layer_outputs, _ = self.forward(X)
        return np.argmax(layer_outputs[-1], axis=1)

# ---------------------------
# Data Loading & Preprocessing
# ---------------------------
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
X_train = train_images.reshape(-1, 28*28) / 255.0
X_test = test_images.reshape(-1, 28*28) / 255.0

# Use 10% of training data as validation set
num_train = X_train.shape[0]
val_size = int(0.1 * num_train)
indices = np.random.permutation(num_train)
X_val = X_train[indices[:val_size]]
y_val = train_labels[indices[:val_size]]
X_train_sub = X_train[indices[val_size:]]
y_train_sub = train_labels[indices[val_size:]]

# ---------------------------
# Experiment Function
# ---------------------------
def run_experiment(loss_type, epochs=20, lr=0.01):
    print(f"\nRunning experiment with {loss_type} loss")
    model = NeuralNetwork(
        input_size=784,
        hidden_layers=[128, 64],
        output_size=10,
        optimizer="adam",  # Change from "5" to "adam"
        weight_init="he",  # Change from "xavier" to "he" initialization
        l2_lambda=0.0005,
        activation="relu",
        loss_type=loss_type
    )
    train_losses = model.train(X_train_sub, y_train_sub, epochs=epochs, learning_rate=lr)
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_acc = np.mean(val_preds == y_val)
    # Evaluate on test set
    test_preds = model.predict(X_test)
    test_acc = np.mean(test_preds == test_labels)
    return train_losses, val_acc, test_acc, model

# ---------------------------
# Run Experiments: Cross-Entropy vs Squared Error
# ---------------------------
epochs = 20
lr = 0.01

# Initialize wandb run for comparison
wandb.init(project="CS24M046_DA6401_Assign1_Q8", name="Loss_Comparison")

# Run experiment with cross entropy loss
ce_losses, ce_val_acc, ce_test_acc, model_ce = run_experiment("cross_entropy", epochs=epochs, lr=lr)

# Run experiment with squared error loss
se_losses, se_val_acc, se_test_acc, model_se = run_experiment("squared_error", epochs=epochs, lr=lr)

# ---------------------------
# Plotting Loss Curves
# ---------------------------
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), ce_losses, label="Cross Entropy Loss", marker='o')
plt.plot(range(1, epochs+1), se_losses, label="Squared Error Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
wandb.log({"loss_comparison": wandb.Image(plt)})
plt.show()

# ---------------------------
# Print and Log Final Accuracy Results
# ---------------------------
print("\nFinal Results:")
print(f"Cross Entropy Loss: Validation Accuracy = {ce_val_acc*100:.2f}%, Test Accuracy = {ce_test_acc*100:.2f}%")
print(f"Squared Error Loss: Validation Accuracy = {se_val_acc*100:.2f}%, Test Accuracy = {se_test_acc*100:.2f}%")

wandb.log({
    "ce_val_accuracy": ce_val_acc,
    "ce_test_accuracy": ce_test_acc,
    "se_val_accuracy": se_val_acc,
    "se_test_accuracy": se_test_acc
})

wandb.finish()
