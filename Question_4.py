# Question 4 

import numpy as np
import wandb
from keras.datasets import fashion_mnist


# - ACTIVATION FUNCTIONS AND HELPERS

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    grad = np.ones_like(x)
    grad[x <= 0] = 0
    return grad

def sigmoid(x):
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



# - NEURAL NETWORK CLASS WITH L2 REGULARIZATION, MULTIPLE OPTIMIZERS, & WEIGHT INIT

class NeuralNetwork:
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        optimizer="1",
        momentum=0.9,
        rmsprop_beta=0.9,
        adam_beta1=0.9,
        adam_beta2=0.999,
        epsilon=1e-8,
        weight_init="random",
        l2_lambda=0.0,
        activation="relu"
    ):
        """
        Parameters:
          input_size    : int, input dimension (e.g., 784)
          hidden_layers : list of ints, hidden layer sizes (e.g., [128, 64, 32])
          output_size   : int, number of classes (e.g., 10)
          optimizer     : str, one of {"1","2","3","4","5","6"} corresponding to:
                          1: SGD, 2: Momentum, 3: Nesterov, 4: RMSProp, 5: Adam, 6: Nadam
          momentum      : float, momentum coefficient for momentum methods
          rmsprop_beta  : float, decay rate for RMSProp
          adam_beta1    : float, beta1 for Adam/Nadam
          adam_beta2    : float, beta2 for Adam/Nadam
          epsilon       : float, to avoid division by zero
          weight_init   : str, "random" or "xavier"
          l2_lambda     : float, L2 regularization coefficient (weight decay)
          activation    : str, activation for hidden layers ("relu", "sigmoid", "tanh")
        """
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
                limit = np.sqrt(1.0 / in_dim)
                W = np.random.uniform(-limit, limit, (in_dim, out_dim))
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
            if i == len(self.weights) - 1:
                A = softmax(Z)
            else:
                A = self._hidden_activation_forward(Z)
            layer_outputs.append(A)
        return layer_outputs, layer_inputs

    def backward(self, layer_outputs, layer_inputs, targets):
        batch_size = targets.shape[0]
        weight_grads = []
        bias_grads = []
        error = layer_outputs[-1] - targets
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
        if self.optim_type == "1":      # SGD
            self._sgd_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "2":    # Momentum
            self._momentum_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "3":    # Nesterov
            self._nesterov_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "4":    # RMSProp
            self._rmsprop_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "5":    # Adam
            self._adam_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "6":    # Nadam
            self._nadam_update(weight_grads, bias_grads, learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_type}")

    def _sgd_update(self, weight_grads, bias_grads, lr):
        for i in range(len(self.weights)):
            self.weights[i] -= lr * weight_grads[i]
            self.biases[i]  -= lr * bias_grads[i]

    def _momentum_update(self, weight_grads, bias_grads, lr):
        for i in range(len(self.weights)):
            self.velocity_w[i] = self.m_coef * self.velocity_w[i] - lr * weight_grads[i]
            self.velocity_b[i] = self.m_coef * self.velocity_b[i] - lr * bias_grads[i]
            self.weights[i]   += self.velocity_w[i]
            self.biases[i]    += self.velocity_b[i]

    def _nesterov_update(self, weight_grads, bias_grads, lr):
        for i in range(len(self.weights)):
            old_v_w = self.velocity_w[i].copy()
            old_v_b = self.velocity_b[i].copy()
            self.velocity_w[i] = self.m_coef * self.velocity_w[i] - lr * weight_grads[i]
            self.velocity_b[i] = self.m_coef * self.velocity_b[i] - lr * bias_grads[i]
            self.weights[i] += -self.m_coef * old_v_w + (1 + self.m_coef) * self.velocity_w[i]
            self.biases[i]  += -self.m_coef * old_v_b + (1 + self.m_coef) * self.velocity_b[i]

    def _rmsprop_update(self, weight_grads, bias_grads, lr):
        for i in range(len(self.weights)):
            self.squared_grad_w[i] = self.rms_decay * self.squared_grad_w[i] + (1 - self.rms_decay) * (weight_grads[i]**2)
            self.squared_grad_b[i] = self.rms_decay * self.squared_grad_b[i] + (1 - self.rms_decay) * (bias_grads[i]**2)
            self.weights[i] -= lr * weight_grads[i] / (np.sqrt(self.squared_grad_w[i]) + self.eps)
            self.biases[i]  -= lr * bias_grads[i]   / (np.sqrt(self.squared_grad_b[i]) + self.eps)

    def _adam_update(self, weight_grads, bias_grads, lr):
        self.time_step += 1
        for i in range(len(self.weights)):
            self.moment1_w[i] = self.adam_decay1 * self.moment1_w[i] + (1 - self.adam_decay1) * weight_grads[i]
            self.moment1_b[i] = self.adam_decay1 * self.moment1_b[i] + (1 - self.adam_decay1) * bias_grads[i]
            self.moment2_w[i] = self.adam_decay2 * self.moment2_w[i] + (1 - self.adam_decay2) * (weight_grads[i]**2)
            self.moment2_b[i] = self.adam_decay2 * self.moment2_b[i] + (1 - self.adam_decay2) * (bias_grads[i]**2)
            m_hat_w = self.moment1_w[i] / (1 - self.adam_decay1**self.time_step)
            m_hat_b = self.moment1_b[i] / (1 - self.adam_decay1**self.time_step)
            v_hat_w = self.moment2_w[i] / (1 - self.adam_decay2**self.time_step)
            v_hat_b = self.moment2_b[i] / (1 - self.adam_decay2**self.time_step)
            self.weights[i] -= lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            self.biases[i]  -= lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)

    def _nadam_update(self, weight_grads, bias_grads, lr):
        self.time_step += 1
        for i in range(len(self.weights)):
            self.moment1_w[i] = self.adam_decay1 * self.moment1_w[i] + (1 - self.adam_decay1) * weight_grads[i]
            self.moment1_b[i] = self.adam_decay1 * self.moment1_b[i] + (1 - self.adam_decay1) * bias_grads[i]
            self.moment2_w[i] = self.adam_decay2 * self.moment2_w[i] + (1 - self.adam_decay2) * (weight_grads[i]**2)
            self.moment2_b[i] = self.adam_decay2 * self.moment2_b[i] + (1 - self.adam_decay2) * (bias_grads[i]**2)
            m_hat_w = self.moment1_w[i] / (1 - self.adam_decay1**self.time_step)
            m_hat_b = self.moment1_b[i] / (1 - self.adam_decay1**self.time_step)
            v_hat_w = self.moment2_w[i] / (1 - self.adam_decay2**self.time_step)
            v_hat_b = self.moment2_b[i] / (1 - self.adam_decay2**self.time_step)
            nesterov_w = self.adam_decay1 * m_hat_w + (1 - self.adam_decay1) * weight_grads[i] / (1 - self.adam_decay1**self.time_step)
            nesterov_b = self.adam_decay1 * m_hat_b + (1 - self.adam_decay1) * bias_grads[i] / (1 - self.adam_decay1**self.time_step)
            self.weights[i] -= lr * nesterov_w / (np.sqrt(v_hat_w) + self.eps)
            self.biases[i]  -= lr * nesterov_b / (np.sqrt(v_hat_b) + self.eps)



# - DATA LOADING (FASHION-MNIST)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
X_train_full = train_images.reshape(-1, 28*28) / 255.0
X_test = test_images.reshape(-1, 28*28) / 255.0
y_train_full = train_labels
y_test = test_labels



# - WANDB SWEEP TRAINING FUNCTION

def train_sweep():
    run = wandb.init(project="CS24M046_DA6401_Assign1")
    config = wandb.config
    run.name = f"hl_{config.num_hidden_layers}_bs_{config.batch_size}_ac_{config.activation}"
    
    num_hidden_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    l2_lambda = config.l2_lambda
    init_type = config.init_type
    activation = config.activation
    optimizer = config.optimizer
    lr = config.learning_rate
    epochs = config.epochs
    batch_size = config.batch_size

    hidden_layers = [hidden_size] * num_hidden_layers

    model = NeuralNetwork(
        input_size=784,
        hidden_layers=hidden_layers,
        output_size=10,
        optimizer=optimizer,
        weight_init=init_type,
        l2_lambda=l2_lambda,
        activation=activation
    )
    
    num_samples = X_train_full.shape[0]
    val_size = int(0.1 * num_samples)
    indices = np.random.permutation(num_samples)
    X_train_ = X_train_full[indices[val_size:]]
    y_train_ = y_train_full[indices[val_size:]]
    X_val    = X_train_full[indices[:val_size]]
    y_val    = y_train_full[indices[:val_size]]
    
    y_train_oh = np.zeros((y_train_.shape[0], 10))
    y_train_oh[np.arange(y_train_.shape[0]), y_train_] = 1
    y_val_oh = np.zeros((y_val.shape[0], 10))
    y_val_oh[np.arange(y_val.shape[0]), y_val] = 1
    
    m = X_train_.shape[0]
    num_batches = (m + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        perm = np.random.permutation(m)
        X_train_ = X_train_[perm]
        y_train_oh = y_train_oh[perm]
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min(start_idx + batch_size, m)
            X_batch = X_train_[start_idx:end_idx]
            y_batch = y_train_oh[start_idx:end_idx]
            
            layer_outputs, layer_inputs = model.forward(X_batch)
            w_grads, b_grads = model.backward(layer_outputs, layer_inputs, y_batch)
            model.update_parameters(w_grads, b_grads, lr)
        
        train_out, _ = model.forward(X_train_)
        train_loss = cross_entropy_loss(train_out[-1], y_train_oh)
        train_acc  = np.mean(np.argmax(train_out[-1], axis=1) == y_train_)
        
        val_out, _ = model.forward(X_val)
        val_loss = cross_entropy_loss(val_out[-1], y_val_oh)
        val_acc  = np.mean(np.argmax(val_out[-1], axis=1) == y_val)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })
    
    y_test_oh = np.zeros((y_test.shape[0], 10))
    y_test_oh[np.arange(y_test.shape[0]), y_test] = 1
    test_out, _ = model.forward(X_test)
    test_loss = cross_entropy_loss(test_out[-1], y_test_oh)
    test_acc  = np.mean(np.argmax(test_out[-1], axis=1) == y_test)
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": test_acc
    })
    run.finish()


# - SWEEP CONFIGURATION

sweep_config = {
    "method": "random",  # Options: "grid", "random", "bayes"
    "metric": {
        "name": "test_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "l2_lambda": {"values": [0.0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["1", "2", "3", "4", "5", "6"]},
        "batch_size": {"values": [16, 32, 64]},
        "init_type": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]}
    }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, train_sweep)
