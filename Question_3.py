# Question 3

import numpy as np
from keras.datasets import fashion_mnist

# - Activation function
def relu(input_values):
    result = input_values.copy()
    result[result < 0] = 0
    return result

def relu_derivative(input_values):
    derivative = np.ones_like(input_values)
    derivative[input_values <= 0] = 0
    return derivative

def softmax(logits):
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted_logits)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

def cross_entropy_loss(y_pred, y_true):
    num_samples = y_pred.shape[0]
    small_value = 1e-10  
    y_pred_safe = np.clip(y_pred, small_value, 1 - small_value)
    loss_value = -np.sum(y_true * np.log(y_pred_safe)) / num_samples
    return loss_value


# Neural Network Class 
class NeuralNetwork:

    def __init__(self, input_size, hidden_layers, output_size, optimizer, momentum=0.9, rmsprop_beta=0.9, adam_beta1=0.9, adam_beta2=0.999, epsilon=1e-8):
      
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.depth = len(self.layer_sizes)
        
        self.optim_type = optimizer.lower()
        self.m_coef = momentum
        self.rms_decay = rmsprop_beta
        self.adam_decay1 = adam_beta1
        self.adam_decay2 = adam_beta2
        self.eps = epsilon
        self.time_step = 0
        
        self.initialize_parameters()
        
        self.initialize_optimizer_memory()


    def initialize_parameters(self):
        self.weights = []
        self.biases = []
        
        for idx in range(len(self.layer_sizes) - 1):
            scale = np.sqrt(2.0 / self.layer_sizes[idx])
            W = np.random.normal(0, scale, (self.layer_sizes[idx], self.layer_sizes[idx+1]))
            b = np.zeros((1, self.layer_sizes[idx+1]))
            
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



    def forward(self, X):
       
        layer_outputs = [X]  # First layer output is the input
        layer_inputs = []   # Pre-activation values
        
        for layer_idx in range(len(self.weights)):
            # Calculate the input to the current layer
            Z = np.dot(layer_outputs[-1], self.weights[layer_idx]) + self.biases[layer_idx]
            layer_inputs.append(Z)
            
            # Apply activation function
            if layer_idx == len(self.weights) - 1:  # Output layer
                A = softmax(Z)
            else:  # Hidden layers
                A = relu(Z)
                
            layer_outputs.append(A)
            
        return layer_outputs, layer_inputs


    def backward(self, layer_outputs, layer_inputs, targets):

        batch_size = targets.shape[0]
        weight_gradients = []
        bias_gradients = []
        
        # Start with output layer error
        error = layer_outputs[-1] - targets
        
        # Backpropagate through each layer
        for layer_idx in reversed(range(len(self.weights))):
            # Calculate gradients for this layer
            dW = np.dot(layer_outputs[layer_idx].T, error) / batch_size
            db = np.sum(error, axis=0, keepdims=True) / batch_size
            
            # Store gradients (prepend to list since we're going backwards)
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Calculate error for previous layer if not at input layer
            if layer_idx > 0:
                # Error propagated to previous layer
                backprop_error = np.dot(error, self.weights[layer_idx].T)
                # Apply derivative of activation function (ReLU)
                error = backprop_error * relu_derivative(layer_inputs[layer_idx-1])
                
        return weight_gradients, bias_gradients


    def update_parameters(self, weight_grads, bias_grads, learning_rate):
        
        # Select optimizer algorithm
        if self.optim_type == "1":
            self._sgd_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "2":
            self._momentum_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "3":
            self._nesterov_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "4":
            self._rmsprop_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "5":
            self._adam_update(weight_grads, bias_grads, learning_rate)
        elif self.optim_type == "6":
            self._nadam_update(weight_grads, bias_grads, learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {self.optim_type}")
        # add your optimizer by adding elif...

    def _sgd_update(self, weight_grads, bias_grads, learning_rate):
        """Basic stochastic gradient descent update"""
        for idx in range(len(self.weights)):
            self.weights[idx] -= learning_rate * weight_grads[idx]
            self.biases[idx] -= learning_rate * bias_grads[idx]
    

    def _momentum_update(self, weight_grads, bias_grads, learning_rate):
        """Momentum-based gradient descent update"""
        for idx in range(len(self.weights)):
            # Update velocities with momentum
            self.velocity_w[idx] = self.m_coef * self.velocity_w[idx] - learning_rate * weight_grads[idx]
            self.velocity_b[idx] = self.m_coef * self.velocity_b[idx] - learning_rate * bias_grads[idx]
            
            # Update parameters with velocities
            self.weights[idx] += self.velocity_w[idx]
            self.biases[idx] += self.velocity_b[idx]
    

    def _nesterov_update(self, weight_grads, bias_grads, learning_rate):
        """Nesterov accelerated gradient update"""
        for idx in range(len(self.weights)):
            # Store previous velocities
            old_velocity_w = self.velocity_w[idx].copy()
            old_velocity_b = self.velocity_b[idx].copy()
            
            # Update velocities
            self.velocity_w[idx] = self.m_coef * self.velocity_w[idx] - learning_rate * weight_grads[idx]
            self.velocity_b[idx] = self.m_coef * self.velocity_b[idx] - learning_rate * bias_grads[idx]
            
            # Apply Nesterov correction
            self.weights[idx] += -self.m_coef * old_velocity_w + (1 + self.m_coef) * self.velocity_w[idx]
            self.biases[idx] += -self.m_coef * old_velocity_b + (1 + self.m_coef) * self.velocity_b[idx]
    

    def _rmsprop_update(self, weight_grads, bias_grads, learning_rate):
        """RMSProp update with adaptive learning rates"""
        for idx in range(len(self.weights)):
            # Update squared gradient moving averages
            self.squared_grad_w[idx] = self.rms_decay * self.squared_grad_w[idx] + (1 - self.rms_decay) * (weight_grads[idx] ** 2)
            self.squared_grad_b[idx] = self.rms_decay * self.squared_grad_b[idx] + (1 - self.rms_decay) * (bias_grads[idx] ** 2)
            
            # Update parameters with adaptive learning rate
            self.weights[idx] -= learning_rate * weight_grads[idx] / (np.sqrt(self.squared_grad_w[idx]) + self.eps)
            self.biases[idx] -= learning_rate * bias_grads[idx] / (np.sqrt(self.squared_grad_b[idx]) + self.eps)
    

    def _adam_update(self, weight_grads, bias_grads, learning_rate):
        """Adam optimizer update"""
        self.time_step += 1
        
        for idx in range(len(self.weights)):
            # Update first moment (momentum)
            self.moment1_w[idx] = self.adam_decay1 * self.moment1_w[idx] + (1 - self.adam_decay1) * weight_grads[idx]
            self.moment1_b[idx] = self.adam_decay1 * self.moment1_b[idx] + (1 - self.adam_decay1) * bias_grads[idx]
            
            # Update second moment (RMSProp)
            self.moment2_w[idx] = self.adam_decay2 * self.moment2_w[idx] + (1 - self.adam_decay2) * (weight_grads[idx] ** 2)
            self.moment2_b[idx] = self.adam_decay2 * self.moment2_b[idx] + (1 - self.adam_decay2) * (bias_grads[idx] ** 2)
            
            # Correct bias in first and second moments
            m_hat_w = self.moment1_w[idx] / (1 - self.adam_decay1 ** self.time_step)
            m_hat_b = self.moment1_b[idx] / (1 - self.adam_decay1 ** self.time_step)
            v_hat_w = self.moment2_w[idx] / (1 - self.adam_decay2 ** self.time_step)
            v_hat_b = self.moment2_b[idx] / (1 - self.adam_decay2 ** self.time_step)
            
            # Update parameters
            self.weights[idx] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            self.biases[idx] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.eps)
    

    def _nadam_update(self, weight_grads, bias_grads, learning_rate):
        """Nesterov-accelerated Adam optimizer update"""
        self.time_step += 1
        
        for idx in range(len(self.weights)):
            # Update first and second moments (same as Adam)
            self.moment1_w[idx] = self.adam_decay1 * self.moment1_w[idx] + (1 - self.adam_decay1) * weight_grads[idx]
            self.moment1_b[idx] = self.adam_decay1 * self.moment1_b[idx] + (1 - self.adam_decay1) * bias_grads[idx]
            
            self.moment2_w[idx] = self.adam_decay2 * self.moment2_w[idx] + (1 - self.adam_decay2) * (weight_grads[idx] ** 2)
            self.moment2_b[idx] = self.adam_decay2 * self.moment2_b[idx] + (1 - self.adam_decay2) * (bias_grads[idx] ** 2)
            
            # Bias correction
            m_hat_w = self.moment1_w[idx] / (1 - self.adam_decay1 ** self.time_step)
            m_hat_b = self.moment1_b[idx] / (1 - self.adam_decay1 ** self.time_step)
            v_hat_w = self.moment2_w[idx] / (1 - self.adam_decay2 ** self.time_step)
            v_hat_b = self.moment2_b[idx] / (1 - self.adam_decay2 ** self.time_step)
            
            # Nesterov momentum term
            nesterov_term_w = (self.adam_decay1 * m_hat_w + (1 - self.adam_decay1) * weight_grads[idx] / (1 - self.adam_decay1 ** self.time_step))
            nesterov_term_b = (self.adam_decay1 * m_hat_b + (1 - self.adam_decay1) * bias_grads[idx] / (1 - self.adam_decay1 ** self.time_step))
            
            # Update parameters with Nesterov-accelerated term
            self.weights[idx] -= learning_rate * nesterov_term_w / (np.sqrt(v_hat_w) + self.eps)
            self.biases[idx] -= learning_rate * nesterov_term_b / (np.sqrt(v_hat_b) + self.eps)
    

    def train(self, X, y, epochs, learning_rate, batch_size=64, report_interval=10):
        
        # Convert integer labels to one-hot encoded format
        num_classes = self.layer_sizes[-1]
        y_encoded = np.zeros((y.shape[0], num_classes))
        y_encoded[np.arange(y.shape[0]), y] = 1
        
        num_samples = X.shape[0]
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division
        
        for epoch in range(epochs):
            # Shuffle data for each epoch
            shuffle_indices = np.random.permutation(num_samples)
            X_shuffled = X[shuffle_indices]
            y_shuffled = y_encoded[shuffle_indices]
            
            # Process mini-batches
            for batch_idx in range(num_batches):
                # Get mini-batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                
                # Backward pass
                weight_grads, bias_grads = self.backward(activations, pre_activations, y_batch)
                
                # Update parameters
                self.update_parameters(weight_grads, bias_grads, learning_rate)
            
            # Print progress
            if epoch % report_interval == 0:
                activations, _ = self.forward(X)
                current_loss = cross_entropy_loss(activations[-1], y_encoded)
                print(f"Epoch {epoch}/{epochs}, Loss: {current_loss:.4f}")
    

    def predict(self, X):
        """
        Make predictions for input data
        
        Returns class indices with highest probability
        """
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

