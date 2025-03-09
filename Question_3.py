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

