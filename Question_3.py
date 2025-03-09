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

