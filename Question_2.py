import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# -------------------------------
# Activation functions and helpers
# -------------------------------
def relu(Z):
    return np.maximum(0, Z)

def relu_derivative(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    # subtract max for numerical stability
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    # Clip predictions to avoid log(0)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    m = predictions.shape[0]
    loss = -np.sum(targets * np.log(predictions)) / m
    return loss

