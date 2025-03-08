# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Download Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define the class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Print dataset shapes to verify data loading
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# Create a figure to display one sample from each class
plt.figure(figsize=(12, 10))

# Find one sample image for each class
samples = {}
for i in range(10):
    # Find the first occurrence of class i in the training set
    idx = np.where(y_train == i)[0][0]
    samples[i] = X_train[idx]

# Create a 2x5 grid to display all 10 classes
for i, (label, image) in enumerate(samples.items()):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"{class_names[label]}")
    plt.axis('off')  # Hide the axes for cleaner visualization

plt.tight_layout()
plt.savefig('fashion_mnist_samples.png')  # Save the figure
plt.show()

# Optional: For documentation with wandb
import wandb

# Initialize wandb run - you need to have wandb installed and be logged in
wandb.init(project="da6401-assignment-1", name="fashion-mnist-visualization")

# Log the samples to wandb
for label, image in samples.items():
    wandb.log({f"{class_names[label]}": wandb.Image(image, caption=class_names[label])})

# Or log the entire grid
wandb.log({"class_samples": wandb.Image(plt)})

# Finish the wandb run
wandb.finish()