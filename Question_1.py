import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="CS24M046_DA6401_Assign1", name="Slider-Dynamic-Images")

# Load Fashion-MNIST dataset
(x_train, y_train), (_, _) = fashion_mnist.load_data()

# Define class labels for Fashion-MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Create sliders in wandb
wandb.config.step = 1  # Default step value
wandb.config.index = 0  # Default index value

# Function to fetch images based on step and index
def get_sample_images(step, index):
    sample_images = []
    sample_labels = []
    for i in range(10):  # 10 classes
        idx = np.where(y_train == i)[0]  # Get all indices for class i
        selected_idx = idx[(index + i * step) % len(idx)]  # Select based on slider values
        sample_images.append(x_train[selected_idx])
        sample_labels.append(class_labels[i])
    return sample_images, sample_labels

# Logging images dynamically
for step in range(0, 2):  # Example step range
    for index in range(0, 35, 10):  # Example index range
        sample_images, sample_labels = get_sample_images(step, index)

        # Log images to wandb
        wandb.log({
            "examples": [wandb.Image(img, caption=label) for img, label in zip(sample_images, sample_labels)],
            "Step": step,
            "Index": index
        })

wandb.finish()
