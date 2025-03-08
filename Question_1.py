# Question 1

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from keras.datasets import fashion_mnist

(train_images, train_labels), (_, _) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Randomly select 7 unique classes out of the 10
np.random.seed(42)  
random_classes = np.random.choice(10, 7, replace=False)

selected_images = []
selected_labels = []

for cls in random_classes:
    indices = np.where(train_labels == cls)[0]
    idx = np.random.choice(indices)
    selected_images.append(train_images[idx])
    selected_labels.append(class_names[cls])

# Function to display the 7 images based on "step" and "index"
def display_images(step=1, index=0):
    fig, axes = plt.subplots(1, 7, figsize=(14, 2))
    for i in range(7):
        mod_idx = (i * step + index) % 7
        axes[i].imshow(selected_images[mod_idx], cmap="gray")
        axes[i].set_title(selected_labels[mod_idx])
        axes[i].axis("off")
    plt.show()

# sliders
step_slider = widgets.IntSlider(min=0, max=2, step=1, value=1, description="Step")
index_slider = widgets.IntSlider(min=0, max=35, step=1, value=0, description="Index")

widgets.interactive(display_images, step=step_slider, index=index_slider)
