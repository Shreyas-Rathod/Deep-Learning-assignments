# Question 1

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from keras.datasets import fashion_mnist
import wandb

wandb.init(project="CS24M046_DA6401_Assign1_Q1")

(train_images, train_labels), (_, _) = fashion_mnist.load_data()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# To store one image per class
unique_images = {}
for img, label in zip(train_images, train_labels):
    if label not in unique_images:
        unique_images[label] = img
    if len(unique_images) == 10:  # Stops when all classes are found
        break

sorted_images = [unique_images[i] for i in range(10)]
sorted_labels = [class_names[i] for i in range(10)]

def display_images(step=1, index=0):
    """
    Shows 10 images in a 4x3 grid (3 per row).
    The last 2 subplots remain blank.
    """
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    axes = axes.ravel()

    for i in range(10):
        img_idx = (i * step + index) % 10  # warp around 10 through images
        axes[i].imshow(sorted_images[img_idx], cmap="gray")
        axes[i].set_title(sorted_labels[img_idx])
        axes[i].axis("off")

    for j in range(10, 12):
        axes[j].axis("off")

    wandb.log({
        "step": step,
        "index": index,
        "fashion_mnist_grid": wandb.Image(fig, caption=f"Step={step}, Index={index}")
    })

    plt.show()

step_slider = widgets.IntSlider(min=1, max=5, step=1, value=1, description="Step")
index_slider = widgets.IntSlider(min=0, max=35, step=1, value=0, description="Index")

widgets.interactive(display_images, step=step_slider, index=index_slider)
