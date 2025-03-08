import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import os
import gzip
import shutil

# Function to download and extract Fashion-MNIST dataset
def download_fashion_mnist():
    # URLs for the Fashion-MNIST dataset
    base_url = "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    
    # Create directory to store data
    data_dir = "./fashion_mnist_data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Download and extract each file
    for key, file_name in files.items():
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Downloading {file_name}...")
            request.urlretrieve(base_url + file_name, file_path)
        
        # Extract the gzipped file
        with gzip.open(file_path, 'rb') as f_in:
            with open(file_path[:-3], 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

# Function to load dataset
def load_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip header
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(-1, 28, 28)

def load_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # Skip header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Download and load the dataset
download_fashion_mnist()

train_images = load_images('./fashion_mnist_data/train-images-idx3-ubyte')
train_labels = load_labels('./fashion_mnist_data/train-labels-idx1-ubyte')
test_images = load_images('./fashion_mnist_data/t10k-images-idx3-ubyte')
test_labels = load_labels('./fashion_mnist_data/t10k-labels-idx1-ubyte')

# Plot 1 sample image for each class
fig, axes = plt.subplots(2, 5, figsize=(10, 5))

for i in range(10):
    idx = np.where(train_labels == i)[0][0]  # Find the first image for each class
    ax = axes[i // 5, i % 5]
    ax.imshow(train_images[idx], cmap='gray')
    ax.set_title(f"Class {i}")
    ax.axis('off')

plt.tight_layout()
plt.show()