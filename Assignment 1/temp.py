import numpy as np
import matplotlib.pyplot as plt
from urllib import request
import gzip
import os
import struct

# Function to download and load Fashion-MNIST dataset
def load_fashion_mnist():
    """
    Download and load the Fashion-MNIST dataset without using Keras/TensorFlow
    """
    # URLs for the Fashion-MNIST dataset files
    urls = {
        'train_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'train_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'test_images': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'test_labels': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz'
    }
    
    # Create a directory for the data
    os.makedirs('data', exist_ok=True)
    
    # Download and extract files
    for key, url in urls.items():
        filename = url.split('/')[-1]
        filepath = os.path.join('data', filename)
        
        # Download if not already present
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            request.urlretrieve(url, filepath)
    
    # Load the data
    with gzip.open('data/train-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        X_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    
    with gzip.open('data/train-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        y_train = np.frombuffer(f.read(), dtype=np.uint8)
    
    with gzip.open('data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        X_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
    
    with gzip.open('data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        y_test = np.frombuffer(f.read(), dtype=np.uint8)
    
    return (X_train, y_train), (X_test, y_test)

# Load the Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = load_fashion_mnist()

# Define class names for Fashion-MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a 2x5 grid for displaying one image from each class
plt.figure(figsize=(12, 6))

# Find one example of each class
for i in range(10):
    # Find the first occurrence of class i
    idx = np.where(y_train == i)[0][0]
    
    # Plot in a 2x5 grid
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[i])
    plt.axis('off')

plt.tight_layout()
plt.savefig('fashion_mnist_samples.png')
plt.show()

# Print the shape of the data for reference
print(f"Training data shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Test labels shape: {y_test.shape}")