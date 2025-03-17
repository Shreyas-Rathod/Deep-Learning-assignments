# (DA6400) Deep Learning Assignment 1

This repository contains code solutions for eight questions (Q1–Q8) related to **Fashion-MNIST** classification, backpropagation, hyperparameter tuning, and loss function comparisons. Each question is implemented in a separate Python file:

- **Question_1.py**  
- **Question_2.py**  
- **Question_3.py**  
- **Question_4.py**  
- **Question_5.py**  
- **Question_6.py**  
- **Question_7.py**  
- **Question_8.py**

## Table of Contents

1. [Overview of Questions](#overview-of-questions)  
2. [Prerequisites and Setup](#prerequisites-and-setup)  
3. [Running Each Question](#running-each-question)  
4. [WandB Integration](#wandb-integration)  
5. [Key Observations and Insights](#key-observations-and-insights)  
6. [Repository Structure](#repository-structure)  
7. [License](#license)  

---

## Overview of Questions

### Question 1
**Goal**:  
- Download the Fashion-MNIST dataset and plot **one sample image** for each of the 10 classes.  
- Uses `from keras.datasets import fashion_mnist` for data loading.

**File**: `Question_1.py`  
**Key Steps**:
1. Load Fashion-MNIST.  
2. Select 1 image per class.  
3. Display a grid of 10 images.

---

### Question 2
**Goal**:  
- Implement a **feedforward neural network** to classify Fashion-MNIST images into 10 classes.  
- Code should be flexible to **easily change** the number of hidden layers and neurons.

**File**: `Question_2.py`  
**Key Steps**:
1. Define a neural network using **NumPy** (no TensorFlow/PyTorch training).  
2. Output **probability distribution** over 10 classes (use softmax).  
3. Test the model and report accuracy.

---

### Question 3
**Goal**:  
- Implement **backpropagation** with support for multiple **optimizers**:  
  - SGD  
  - Momentum  
  - Nesterov Accelerated Gradient  
  - RMSProp  
  - Adam  
  - Nadam  
- Code should be flexible for different batch sizes.

**File**: `Question_3.py`  
**Key Steps**:
1. Build a backpropagation framework from scratch.  
2. Provide separate update rules for each optimizer.  
3. Evaluate ease of adding new optimizers.

---

### Question 4
**Goal**:  
- Use **WandB sweeps** to find optimal hyperparameters for your neural network.  
- Hyperparameters to sweep over include epochs, hidden layers, hidden size, L2 regularization, learning rate, optimizer type, batch size, weight initialization, activation function, etc.  
- Keep 10% of the training data aside as validation data.

**File**: `Question_4.py`  
**Key Steps**:
1. Set up a **wandb.sweep** configuration.  
2. Train the model for each hyperparameter combination.  
3. Log metrics (loss, accuracy) and produce plots (e.g., parallel coordinates).

---

### Question 5
**Goal**:  
- Observe the **best accuracy on the validation set** across all models.  
- Use wandb’s **accuracy vs. creation time** or summary plot to visualize top-performing runs.

**File**: `Question_5.py`  
**Key Steps**:
1. Aggregate all runs or sweeps.  
2. Display or log a **time-based** or summary plot of validation/test accuracy.  
3. Identify best runs.

---

### Question 6
**Goal**:  
- **Make inferences** about which hyperparameter configurations worked and which did not.  
- Reference the wandb **parallel coordinates plot** and correlation summary to glean insights.

**File**: `Question_6.py`  
**Key Steps**:
1. Summarize interesting observations.  
2. Discuss the effect of hidden layers, batch size, optimizer, learning rate, etc.  
3. Possibly create bullet points or short paragraphs on insights.

---

### Question 7
**Goal**:  
- For the **best model** identified in the previous questions, report test accuracy and produce a **confusion matrix**.  
- More marks for creativity in plotting (e.g., color-coded, annotated, etc.).

**File**: `Question_7.py`  
**Key Steps**:
1. Load or train the best model.  
2. Evaluate on the test set.  
3. Plot and log a confusion matrix with integer counts or normalized values.

---

### Question 8
**Goal**:  
- Compare **cross-entropy** loss with **squared error** loss for classification tasks.  
- Provide plots or metrics to illustrate which one is better and why.

**File**: `Question_8.py`  
**Key Steps**:
1. Train with cross-entropy vs. squared error.  
2. Plot or compare training curves and final accuracy.  
3. Conclude which loss is better for classification.

---

## Prerequisites and Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Shreyas-Rathod/Deep-Learning-assignments.git
   cd Deep-Learning-assignments
   ```
2. **Install dependencies** (preferably in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
   - **Key libraries**: `numpy`, `matplotlib`, `seaborn`, `wandb`, `scikit-learn`, `keras` (for dataset), etc.
3. **(Optional) wandb Login**:
   ```bash
   wandb login
   ```
   If you want to track runs with Weights & Biases.

---

## Running Each Question

Below are the general commands to run each question’s code:

- **Question 1**:  
  ```bash
  python Question_1.py
  ```
  Displays one sample image per class from Fashion-MNIST.

- **Question 2**:  
  ```bash
  python Question_2.py
  ```
  Trains a feedforward neural network (flexible hidden layers).

- **Question 3**:  
  ```bash
  python Question_3.py
  ```
  Demonstrates backprop with multiple optimizers (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam).

- **Question 4**:  
  ```bash
  python Question_4.py
  ```
  Runs wandb sweeps for hyperparameter tuning. Logs results and produces parallel coordinates, correlation summaries, etc.

- **Question 5**:  
  ```bash
  python Question_5.py
  ```
  Shows the best validation accuracy across all runs.

- **Question 6**:  
  ```bash
  python Question_6.py
  ```
  Provides insights and observations about which hyperparameters matter the most.

- **Question 7**:  
  ```bash
  python Question_7.py
  ```
  Loads (or trains) the best model and prints test accuracy, plus logs a confusion matrix.

- **Question 8**:  
  ```bash
  python Question_8.py
  ```
  Compares cross-entropy vs. squared error loss, plotting and discussing differences.

---

## WandB Integration

- **Sweep Configuration**:  
  In `Question_4.py`, you’ll find a `sweep_config` dictionary. Adjust the hyperparameters and search strategy as needed (`grid`, `random`, or `bayes`).  
- **Logging**:  
  Each script logs metrics (loss, accuracy, etc.) to wandb. The final test accuracy and confusion matrix (Question 7) are also logged.  
- **Viewing Results**:  
  Go to your [wandb dashboard](https://wandb.ai/) and select the project to see runs, parallel coordinate plots, correlation heatmaps, and so forth.

---

## Key Observations and Insights

- **Cross-Entropy vs. Squared Error** (Q8):  
  - Cross-entropy typically aligns better with classification goals and leads to higher accuracy.  
  - Squared error may converge more slowly or get stuck in plateaus.

- **Hyperparameter Tuning (Q4–Q6)**:  
  - The number of hidden layers and their size significantly affects accuracy.  
  - Learning rate and optimizer choice can make or break convergence.  
  - L2 regularization (weight decay) can help reduce overfitting, but too large a value can harm performance.

- **Optimizers (Q3)**:  
  - Adam and RMSProp often converge faster, whereas vanilla SGD may require careful tuning.  
  - Momentum or Nesterov can speed up training in certain scenarios.

- **Best Model & Confusion Matrix (Q7)**:  
  - The confusion matrix reveals which classes are most often misclassified.  
  - Visualizing it can highlight classes that look visually similar (e.g., “Shirt” vs. “T-shirt/top”).

---

## Repository Structure

```
.
├── Question_1.py       # Q1: Plot 1 sample image per class
├── Question_2.py       # Q2: Feedforward network for classification
├── Question_3.py       # Q3: Backprop with multiple optimizers
├── Question_4.py       # Q4: WandB sweeps for hyperparameter tuning
├── Question_5.py       # Q5: Best validation accuracy summary
├── Question_6.py       # Q6: Observations and insights (parallel coords)
├── Question_7.py       # Q7: Best model + confusion matrix
├── Question_8.py       # Q8: Cross-entropy vs. squared error comparison
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## License

This project is published under an open-source license. See the [LICENSE](LICENSE) file for details, or include any licensing details your institution or course requires.

---

**Contact**:  
For any questions or issues, please open a GitHub issue or contact [Shreyas Rathod](https://github.com/Shreyas-Rathod). Feel free to fork and modify this repository for your own experiments!
