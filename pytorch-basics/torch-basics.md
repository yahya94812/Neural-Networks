# PyTorch Basics: Complete Beginner's Guide

## Table of Contents
1. [Installation](#installation)
2. [Tensors - The Foundation](#tensors)
3. [Tensor Operations](#tensor-operations)
4. [Autograd - Automatic Differentiation](#autograd)
5. [Building Neural Networks](#building-neural-networks)
6. [Training a Model](#training-a-model)
7. [GPU Acceleration](#gpu-acceleration)
8. [Saving and Loading Models](#saving-and-loading-models)

---

## Installation

First, install PyTorch. Visit [pytorch.org](https://pytorch.org) to get the installation command for your system, or use:

```bash
# CPU version
pip install torch torchvision

# GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 1. Tensors - The Foundation

Tensors are multi-dimensional arrays, similar to NumPy arrays but with GPU support and automatic differentiation.

### Creating Tensors

```python
import torch

# From Python lists
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])

# With specific shapes and values
zeros = torch.zeros(3, 3)      # 3x3 matrix of zeros
ones = torch.ones(2, 4)        # 2x4 matrix of ones
random = torch.randn(3, 3)     #  Random values from a standard normal distribution
"""
Mean (μ) = 0
Standard deviation (σ) = 1
Range: theoretically unbounded (−∞ to +∞), but in practice most values fall roughly within [-3, 3] due to the properties of the normal distribution.
"""
uniform = torch.rand(2, 2)     # Random values from uniform distribution [0, 1)
torch.randn(3, 3) * 2 + 5 # normal distribution with mean=5, std=2

# From NumPy arrays
import numpy as np
numpy_array = np.array([1, 2, 3])
tensor_from_numpy = torch.from_numpy(numpy_array)

# Create tensor with specific range
arange_tensor = torch.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
linspace_tensor = torch.linspace(0, 1, 5)  # 5 values from 0 to 1
```

### Tensor Properties

```python
tensor = torch.randn(3, 4, 5)

print(tensor.shape)      # torch.Size([3, 4, 5])
print(tensor.size())     # Same as shape
print(tensor.dtype)      # Data type (float32, int64, etc.)
print(tensor.device)     # Device (cpu or cuda)
print(tensor.ndim)       # Number of dimensions (3)
print(tensor.numel())    # Total number of elements (60)
```

---

## 2. Tensor Operations

### Basic Arithmetic

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Element-wise operations
addition = a + b           # [5, 7, 9]
subtraction = a - b        # [-3, -3, -3]
multiplication = a * b     # [4, 10, 18]
division = a / b           # [0.25, 0.4, 0.5]

# Scalar operations
scaled = a * 2             # [2, 4, 6]
```

### Matrix Operations

```python
# Dot product
dot_product = torch.dot(a, b)  # Scalar: 32.0

# Matrix multiplication
matrix_a = torch.randn(3, 4)
matrix_b = torch.randn(4, 2)
result = torch.matmul(matrix_a, matrix_b)  # Shape: (3, 2)
# Alternative: result = matrix_a @ matrix_b

# Transpose
transposed = matrix_a.T  # or matrix_a.transpose(0, 1)
```

### Reshaping and Indexing

```python
# Reshape
original = torch.arange(12)  # [0, 1, 2, ..., 11]
reshaped = original.view(3, 4)  # 3x4 matrix
# view() requires contiguous memory
# reshape() works even if not contiguous

# Indexing
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(tensor[0])        # First row: [1, 2, 3]
print(tensor[:, 1])     # Second column: [2, 5, 8]
print(tensor[1, 2])     # Element at row 1, col 2: 6

# Slicing
print(tensor[0:2, 1:3]) # First 2 rows, columns 1-2

# Boolean indexing
mask = tensor > 5
print(tensor[mask])     # [6, 7, 8, 9]
```

### Useful Operations

```python
# Statistical operations
tensor = torch.randn(3, 4)
mean = tensor.mean()
std = tensor.std()
sum_all = tensor.sum()
sum_dim0 = tensor.sum(dim=0)  # Sum along dimension 0

# Min/Max
max_val = tensor.max()
max_indices = tensor.argmax()  # Index of max value
min_val, min_idx = tensor.min(dim=1)  # Min along dim 1

# Concatenation
t1 = torch.ones(2, 3)
t2 = torch.zeros(2, 3)
concat = torch.cat([t1, t2], dim=0)  # Stack vertically (4, 3)
stack = torch.stack([t1, t2], dim=0)  # Add new dimension (2, 2, 3)
```

---

## 3. Autograd - Automatic Differentiation

PyTorch's autograd automatically computes gradients, which is essential for training neural networks.

### Basic Gradient Computation

```python
# Enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Perform operations
z = x**2 + y**3

# Compute gradients
z.backward()

print(f"dz/dx = {x.grad}")  # 2x = 4.0
print(f"dz/dy = {y.grad}")  # 3y² = 27.0
```

### Multiple Variables

```python
# More complex example
x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(4.0, requires_grad=True)

# z = 2x² + 3xy + y²
z = 2 * x**2 + 3 * x * y + y**2

z.backward()

print(f"dz/dx = {x.grad}")  # 4x + 3y = 24.0
print(f"dz/dy = {y.grad}")  # 3x + 2y = 17.0
```

### Gradient Management

```python
# Clear gradients (important in training loops!)
x.grad.zero_()

# Disable gradient tracking (useful for inference)
with torch.no_grad():
    y = x * 2  # No gradient computed

# Detach from computation graph
y = x.detach()  # Creates new tensor without gradient
```

---

## 4. Building Neural Networks

Use `torch.nn` module to build neural networks.

### Simple Network

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(10, 20)  # Fully connected: 10 inputs, 20 outputs
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
        self.fc3 = nn.Linear(5, 1)
    
    def forward(self, x):
        # Define how data flows through the network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Create model instance
model = SimpleNet()
print(model)

# Test with random input
input_data = torch.randn(1, 10)  # Batch size 1, 10 features
output = model(input_data)
print(f"Output: {output}")
```

### Common Layers

```python
# Fully connected (Linear)
fc = nn.Linear(in_features=100, out_features=50)

# Convolutional (for images)
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)

# Recurrent (for sequences)
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2)
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)

# Activation functions
relu = nn.ReLU()
sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softmax = nn.Softmax(dim=1)

# Normalization
batch_norm = nn.BatchNorm1d(num_features=50)
dropout = nn.Dropout(p=0.5)
```

### Sequential Models

```python
# Alternative way to build models
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)

# Use it
output = model(torch.randn(1, 10))
```

---

## 5. Training a Model

Complete training workflow for a simple regression problem.

### Full Example: Linear Regression

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Create synthetic data: y = 3x + 2 + noise
torch.manual_seed(42)
X_train = torch.randn(100, 1) * 10
y_train = 3 * X_train + 2 + torch.randn(100, 1) * 2

# 2. Define model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)

model = LinearModel()

# 3. Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass: compute predictions
    predictions = model(X_train)
    
    # Compute loss
    loss = criterion(predictions, y_train)
    
    # Backward pass: compute gradients
    optimizer.zero_grad()  # Clear old gradients
    loss.backward()        # Compute new gradients
    
    # Update parameters
    optimizer.step()
    
    # Print progress
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Check learned parameters
print(f"Learned weight: {model.linear.weight.item():.4f} (should be ~3)")
print(f"Learned bias: {model.linear.bias.item():.4f} (should be ~2)")
```

### Classification Example

```python
# Binary classification with sigmoid
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = BinaryClassifier()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop is similar to above
```

### Common Optimizers

```python
# Stochastic Gradient Descent
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (usually a good default choice)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# RMSprop
optimizer = optim.RMSprop(model.parameters(), lr=0.01)

# Learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

### Common Loss Functions

```python
# Regression
mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()

# Binary Classification
bce_loss = nn.BCELoss()  # Requires sigmoid output
bce_with_logits = nn.BCEWithLogitsLoss()  # Includes sigmoid

# Multi-class Classification
cross_entropy = nn.CrossEntropyLoss()  # Includes softmax
```

---

## 6. GPU Acceleration

Move computations to GPU for faster training.

```python
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Move tensors to GPU
cpu_tensor = torch.randn(100, 100)
gpu_tensor = cpu_tensor.to(device)

# Move model to GPU
model = SimpleNet().to(device)

# In training loop, move data to GPU
for epoch in range(num_epochs):
    # Move input and target to GPU
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    # Rest of training code...
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    # ...

# Move back to CPU if needed
cpu_result = gpu_tensor.cpu()
```

---

## 7. Saving and Loading Models

### Save Model Weights

```python
# Save only the model parameters (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load weights
model = SimpleNet()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()  # Set to evaluation mode

# Save entire model (not recommended - less flexible)
torch.save(model, 'entire_model.pth')
loaded_model = torch.load('entire_model.pth')
```

### Save Training Checkpoint

```python
# Save complete training state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Resume training
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## Quick Reference: Essential Patterns

### Training Loop Template

```python
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    
    for batch_inputs, batch_targets in train_loader:
        # Move to device
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)
            
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
```

### Data Loading

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create dataset and loader
dataset = CustomDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Use in training
for inputs, labels in dataloader:
    # Training code here
    pass
```

---

## Next Steps

1. **Practice with real datasets**: Try MNIST, CIFAR-10, or your own data
2. **Explore torchvision**: Pre-built models and image transformations
3. **Learn about CNNs**: For computer vision tasks
4. **Learn about RNNs/LSTMs**: For sequence data
5. **Try transfer learning**: Use pre-trained models
6. **Read the docs**: [pytorch.org/docs](https://pytorch.org/docs)

Happy learning!