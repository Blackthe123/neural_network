import numpy as np
from nnfs.datasets import spiral_data

# Set a random seed for reproducibility
np.random.seed(0)

# Define a dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with a small random value
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases to zero
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        # Calculate the output of the layer
        self.output = np.dot(inputs, self.weights) + self.biases

# Define a ReLU activation class
class Activation_ReLU:
    def forward(self, inputs):
        # Apply the ReLU activation function
        self.output = np.maximum(0, inputs)

# Define a softmax activation class
class Activation_Softmax:
    def forward(self, inputs):
        # Compute the exponential values
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize to get the probabilities
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Store the output
        self.output = probabilities

# Generate spiral data with 100 samples and 3 classes
X, y = spiral_data(samples=100, classes=3)

# Create the first dense layer with 2 inputs and 3 neurons
dense1 = Layer_Dense(2, 3)
# Create the ReLU activation function
activation1 = Activation_ReLU()

# Create the second dense layer with 3 inputs and 3 neurons
dense2 = Layer_Dense(3, 3)
# Create the softmax activation function
activation2 = Activation_Softmax()

# Perform a forward pass through the first dense layer
dense1.forward(X)
# Perform a forward pass through the ReLU activation function
activation1.forward(dense1.output)

# Perform a forward pass through the second dense layer
dense2.forward(activation1.output)
# Perform a forward pass through the softmax activation function
activation2.forward(dense2.output)

# Print the output of the softmax activation for the first 5 samples
print(activation2.output[:5])