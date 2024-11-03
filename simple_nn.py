# NOTE 0: REFERENCE -> GFG
# PERFORMING BINARY CLASSIFICATION USING SINGLE NEURON
import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

class NeuralNetwork:
    def __init__(self):
        # Using seed to make sure it'll  
        # generate same weights in every run
        np.random.seed(1)
        # expects to be 3 inputs and single output
        # self.weight_matrix = 2*np.random.random((3,1))-1 (FOR BINARY CLASSIFICATION)
        self.weight_matrix = 2*np.random.random((2,1))-1
        # bias
        self.bias = 2*np.random.random((1,)) - 1
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derv(self, x):
        return 1.0-np.tanh(x)**2
    
    def forward_prop(self, inputs):
        return self.tanh(np.dot(inputs, self.weight_matrix)+ self.bias)
    
    def train(self, train_inps, train_ops, lr, epochs):
        for epoch in range(epochs):
            # GET DOT PRODUCT (Forward Pass)
            output = self.forward_prop(inputs=train_inps)

            error = train_ops-output # LOSS
            # multiply the error by input and then 
            # by gradient of tanh function to calculate
            # the adjustment needs to be made in weights
            # adjustments by back propogation
            update = error*lr*self.tanh_derv(output)
            adj = np.dot(train_inps.T, update)
            bias_adj = np.sum(update)

            # Update weights and bias (Backward Pass)
            self.weight_matrix += adj
            self.bias += bias_adj

def visualize(X, y, color,title, xlabel, ylabel, label):
    plt.figure(figsize=(10, 8))
    plt.scatter(X, y, color=color, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def IrisClassification():
    iris = datasets.load_iris()
    X = iris.data[:100, :2]
    y = iris.target[:100]
    # convert labels to binary
    lb = LabelBinarizer()
    y = lb.fit_transform(y).flatten()
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # convert y column to vector
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # Initialize and train
    nn = NeuralNetwork()
    print("Training", end="")
    for i in range(3): print(".",end="");time.sleep(1) #JUST FOR FUN
    print()
    nn.train(train_inps=X_train, train_ops=y_train, lr=0.1, epochs=10000)
    print("Testing on new Examples: ")
    preds = nn.forward_prop(X_test)
    # convert output to binary classification
    preds = (preds>0).astype(int)
    # Output the results
    for i, (pred, actual) in enumerate(zip(preds, y_test)):
        print(f"Example {i+1} -> Predicted: {pred[0]}, Actual: {actual[0]}")
    
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    accuracy = accuracy_score(y_test, preds)
    print("Accuracy Score: ", accuracy)
    visualize(X_test[:,0], y_test,'red', "Testing Data X[0] to Y", "X_test", "y_test", "X[0] vs Y")
    visualize(X_test[:,1], y_test,'red', "Testing Data X[1] to Y", "X_test", "y_test", "X[1] vs Y")
    visualize(y_test, preds,'blue', "Actual vs Predicted", "y_test", "y_pred", f"Accuracy: {accuracy}")


def BinaryClassification():
    nn = NeuralNetwork()
    print("Initial Weights and Bias: ")
    print("Weights: ",nn.weight_matrix)
    print("Bias: ", nn.bias)
    # training
    train_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]) # (4, 3) input size => (n, m)
    train_outputs = np.array([[0], [1], [1], [0]]) # (4, 1) output size => (n, 1)

    nn.train(train_inps=train_inputs, train_ops=train_outputs, lr=0.1, epochs=10000)
    print("Training",end="")
    for i in range(3): print(".",end="");time.sleep(1) #JUST FOR FUN
    print()
    print("Weights and Bias after Training:")
    print("Weights:", nn.weight_matrix)
    print("Bias:", nn.bias)

    # TESTING
    print("TESTING network on new examples ->")
    print(nn.forward_prop(np.array([1 ,0, 0])))

if __name__=="__main__":
    # BinaryClassification()
    IrisClassification()

    
# NOTE 1:
'''
EXPLANATION OF TRANSPOSING TRAIN_INPS AT ADJ
In the line adj = np.dot(train_inps.T, error * lr * self.tanh_derv(output)), the transpose of train_inps is used to align dimensions. Here’s why:

Shape Alignment:

train_inps has a shape of (4, 3), and error * lr * self.tanh_derv(output) results in a (4, 1) array.
To calculate the weight adjustment, we need the resulting matrix to have the same shape as self.weight_matrix ((3,1)), meaning train_inps.T should have shape (3,4).
Dot Product for Weight Update:

The transpose aligns train_inps for the dot product, effectively summing up the contributions of each input feature across all samples. This way, we get the correct adjustment shape for updating weights.
'''
# NOTE 2:
'''
The bias allows the neural network to have more flexibility in fitting the data by effectively shifting the activation function’s threshold. This can lead to better performance, especially when the network needs to learn patterns that aren’t centered around the origin.
'''
# C:/Code Playground/hari.mL/Data/insurance.csv