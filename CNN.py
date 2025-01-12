import numpy as np
from enum import Enum
from typing import Tuple
from scipy.signal import convolve
import pickle
from PIL import Image

class Activation(Enum):
    RELU = "relu"
    SOFTMAX = "softmax"

class Loss(Enum):
    CROSS_ENTROPY = "cross_entropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"

class CNN:

    def __init__(self, input_shape:Tuple[int, int, int], output_shape:int):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.layers = []

        self.prev_output = None

    def add(self, layer):
        self.layers.append(layer)

        return self

    def forward(self, input_data:np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)

        self.prev_output = output

        return output
    
    def backward(self, ground_truth:np.ndarray, learning_rate:float): #loss_function:Loss

        if self.layers[-1].activation == Activation.SOFTMAX:
            loss_function = Loss.CROSS_ENTROPY
        else:
            loss_function = Loss.MEAN_SQUARED_ERROR
        
        if loss_function == Loss.CROSS_ENTROPY:
            output_grad = self.prev_output - ground_truth

            for layer in self.layers[::-1]:
                output_grad = layer.backward(output_grad, learning_rate)

    def save(self, filename:str, path:str = None):
        
        filename += ".pkl"

        if path:
            filename = path + "/" + filename

        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename:str, path:str = None):

        filename += ".pkl"

        if path:
            filename = path + "/" + filename

        with open(filename, 'rb') as file:
            return pickle.load(file)

class Layers:

    class Conv2D:

        def __init__(self, input_shape:Tuple[int,int,int], num_kernels:int, kernel_size:Tuple[int,int], activation:Activation = None, padding:int=-1):
            self.input_shape = input_shape
            self.num_kernels = num_kernels
            self.kernel_size = kernel_size
            self.activation = activation

            self.padding = padding if padding >= 0 else kernel_size[0] // 2

            self.kernels = [np.random.randn(input_shape[0], *kernel_size) / (kernel_size[0] * kernel_size[1]) for _ in range(num_kernels)]
            # self.kernels = [np.arange(input_shape[0]* math.prod(kernel_size), dtype=float).reshape(input_shape[0], *kernel_size) for _ in range(num_kernels)]

            self.prev_input = None
            
        def forward(self, input_data) -> np.ndarray:

            self.prev_input = input_data

            if input_data.shape != self.input_shape:
                raise ValueError(f"Input data shape {input_data.shape} does not match expected shape {self.input_shape}")
            
            output_shape = (
                self.num_kernels, 
                input_data.shape[1] + self.padding*2 - self.kernel_size[0] + 1, 
                input_data.shape[2] + self.padding*2 - self.kernel_size[1] + 1
            )

            output = np.zeros(output_shape)

            for i in range(self.num_kernels):
                for j in range(self.input_shape[0]):
                    output[i] += convolve(np.pad(input_data[j], pad_width=self.padding, mode='constant'), np.flip(self.kernels[i][j], axis=(0,1)), mode='valid')
            if self.activation == Activation.RELU:
                output = np.maximum(0, output)
            
            return output
        
        def backward(self, output_grad:np.ndarray, learning_rate:float):

            input_grad = np.zeros(self.prev_input.shape)
            
            # copilot partially wrote this loop but it seems right - if things dont work check this again
            for i in range(self.num_kernels):
                for j in range(self.input_shape[0]):
                    padded_output_grad = np.pad(output_grad[i], pad_width=self.padding, mode='constant')
                    input_grad[j] += convolve(padded_output_grad, self.kernels[i][j], mode='valid')

            for i in range(self.num_kernels):
                kernel_grad = np.zeros(self.kernels[i].shape)
                for j in range(self.input_shape[0]):
                    kernel_grad[j] = convolve(np.pad(self.prev_input[j], pad_width=self.padding, mode='constant'), np.flip(output_grad[i], axis=(0,1)), mode='valid')
                    # print("post")
                    # print(kernel_grad[j])

                self.kernels[i] -= learning_rate * kernel_grad

            return input_grad


    # class MaxPooling2D:
    
    class Flatten:

        def __init__(self, input_shape:Tuple[int,int,int]):
            self.input_shape = input_shape

        def forward(self, input_data:np.ndarray) -> np.ndarray:
            return input_data.flatten()
        
        def backward(self, output_grad:np.ndarray, learning_rate:float = None):
            return output_grad.reshape(self.input_shape)


    class Dense:

        def __init__(self, input_units:int, units:int, activation:Activation = None):
            self.input_units = input_units
            self.units = units
            self.activation = activation

            # self.weights = np.arange(units*input_units, dtype=float).reshape(units, input_units)

            self.weights = np.random.randn(self.units, self.input_units) / self.units
            # print(self.weights)

            self.prev_inputs = None
        
        def forward(self, inputs: np.ndarray) -> np.ndarray:

            self.prev_inputs = inputs

            output = self.weights.dot(inputs)

            if self.activation == Activation.RELU:
                return np.maximum(0, output)
            
            elif self.activation == Activation.SOFTMAX:
                # print(output)
                exp_output = np.exp(output - np.max(output))
                return exp_output / np.sum(exp_output)
            
            return output
        
        def backward(self, output_grad:np.ndarray, learning_rate:float):

            input_grad = output_grad.dot(self.weights)
            
            # print("prev")
            # print(self.weights)

            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    self.weights[i][j] -= learning_rate*output_grad[i]*self.prev_inputs[j]
            
            # print("post")
            # print(self.weights)

            return input_grad

cnn = CNN(input_shape=(3, 50, 50), output_shape=(2))
cnn.add(Layers.Conv2D(input_shape=(3, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Conv2D(input_shape=(10, 50, 50), num_kernels=10, kernel_size=(5,5), activation=Activation.RELU))
cnn.add(Layers.Flatten((10, 50, 50)))
cnn.add(Layers.Dense(input_units=25000, units=50, activation=Activation.RELU))
cnn.add(Layers.Dense(input_units=50, units=2, activation=Activation.SOFTMAX))


for i in range(300):
    # print(cnn.forward(np.ones((1, 50, 50))))
    # cnn.backward(np.array([0.2, 0.3, 0.5]), 0.01)

    from test_data import make_shape
    import random
    
    truth = [0,1]
    random.shuffle(truth)

    index = 0

    for i in range(len(truth)):
        if truth[i] == 1:
            index = i
            break
    print(truth)
    output = cnn.forward(np.rot90(make_shape(["circle", "square"][index]), k = 1, axes=(0, 2)))
    cnn.backward(truth, 0.001)

    print(output)
    print()

cnn.save("model")

# cnn = CNN.load("model")