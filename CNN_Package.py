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
            self.biases = np.zeros(num_kernels)

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
            output += self.biases[:, None, None]
            
            return output
        
        def backward(self, output_grad:np.ndarray, learning_rate:float, clip_value:int = 1.0):

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
                    
                kernel_grad = np.clip(kernel_grad, -clip_value, clip_value)

                self.kernels[i] -= learning_rate * kernel_grad
                self.biases[i] -= learning_rate * np.sum(output_grad[i])

            return input_grad


    # check/rewrite this perhaps
    class MaxPooling2D:

        def __init__(self, pool_size:Tuple[int,int]):
            self.pool_size = pool_size

            self.prev_input = None
        
        def forward(self, input_data:np.ndarray) -> np.ndarray:

            self.prev_input = input_data

            output_shape = (
                input_data.shape[0], 
                input_data.shape[1] // self.pool_size[0], 
                input_data.shape[2] // self.pool_size[1]
            )

            output = np.zeros(output_shape)

            for i in range(input_data.shape[0]):
                for j in range(output_shape[1]):
                    for k in range(output_shape[2]):
                        output[i][j][k] = np.max(input_data[i][j*self.pool_size[0]:(j+1)*self.pool_size[0], k*self.pool_size[1]:(k+1)*self.pool_size[1]])
            
            return output
        
        def backward(self, output_grad:np.ndarray, learning_rate:float = None):

            input_grad = np.zeros(self.prev_input.shape)

            for i in range(self.prev_input.shape[0]):
                for j in range(output_grad.shape[1]):
                    for k in range(output_grad.shape[2]):
                        window = self.prev_input[i][j*self.pool_size[0]:(j+1)*self.pool_size[0], k*self.pool_size[1]:(k+1)*self.pool_size[1]]
                        max_val = np.max(window)
                        for l in range(self.pool_size[0]):
                            for m in range(self.pool_size[1]):
                                if window[l][m] == max_val:
                                    input_grad[i][j*self.pool_size[0]+l][k*self.pool_size[1]+m] = output_grad[i][j][k]
            
            return input_grad
    
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
            self.biases = np.zeros(units)
            # print(self.weights)

            self.prev_inputs = None
        
        def forward(self, inputs: np.ndarray) -> np.ndarray:

            self.prev_inputs = inputs

            output = self.weights.dot(inputs) + self.biases

            if self.activation == Activation.RELU:
                return np.maximum(0, output)
            
            elif self.activation == Activation.SOFTMAX:
                print(output)
                exp_output = np.exp(output - np.max(output))
                print("exp_output")
                print(np.sum(exp_output))
                return exp_output / np.sum(exp_output)
            return output
        
        def backward(self, output_grad:np.ndarray, learning_rate:float, clip_value:int = 1.0):

            input_grad = output_grad.dot(self.weights)
            
            # print("prev")
            # print(self.weights)

            output_grad = np.clip(output_grad, -clip_value, clip_value)

            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    self.weights[i][j] -= learning_rate*output_grad[i]*self.prev_inputs[j]
            
            self.biases -= learning_rate * output_grad

            # print("post")
            # print(self.weights)

            return input_grad