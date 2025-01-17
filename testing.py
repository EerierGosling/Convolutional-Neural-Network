from CNN_Package import CNN, Layers, Activation
import numpy as np


cnn = CNN(input_shape=(3, 50, 50), output_shape=(2))
cnn.add(Layers.Conv2D(input_shape=(3, 50, 50), num_kernels=32, kernel_size=(3,3), activation=Activation.RELU))
cnn.add(Layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(Layers.Conv2D(input_shape=(32, 25, 25), num_kernels=64, kernel_size=(3,3), activation=Activation.RELU))
cnn.add(Layers.MaxPooling2D(pool_size=(2, 2)))
cnn.add(Layers.Conv2D(input_shape=(64, 12, 12), num_kernels=128, kernel_size=(3,3), activation=Activation.RELU))
cnn.add(Layers.Flatten((128, 12, 12)))
cnn.add(Layers.Dense(input_units=18432, units=256, activation=Activation.RELU))
cnn.add(Layers.Dense(input_units=256, units=2, activation=Activation.SOFTMAX))

# Check weights initialization
for layer in cnn.layers:
    if hasattr(layer, 'weights'):
        print(f"Layer {layer}: Weights mean = {np.mean(layer.weights)}, std = {np.std(layer.weights)}")

    if hasattr(layer, 'kernels'):
        print(f"Layer {layer}: Weights mean = {np.mean(layer.kernels)}, std = {np.std(layer.kernels)}")


all_loss = []

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

    loss = np.sum(np.power(output-truth, 2))

    all_loss.append(loss)

    if len(all_loss) > 20:
        all_loss.pop(0)

    print(output)
    print(output-truth)
    print(loss)
    print(np.mean(all_loss))
    print()

cnn.save("model")

# cnn = CNN.load("model")