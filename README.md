I've been working on building my own Convolutional Neural Network (CNN) from scratch based on [this presentation](https://math.mit.edu/research/highschool/primes/materials/2024/December/2-5-Egan.pdf) I gave for MIT PRIMES CS Circle! Right now, the underlying math works and I'm working on training it to recognize shapes.

## The package

Basically, a CNN is a neural network that's very useful for image processing because it's fundamentally based on 2-3D arrays. Images have more than one dimension, so convolutions are more efficient for processing them because they maintain the multidimensional properties that an image inherently has.

Most of the code in the package is essentially a ton of math (more specifically, some basic multivariable calculus and linear algebra). The challege is less knowing how to write the code but more knowing what to write, given it's a bit complicated.

The layers my package has are:
- dense (or fully connected, the kind you'd find in a standard neural network)
- convolution (perfoms a convolution on the input with a kernel - useful for images, because the input is a 2-3D array)
- pooling (makes the inputs to convolutional layers smaller)
- flatten (in between convolutional and dense to convert the array to a list)

All of these have code to make the forward and backward (training) passes work.

You can also save the model you train to a file and load it later!

## Training

I've also been working on training a model to recognize shapes. I wrote some code using Pillow to generate images of circles, squares, and triangles to feed into the model and train it.

Once it's fully trained, I'll use the drawing app I've written using Tkinter to test it on manually drawn images.
