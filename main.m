clear; close all; clc

# Load images
trainImages = loadMNISTImages('train-images.idx3-ubyte')';
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte')';
testImages = loadMNISTImages('t10k-images.idx3-ubyte')';
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte')';

# Set used variables
inputSize = 784;
labelCount = 10;

theta1 = rand(labelCount, inputSize + 1);
J = cost(testImages, testLabels, theta1);
#fprintf('Cost with random parameters: %f', J);
disp('Cost:');
disp(J);