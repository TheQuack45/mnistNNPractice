clear; close all; clc

# Load images.
trainImages = loadMNISTImages('train-images.idx3-ubyte')';
trainLabels = loadMNISTLabels('train-labels.idx1-ubyte')';
testImages = loadMNISTImages('t10k-images.idx3-ubyte')';
testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte')';
# Changes "0" digit encoding from 0 to 10.
#disp(size(testLabels));
trainVec = trainLabels(:);
trainVec(trainVec == 0) = 10;
trainLabels = reshape(trainVec, size(trainLabels, 1), size(trainLabels, 2));
testVec = testLabels(:);
testVec(testVec == 0) = 10;
testLabels = reshape(testVec, size(testLabels, 1), size(testLabels, 2));
#trainLabels(trainLabels(1, :) == 0, 1) = 10;
#testLabels(testLabels(1, :) == 0, 1) = 10;

# Set used variables.
inputSize = 784;
labelCount = 10;

theta1 = rand(labelCount, inputSize + 1);
J = cost(testImages, testLabels, theta1);
#fprintf('Cost with random parameters: %f', J);
disp('Cost:');
disp(J);