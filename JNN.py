import numpy as np
import math
import random
import csv

#Artifical neural network
class ANN(object):

    #number of hidden layer node
    #number of feature nodes
    hiddenLayerSize = 0
    numberOfFeatures = 0

    #initialize the weights, learning parameter,
    #initialize the weights between input & hidden layer
    #initialize the values of the hidden nodes
    #initialize the weights between hidden to output layer
    #initialize the number of outputs
    #initialize the learning parameter
    inputToHiddenWeights = []
    hiddenNodeValues = []
    hiddenToOutputWeights = []
    numberOfOutputs = 0
    learningParameter = 0

    #initialize neural network object
    def __init__(self, hiddenLayerSize, numberOfOutputs, learningParameter, numberOfFeatures):
        self.hiddenLayerSize = hiddenLayerSize
        self.numberOfFeatures = numberOfFeatures
        self.numberOfOutputs = numberOfOutputs
        self.learningParameter = learningParameter
        self.numberOfFeatures = numberOfFeatures

        #instantiate the weights between the input and hidden nodes
        for i in range(0, numberOfFeatures):
            self.inputToHiddenWeights.append([])
            for j in range(0, hiddenLayerSize):
                self.inputToHiddenWeights[i].append(random.uniform(-1, 1))

        #insantiate the weights between the hidden and output nodes
        for i in range(0, numberOfOutputs):
            self.hiddenToOutputWeights.append([])
            for j in range(0, hiddenLayerSize):
                self.hiddenToOutputWeights[i].append(random.uniform(-1, 1))

    #Define the feature layer and the weights between the different layers
    def setNumberOfFeatures(self, numberOfFeatures):
        self.numberOfFeatures = numberOfFeatures

        for i in range(0, numberOfFeatures):
            self.inputToHiddenWeights.append([])
            for j in range(0, self.hiddenLayerSize):
                self.inputToHiddenWeights[i].append(random.uniform(-1, 1))

        for i in range(0, self.numberOfOutputs):
            self.hiddenToOutputWeights.append([])
            for j in range(0, self.hiddenLayerSize):
                self.hiddenToOutputWeights[i].append(random.uniform(-1, 1))

    #Simulate the firing of a neuron (or node)
    def neuronOutput(self, input):
        return float(1 / float((1.0) + (np.exp(-1.0*input))))

    #Backpropagation from result to output layer (helps the output layer learn)
    def batchProcessOutputLayer(self, theta, outputLoss, learningParameter, N, H):
        learningParameterOverNTimesH = np.array((learningParameter / N) * np.array(H).T)
        return np.array(theta).T + np.dot(learningParameterOverNTimesH, np.array(outputLoss))

    def batchProcessOutputLayerMini(self, theta, outputLoss, learningParameter, N, H):
        learningParameterOverNTimesH = np.array((learningParameter / N) * np.array(H).T)
        return np.array(theta).T + np.dot(learningParameterOverNTimesH, np.array(outputLoss))
 
    #Backpropagation from output layer to hidden layer (helps the hidden learn from what the output layer learned)
    def batchProcessHiddenLayer(self, outputLoss, theta, H, Beta, learningParameter, N, X):
        hiddenLoss = (np.dot(np.array(outputLoss), np.array(theta).T)) * (np.array(H) * np.array(1 - np.array(H)))
        return Beta + np.dot(np.array((learningParameter / N) * np.array(X).T), hiddenLoss)

    #Run a test given sample 2D array of features (with the correct answer at the end of each array)
    def test(self, features):
        #Compute the value of hidden nodes from dot multiplying the input values and weights
        self.hiddenNodeValues = np.dot(np.array(features), np.array(self.inputToHiddenWeights))

        #Compute the values of the hidden nodes after you fire them
        for j in range(0, len(self.hiddenNodeValues)):
            self.hiddenNodeValues[j] = self.neuronOutput(self.hiddenNodeValues[j])

        #initialize a random list
        randomList = []
 
        #Compute the value of output nodes from dot multiplying the hidden values and weights
        outputValue = np.dot(self.hiddenNodeValues, np.array(self.hiddenToOutputWeights).T)

        #Compute the values of the output nodes after you fire them
        for j in range(0, len(outputValue)):
            outputValue[j] = self.neuronOutput(outputValue[j])

        #Get the index of the highest value (+1 to start from 1)
        ans = list(outputValue).index(max(outputValue)) + 1
        return ans

    def test_weighted_selection(self, features):
        self.hiddenNodeValues = np.dot(np.array(features), np.array(self.inputToHiddenWeights))

        for j in range(0, len(self.hiddenNodeValues)):
            self.hiddenNodeValues[j] = self.neuronOutput(self.hiddenNodeValues[j])

        randomList = []
        outputValue = np.dot(self.hiddenNodeValues, np.array(self.hiddenToOutputWeights).T)

        for j in range(0, len(outputValue)):
            outputValue[j] = self.neuronOutput(outputValue[j])

        sel = list(outputValue)

        #Choose an answer based on a limit helps with decimal answers (0.74, .01, etc)
        limit = 0
        for i in sel:
            limit += int(round(i,2)*100)

        choice = random.randint(0,int(limit))
        limit = 0
        ans = 0
        for i in sel:
            limit += int(round(i,2)*100)
            if (choice <= limit):
                return (ans+1)
            ans += 1
        return (ans+1)

    def train(self, trainingData, answers, numberOfIterations):

        #Randomly assign values to the weights between the input, hidden, and output layer
        if (self.numberOfFeatures == 0):
            for i in range(0, self.numberOfFeatures):
                self.inputToHiddenWeights.append([])
                for j in range(0, self.hiddenLayerSize):
                    self.inputToHiddenWeights[i].append(random.uniform(-1, 1))

            for i in range(0, self.numberOfOutputs):
                self.hiddenToOutputWeights.append([])
                for j in range(0, self.hiddenLayerSize):
                    self.hiddenToOutputWeights[i].append(random.uniform(-1, 1))

        #Train on the data for a certain amount of iterations
        for i in range(0, numberOfIterations):
 
            #Initialize the arrays for output values
            outputValue = []
  
            #array that contains output layers (it's a 2D array)
            totalOutputs = []
  
            #array that contains hidden layers (it's a 2D array)
            totalHiddenOutputs = []
 
            #array that contains the difference between the actual output and the selected
            outputLossT = []
            count = 0

            #For k in the training data
            for k in trainingData:
                self.hiddenNodeValues = np.dot(np.array(k), np.array(self.inputToHiddenWeights))
                for j in range(0, len(self.hiddenNodeValues)):
                    self.hiddenNodeValues[j] = self.neuronOutput(self.hiddenNodeValues[j])
                totalHiddenOutputs.append(self.hiddenNodeValues)

                #compute the values of the output nodes
                outputValue = np.dot(self.hiddenNodeValues, np.array(self.hiddenToOutputWeights).T)

                #fire the output nodes
                for j in range(0, len(outputValue)):
                    outputValue[j] = self.neuronOutput(outputValue[j])
                totalOutputs.append(outputValue)

            totalOutputsT = []

            #Transpose the total outputs array
            tmp = np.array(totalOutputs).T
            for i in range(0, self.numberOfOutputs):
                tmp = np.array(totalOutputs).T
                totalOutputsT.append(np.array([tmp[i]]).T)

            #Subtract 1 from the value of the correct output layer index to reinforce it
            for i in range(0, len(answers)):
                index = int(answers[i]) - 1
                for j in range(0, self.numberOfOutputs):
                    if (j == index):
                        totalOutputsT[j][i][0] = 1 - totalOutputsT[j][i][0]
                    else:
                        totalOutputsT[j][i][0] = 0 - totalOutputsT[j][i][0]

            #Perform back propagation from input to hidden layer to make the neural network learn then repeat
            for i in range(0, self.numberOfOutputs):
                tmp = [self.hiddenToOutputWeights[i]]
                tmp = self.batchProcessOutputLayer(tmp, totalOutputsT[i], self.learningParameter, len(totalOutputs),
                                              totalHiddenOutputs)
                self.hiddenToOutputWeights[i] = list(list((np.array(tmp).T))[0])
                self.inputToHiddenWeights = self.batchProcessHiddenLayer(totalOutputsT[i], tmp, totalHiddenOutputs,
                                                                self.inputToHiddenWeights, self.learningParameter,
                                                                len(totalOutputs), trainingData)
