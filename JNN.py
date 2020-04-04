import numpy as np
import math
import random
import csv

class ANN(object):

    hiddenLayerSize = 0
    numberOfFeatures = 0
    #initialize the weights, learning parameter,
    inputToHiddenWeights = []
    hiddenNodeValues = []
    hiddenToOutputWeights = []
    numberOfOutputs = 0
    learningParameter = 0

    def __init__(self, hiddenLayerSize, numberOfOutputs, learningParameter, numberOfFeatures):
        self.hiddenLayerSize = hiddenLayerSize
        self.numberOfFeatures = numberOfFeatures
        self.numberOfOutputs = numberOfOutputs
        self.learningParameter = learningParameter
        self.numberOfFeatures = numberOfFeatures

        for i in range(0, numberOfFeatures):
            self.inputToHiddenWeights.append([])
            for j in range(0, hiddenLayerSize):
                self.inputToHiddenWeights[i].append(random.uniform(-1, 1))

        for i in range(0, numberOfOutputs):
            self.hiddenToOutputWeights.append([])
            for j in range(0, hiddenLayerSize):
                self.hiddenToOutputWeights[i].append(random.uniform(-1, 1))

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

    def neuronOutput(self, input):
        return float(1 / float((1.0) + (np.exp(-1.0*input))))

    def batchProcessOutputLayer(self, theta, outputLoss, learningParameter, N, H):
        learningParameterOverNTimesH = np.array((learningParameter / N) * np.array(H).T)
        return np.array(theta).T + np.dot(learningParameterOverNTimesH, np.array(outputLoss))

    def batchProcessOutputLayerMini(self, theta, outputLoss, learningParameter, N, H):
        learningParameterOverNTimesH = np.array((learningParameter / N) * np.array(H).T)
        return np.array(theta).T + np.dot(learningParameterOverNTimesH, np.array(outputLoss))

    def batchProcessHiddenLayer(self, outputLoss, theta, H, Beta, learningParameter, N, X):
        hiddenLoss = (np.dot(np.array(outputLoss), np.array(theta).T)) * (np.array(H) * np.array(1 - np.array(H)))
        return Beta + np.dot(np.array((learningParameter / N) * np.array(X).T), hiddenLoss)

    def test(self, features):
        self.hiddenNodeValues = np.dot(np.array(features), np.array(self.inputToHiddenWeights))

        for j in range(0, len(self.hiddenNodeValues)):
            self.hiddenNodeValues[j] = self.neuronOutput(self.hiddenNodeValues[j])

        randomList = []
        outputValue = np.dot(self.hiddenNodeValues, np.array(self.hiddenToOutputWeights).T)

        for j in range(0, len(outputValue)):
            outputValue[j] = self.neuronOutput(outputValue[j])

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

        if (self.numberOfFeatures == 0):
            for i in range(0, self.numberOfFeatures):
                self.inputToHiddenWeights.append([])
                for j in range(0, self.hiddenLayerSize):
                    self.inputToHiddenWeights[i].append(random.uniform(-1, 1))

            for i in range(0, self.numberOfOutputs):
                self.hiddenToOutputWeights.append([])
                for j in range(0, self.hiddenLayerSize):
                    self.hiddenToOutputWeights[i].append(random.uniform(-1, 1))

        for i in range(0, numberOfIterations):
            outputValue = []
            totalOutputs = []
            totalHiddenOutputs = []
            outputLossT = []
            count = 0

            for k in trainingData:
                self.hiddenNodeValues = np.dot(np.array(k), np.array(self.inputToHiddenWeights))
                for j in range(0, len(self.hiddenNodeValues)):
                    self.hiddenNodeValues[j] = self.neuronOutput(self.hiddenNodeValues[j])
                totalHiddenOutputs.append(self.hiddenNodeValues)

                outputValue = np.dot(self.hiddenNodeValues, np.array(self.hiddenToOutputWeights).T)

                for j in range(0, len(outputValue)):
                    outputValue[j] = self.neuronOutput(outputValue[j])
                totalOutputs.append(outputValue)

            totalOutputsT = []

            tmp = np.array(totalOutputs).T
            for i in range(0, self.numberOfOutputs):
                tmp = np.array(totalOutputs).T
                totalOutputsT.append(np.array([tmp[i]]).T)

            for i in range(0, len(answers)):
                index = int(answers[i]) - 1
                for j in range(0, self.numberOfOutputs):
                    if (j == index):
                        totalOutputsT[j][i][0] = 1 - totalOutputsT[j][i][0]
                    else:
                        totalOutputsT[j][i][0] = 0 - totalOutputsT[j][i][0]

            for i in range(0, self.numberOfOutputs):
                tmp = [self.hiddenToOutputWeights[i]]
                tmp = self.batchProcessOutputLayer(tmp, totalOutputsT[i], self.learningParameter, len(totalOutputs),
                                              totalHiddenOutputs)
                self.hiddenToOutputWeights[i] = list(list((np.array(tmp).T))[0])
                self.inputToHiddenWeights = self.batchProcessHiddenLayer(totalOutputsT[i], tmp, totalHiddenOutputs,
                                                                self.inputToHiddenWeights, self.learningParameter,
                                                                len(totalOutputs), trainingData)