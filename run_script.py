import sys
import math
import random
from decimal import *
import numpy as np
from random import randint
import sys
import csv
import time
from JNN import ANN
import matplotlib.pyplot as plt
from functions import standardize, standardizeSingleList, extractColumnsFromMatrix, standardizeExceptLastColumn, binaryMode, standardizeExceptLastColumnUsingAnotherList, addOnesColumn

user_input = "Y"

#Read the csv
f = open(sys.argv[1])
reader = csv.reader(f)


hiddenLayerSize = int(sys.argv[2])
outputLayerSize = int(sys.argv[3])
learningParamater = float(sys.argv[4])
numberOfIterations = int(sys.argv[5])
t0 = time.time()

#User count variable to skip the first two lines of csv
#First line contains the column headers
#Second line contains the a blank row
count = 0
csvData = []
for row in reader:
    if (count < 2):
        count = count + 1
    else:
        a = np.array(row).astype(np.float)
        if (len(a) != 0):
            csvData.append(a)
        count = count + 1

random.seed(0)

randomizedList = []

random.shuffle(csvData)

randomizedList = csvData

#Grab the training data
trainingDataAmount = np.ceil(len(randomizedList)*.66)

trainingData = []

for i in range(0, int(trainingDataAmount)):
    trainingData.append(randomizedList[i])

#Standardize based on training data
randomizedList = standardizeExceptLastColumnUsingAnotherList(randomizedList, trainingData)

trainingData = []

#Grab training and testing data
for i in range(0, int(trainingDataAmount)):
    trainingData.append(randomizedList[i])

testingData = []
for i in range(int(trainingDataAmount), len(randomizedList)):
    testingData.append(randomizedList[i])

#Add ones column to training data
trainingData = addOnesColumn(trainingData)

#Add ones column to testingData data
testingData = addOnesColumn(testingData)

#Grab the correct answers from the testing and training data
Y = []
Y1 = []

for i in range(0, len(trainingData)):
    Y.append(trainingData[i][len(trainingData[i])-1])
    del trainingData[i][len(trainingData[i])-1]
    del trainingData[i][len(trainingData[i]) - 1]

for i in range(0, len(testingData)):
    Y1.append(testingData[i][len(testingData[i])-1])
    del testingData[i][len(testingData[i])-1]
    del testingData[i][len(testingData[i]) - 1]


#Instantiate a neural network
ann = ANN("ANN", hiddenLayerSize, outputLayerSize, learningParamater, len(trainingData[0]))

#Train the network over the amount of iterations, Y is the correct answer from the training data
ann.train(trainingData, Y, numberOfIterations)

#Display the results
ans = 0
count = 0
ansCount = 0
print("-----------------")
for i in testingData:
    ans = ann.test(i)

    if (ans == Y1[count]):
        ansCount = ansCount + 1
    count = count + 1

t1 = time.time()
total = t1-t0

print("ACCURACY")
print(ansCount/len(Y1))
print("")
print("TIME")
print(total)
