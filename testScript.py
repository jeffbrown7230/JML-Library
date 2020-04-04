import sys
import math
import random
from decimal import *
import numpy as np
from random import randint
import csv
import time
from JNN import ANN
import matplotlib.pyplot as plt
from functions import standardize, standardizeSingleList, extractColumnsFromMatrix, standardizeExceptLastColumn, binaryMode, standardizeExceptLastColumnUsingAnotherList, addOnesColumn

#Read the csv
f = open("CTG.csv")
reader = csv.reader(f)
count = 0

t0 = time.time()

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


ann = ANN(20, 3, .5, len(trainingData[0]))

ann.train(trainingData, Y, 1000)

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