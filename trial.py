import sys
import math
import random
from decimal import *
import numpy as np
from random import randint
import csv
import time
from JNN import ANN
from alt_cybernetic_automata import ACA
import matplotlib.pyplot as plt
from functions import standardize, standardizeSingleList, extractColumnsFromMatrix, standardizeExceptLastColumn, binaryMode, standardizeExceptLastColumnUsingAnotherList, addOnesColumn

#Read the csv
f = open("testingDataUCS.csv")
reader = csv.reader(f)
count = 0

t0 = time.time()

csvData = []
for row in reader:
    if (count < 1):
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

trainingData = []

for i in range(0, len(csvData)):
    trainingData.append(randomizedList[i])

#Standardize based on training data
randomizedList = standardizeExceptLastColumnUsingAnotherList(randomizedList, trainingData)

trainingData = []

#Grab training and testing data
for i in range(0, len(csvData)):
    trainingData.append(randomizedList[i])

#Add ones column to training data
trainingData = addOnesColumn(trainingData)

Y = []

for i in range(0, len(trainingData)):
    Y.append(trainingData[i][len(trainingData[i])-1])
    del trainingData[i][len(trainingData[i]) - 1]



ann = ACA(3, 3, .5, 3)

ann.train(trainingData, Y, 2000)


#Check if it's conditioned
print("Test to see if UCS is properly conditioned")
ann.test(trainingData,Y)

#Now grab testing data AKA the CS
#Read the csv
f = open("testingDataCS.csv")
reader = csv.reader(f)
count = 0

t0 = time.time()

csvData = []
for row in reader:
    if (count < 1):
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

testingData = []

for i in range(0, len(csvData)):
    testingData.append(randomizedList[i])

#Standardize based on training data
randomizedList = standardizeExceptLastColumnUsingAnotherList(randomizedList, testingData)

testingData = []

#Grab training and testing data
for i in range(0, len(csvData)):
    testingData.append(randomizedList[i])

#Add ones column to training data
testingData = addOnesColumn(testingData)
test_Y = []
UCS = []
count = 0
for i in Y:
    if (i == 1):
        test_Y.append(1)
        UCS.append(trainingData[count])
    count += 1

#Test how conditioned the ACA is to the UCS
print("Test to see if CS is properly unconditioned")
ann.test(testingData,test_Y)

print("Conditioning")
for k in range(0,15):
    print("Batch: "+str(k))
    #Now condition it
    ann.condition(UCS,testingData,1)

    ann.test(testingData,test_Y)