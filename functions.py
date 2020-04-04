import sys
import csv
from random import randint
import collections
import numpy as np
import matplotlib.pyplot as plt
#from PIL import Image

def extractColumnsFromMatrix(mylist, startingPos, length):
    ret = []
    count = 0
    for i in range(0,len(mylist)):
        ret.append([])
        for j in range(startingPos, startingPos+length):
            ret[i].append(mylist[i][j])

    return ret

#def processImage(filename) :
#    im = Image.open(filename)
#    size = 40, 40
#    im_resized = im.resize(size)
#    pix2DSub = np.asarray(im_resized)
#    pixFlatSample = []
#    for i in range(0, 40):
#        for j in range(0, 40):
#            pixFlatSample.append(pix2DSub[i, j])
#    return pixFlatSample

def standardizeSingleList(mylist):
    array = np.array(mylist)
    mean = mylist.mean()
    stdev = mylist.std()
    for i in range(0, len(mylist)):
        mylist[i] = (mylist[i] - mean)/stdev

    return mylist

def standardize(mylist):
    list2D = []
    listOfMeans = []
    listOfStandardDevs = []
    list2DMapTransposed = map(list, zip(*mylist))
    list2DTransposed = []

    for i in list2DMapTransposed:
        list2DTransposed.append(i)
        a = np.array(i)
        listOfMeans.append(a.mean())
        listOfStandardDevs.append(a.std())

    for i in range(0, len(list2DTransposed)):
        for j in range(0, len(list2DTransposed[0])):
            list2DTransposed[i][j] = ((list2DTransposed[i][j] - listOfMeans[i]) / listOfStandardDevs[i])

    tmpMap = map(list, zip(*list2DTransposed))

    finalList = []

    for i in tmpMap:
        finalList.append(i)

    return finalList

def standardizeExceptLastColumn(mylist):
    listOfMeans = []
    listOfStandardDevs = []
    list2DMapTransposed = map(list, zip(*mylist))
    list2DTransposed = []

    for i in list2DMapTransposed:
        list2DTransposed.append(i)
        a = np.array(i)
        listOfMeans.append(a.mean())
        listOfStandardDevs.append(a.std())

    for i in range(0, len(list2DTransposed)-1):
        for j in range(0, len(list2DTransposed[0])):
            if (listOfStandardDevs[i] == 0):
                print("HOLDUP!")
            list2DTransposed[i][j] = ((list2DTransposed[i][j] - listOfMeans[i]) / listOfStandardDevs[i])

    tmpMap = map(list, zip(*list2DTransposed))

    finalList = []

    for i in tmpMap:
        finalList.append(i)

    return finalList

def standardizeExceptLastColumnUsingAnotherList(mylistT, mylist):
    listOfMeans = []
    listOfStandardDevs = []
    list2DMapTransposed = map(list, zip(*mylist))


    list2DTransposed = []

    for i in list2DMapTransposed:
        a = np.array(i)
        listOfMeans.append(a.mean())
        listOfStandardDevs.append(float(a.std()))

    list2DMapTransposed2 = map(list, zip(*mylistT))
    for i in list2DMapTransposed2:
        list2DTransposed.append(i)

    for i in range(0, len(list2DTransposed)-1):
        for j in range(0, len(list2DTransposed[0])):
            if (listOfStandardDevs[i] == 0):
                print("HOLDUP!")
                print(listOfStandardDevs[i])
            list2DTransposed[i][j] = ((list2DTransposed[i][j] - listOfMeans[i]) / listOfStandardDevs[i])

    tmpMap = map(list, zip(*list2DTransposed))

    finalList = []

    for i in tmpMap:
        finalList.append(i)

    return finalList

def addOnesColumn(myList):
    tmpList = []
    for i in range(0, len(myList)):
        tmpList.append([])
        tmpList[i].append(1)
        for j in range(0, len(myList[i])):
            tmpList[i].append(myList[i][j])

    return tmpList

def mode(myList):
    uniqueElements = list(set(myList))
    counters = []
    for i in uniqueElements:
        counters.append(0)

    arrayL = np.array(myList)
    for i in np.nditer(arrayL):
        ++counters[int(i)-1]

    max = 0
    for i in range(1,len(counters)):
        if (counters[i] == counters[max] or counters[i] > counters[max]):
            max = i

    return (max+1)



def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

def pca(listOfData, k):
    ###Covariance matrix
    covarianceMatrix = np.cov(np.array(listOfData).T)

    ###Get eigen vectors and eigen values, v is the eigen vector w is the
	###eigen values###
    w, v = np.linalg.eigh(covarianceMatrix)

    ###Temporary dictionary that will be used to sort the eigen vectors
    myDict = {}

    ###Transposed the eigen values
    vTransposed = map(list, zip(*v))

    ###Sort the eigenvectors based on the eigen values
    count = 0
    for i in vTransposed:
        myDict[w[count]] = i
        count = count + 1

    # myDict = sorted(myDict.items())

    myDict = collections.OrderedDict(sorted(myDict.items()))

    ##Pick the top eigenvectors
    listOfValues = [v for v in myDict.values()]
    k = 2

    w = []

    for i in range(len(listOfValues) - 1, len(listOfValues) - (k + 1), -1):
        w.append(listOfValues[i])

    tmpList = map(list, zip(*w))

    eigenVectorsTransposdedK = []

    for i in tmpList:
        eigenVectorsTransposdedK.append(i)

    Z = np.dot(listOfData, eigenVectorsTransposdedK)
    return Z

def kmeans(data, k, xcol,ycol):
	#Color codes
    colorCodesForProblem1 = ["yx", "mx", "cx", "rx", "gx", "bx", "kx"]

	#standardized data
    dataStandardized = standardize(data)

	#data
    columnsXandY = dataStandardized

    numberOfFeatures = len(data[0])
    numberOfClusters = k

	#List of random data points
    randomSelections = []

	#Get random indexes make sure each index
	#is unique
    for i in range(0, numberOfClusters):
        loop = 0
        while loop == 0:
            index = randint(0, len(columnsXandY))
            if (len(randomSelections) == 0):
                loop = 1
                randomSelections.append(index)
            elif index not in randomSelections:
                loop = 1
                randomSelections.append(index)

	#Select the random data points based on
	#the random indices to be our cluster
	#centers
    clusterPoints = []
    for i in range(0, len(randomSelections)):
        index = randomSelections[i]
        clusterPoints.append([])
        for j in range(0, numberOfFeatures):
            clusterPoints[i].append(columnsXandY[index][j])

	#Perform kmeans
    iter = 0
    run = 0
    while run == 0:

		#Get euclidean distances to the cluster points
        euclideanDistances = []

        tempDistances = []
        for i in range(0, len(columnsXandY)):
            euclideanDistances.append([])
            for j in range(0, numberOfClusters):
                deltaSum = 0
                for h in range(0, numberOfFeatures):
                    delta = columnsXandY[i][h] - clusterPoints[j][h]
                    deltaSquared = delta ** 2
                    deltaSum = deltaSum + deltaSquared
                euclideanDistances[i].append(np.sqrt(deltaSum))

		#Assign points to cluster based on proximity
        listOfMarkers = []
        for i in range(0, len(euclideanDistances)):
            index = euclideanDistances[i].index(min(euclideanDistances[i]))
            listOfMarkers.append(colorCodesForProblem1[index])

		#Create a list for plotting clusters
        plotClusters = []
        for i in range(0, len(colorCodesForProblem1)):
            plotClusters.append([])
            for j in range(0, numberOfFeatures):
                plotClusters[i].append([])
                plotClusters[i].append([])

		#Assign each data point to their respected cluster
		#to be plotted
        for i in range(0, len(colorCodesForProblem1)):
            for j in range(0, len(listOfMarkers)):
                if (colorCodesForProblem1[i] == listOfMarkers[j]):
                    for h in range(0, numberOfFeatures):
                        plotClusters[i][h].append(columnsXandY[j][h])

		#Plot the initial seed and 1st iteration
        if (iter == 0):
            for i in range(0, len(plotClusters)):
                plt.plot(plotClusters[i][0], plotClusters[i][1], 'rx')

            for i in range(0, len(clusterPoints)):
                plt.plot(clusterPoints[i][xcol - 1], clusterPoints[i][ycol - 1], 'yo')
            plt.title('Initial Seeds')
            plt.show()

            for i in range(0, len(plotClusters)):
                plt.plot(plotClusters[i][0], plotClusters[i][1], colorCodesForProblem1[i])

            for i in range(0, len(clusterPoints)):
                plt.plot(clusterPoints[i][xcol - 1], clusterPoints[i][ycol - 1], 'yo')
            plt.title('1 Iteration')
            plt.show()

		#Get the new center of each cluster
        averageXsandYs = []
        tempDistances = clusterPoints
        clusterPoints = []
        for i in range(0, numberOfClusters):
            clusterPoints.append([])
            for j in range(0, numberOfFeatures):
                if (len(plotClusters[i][j]) != 0):
                    clusterPoints[i].append(sum(plotClusters[i][j]) / float(len(plotClusters[i][j])))
                else:
                    clusterPoints[i].append(0)

		#Calculate the magnitude of change of the clusters
        deltaSum = 0
        for i in range(0, numberOfClusters):
            for j in range(0, numberOfFeatures):
                delta = tempDistances[i][j] - clusterPoints[i][j]
                deltaSquared = delta ** 2
                deltaSum = deltaSum + deltaSquared

        euclideanDistance = np.sqrt(deltaSum)

		#Check to see if the magnitude of change is less than
		#epsilon if so leave if not keep going
        if (euclideanDistance < (2 ** -23)):
            run = 1
        else:
            iter = iter + 1

	#Make final plot
    for i in range(0, len(plotClusters)):
        plt.plot(plotClusters[i][0], plotClusters[i][1], colorCodesForProblem1[i])

    for i in range(0, len(clusterPoints)):
        plt.plot(clusterPoints[i][xcol - 1], clusterPoints[i][ycol - 1], 'yo')

    s = str(iter) + ' iterations'
    plt.title(s)
    plt.show()

def binaryMode(myList):
    count0 = 0
    count1 = 0
    for i in myList:
        if (i == 0):
            count0 = count0 + 1
        else:
            count1 = count1 + 1

    if (count0 > count1):
        return 0
    else:
        return 1