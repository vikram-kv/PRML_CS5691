# module with functions to read different types of data and pre-process them by normalizing or applying
# a length-fixing transformation on them if needed
# PRML A4 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np
import csv
import os

# function to convert series to targetlength by using a rolling average
def rollingWindowTransform(series: np.array, targetlength : int):
    # size of the rolling window to use for taking the average
    wsize = len(series) - targetlength + 1
    supervector = []
    # we take the vectors in the range (i , i + wsize -1), average them and extend the supervector
    # by this average. we move the window by 1 vector towards right in every iteration using the var i
    for i in range(targetlength):
        avg = np.mean(series[i:(i+wsize)], axis=0)
        supervector += avg.tolist()
    supervector = np.array(supervector)
    return supervector



# function for reading synthetic data from its folder
def inputSYN():
    fdr = input('Enter the name of the folder with the synthetic character data: ')
    # code to get train data and their labels
    trainData = []
    trainLabels = []
    with open(fdr+'/train.txt','r') as f:
        lines = csv.reader(f)
        for l in lines:
            trainData.append([float(l[0]), float(l[1])])
            trainLabels.append(l[2])
        trainData = np.array(trainData, dtype=np.float64)

    # code to get dev data and their labels
    devData = []
    devTrueLabels = []
    with open(fdr+'/dev.txt','r') as f:
        lines = csv.reader(f)
        for l in lines:
            devData.append([float(l[0]), float(l[1])])
            devTrueLabels.append(l[2])
        devData = np.array(devData, dtype=np.float64)
    
    # here, there are 2 classes - 1 and 2
    classlist = ['1', '2']

    # return the 5 tuple with train data, train labels, dev data, dev data labels, class names
    return (trainData, trainLabels, devData, devTrueLabels, classlist)



# function for reading image data from its folder
def inputIMG(norm:int = 0):
    fdr = str(input('Enter the name of the folder containing the image data : '))

    # we have 5 classes of data as in the below list
    classlist =['coast', 'forest', 'highway', 'mountain', 'opencountry']

    # code to get train data and their labels
    trainData = []
    trainLabels = []
    for cls in classlist:
        inputfiles = os.listdir(fdr+'/'+cls+'/train/')
        for inputfile in inputfiles:
            with open(fdr+'/'+cls+'/train/'+ inputfile) as f:
                reader = np.genfromtxt(f, delimiter=' ')
                for row in reader:
                    trainData.append(row)
                trainLabels.append(cls)

    # we reshape train data so that each vector in it is of dimension 828
    trainData = np.array(trainData, dtype=np.float64)
    trainData = trainData.reshape(-1,23*36)

    # code to get dev data and their labels
    devData = []
    devTrueLabels = []
    for cls in classlist:
        inputfiles = os.listdir(fdr+'/'+cls+'/dev/')
        for inputfile in inputfiles:
            with open(fdr+'/'+cls+'/dev/'+ inputfile) as f:
                reader = np.genfromtxt(f, delimiter=' ')
                for row in reader:
                    devData.append(row)
                devTrueLabels.append(cls)

    # we reshape dev data so that each vector in it is of dimension 828
    devData = np.array(devData, dtype=np.float64)
    devData = devData.reshape(-1,23*36)

    # code to normalize the data by min-max scaling if norm == 1
    # we normalize by taking the train data and dev data together
    if(norm == 1):
        tsize = len(trainData)
        data = np.append(trainData, devData, axis=0)
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0)) * 10
        trainData = data[0:tsize]
        devData = data[(tsize):]

    # return the 5 tuple with train data, train labels, dev data, dev data labels, class names
    return (trainData, trainLabels, devData, devTrueLabels, classlist)



# code to extract ISD data from a file. this is assumed to be in the proper format per specs.
def extractISDData(file):
    lines = file.readlines()
    l0 = lines[0].strip().split(' ')
    DIM, num_vectors = int(l0[0]), int(l0[1])     # entry 2 of line 1 = number of feature vectors
    # loop to get all the feature vectors
    vectors = []
    for i in range(1,num_vectors+1):
        l = lines[i]
        l = l.strip().split(' ')
        l = np.array([np.double(l[j]) for j in range(0,DIM)],dtype=np.double)
        vectors.append(l)
    return np.array(vectors)                    # return the mfcc data as a numpy 2D array



# function for reading isolated spoken digits data from its folder
def inputISD():
    fdr = input('Enter the name of the folder containing the isolated digits data : ')

    # we have 5 classes of data as in the below list
    classlist = ['1', '4', '6', 'o', 'z']

    # code to extract training data from FOLDER/cname/train/ and build train data with their labels
    trainData = []
    trainLabels = []
    for cname in classlist:
        path = fdr + '/' + cname + '/train/'
        for fname in os.listdir(path):
            if(fname.__contains__('.mfcc')):
                with open(os.path.join(path,fname),'r') as f:
                    trainData.append(extractISDData(f))
                    trainLabels.append(cname)

    # code to extract dev data from FOLDER/cname/dev/ and build dev data with their labels
    devData = []
    devTrueLabels = []
    for cname in classlist:
        path = fdr + '/' + cname + '/dev/'
        for fname in os.listdir(path):
            if(fname.__contains__('.mfcc')):
                with open(os.path.join(path,fname),'r') as f:
                    devData.append(extractISDData(f))
                    devTrueLabels.append(cname)
    
    # we find the minimum length of the time series in both train data and dev data
    # and store the result in minlen
    minlen = np.PINF
    for t in trainData:
        minlen = min(minlen, len(t))
    for d in devData:
        minlen = min(minlen, len(d))

    # code to transform each train vector series to a fixed length supervector of dimension = minlen * 38
    # using the rolling window transform
    transformedTrainData = []
    for t in trainData:
        transformedTrainData.append(rollingWindowTransform(t, minlen))
    transformedTrainData = np.array(transformedTrainData)

    # code to transform each dev vector series to a fixed length supervector of dimension = minlen * 38
    # using the rolling window transform
    transformedDevData = []
    for d in devData:
        transformedDevData.append(rollingWindowTransform(d, minlen))
    transformedDevData = np.array(transformedDevData)

    # return the 5 tuple with train data, train labels, dev data, dev data labels, class names
    return (transformedTrainData, trainLabels, transformedDevData, devTrueLabels, classlist)



# code for normalizing the character defined by points
def normalize(points):
    # maxX - maximum of x coordinates of all pts, minX = minimum of x coordinates of all points. for y, the vars are defined similarly
    minX, minY = np.PINF, np.PINF
    maxX, maxY = np.NINF, np.NINF
    for c in points:
        x, y = c[0], c[1]
        minX, minY = np.amin([minX,x]), np.amin([minY,y])
        maxX, maxY = np.amax([maxX,x]), np.amax([maxY,y])
    
    # code for finding the center's coordinates and making the center the origin by shifting all points
    cenX, cenY = (minX+maxX)/2, (minY+maxY)/2
    N = len(points)
    for i in range(0,N):
        points[i][0] -= cenX
        points[i][1] -= cenY
    
    # code for rescaling the character to a 1 x 1 figure (ideally). however, because most characters have diff
    # widths and heights, we take a common scale factor for both dimensions
    idH = idW = 1
    curH = maxY - minY
    curW = maxX - minX
    r = max(idH/curH,idW/curW)
    for i in range(0,N):
        points[i][0] *= r
        points[i][1] *= r



# code to extract data from file. this is assumed to be in the proper format per specs.
def extractHCDData(file):
    lines = file.readlines()
    ln = lines[0].strip().split(' ')
    num_vectors = int(ln[0])        # entry 1 = num of points = n
    # subsequent entries are x[1] y[1] x[2] y[2] ... x[n] y[n] where x[i] and y[i] are coordinates of point i
    ln = ln[1:]
    vectors = []
    for i in range(0,num_vectors):
        l = np.array([np.double(ln[2*i]),np.double(ln[2*i+1])],dtype=np.double)
        vectors.append(l)
    normalize(vectors)              # normalize the character
    return np.array(vectors)        # return the character data as a numpy 2D array



# function for reading handwritten char data from its folder
def inputHCD():
    fdr = str(input('Enter the name of the folder containing the handwritten char data : '))

    # we have 5 classes of data as in the below list
    classlist = ['a', 'ai', 'chA', 'dA', 'lA']

    # code to extract training data from FOLDER/cname/train/ and build train data with their labels
    trainData = []
    trainLabels = []
    for cname in classlist:
        path = fdr + '/' + cname + '/train/'
        for fname in os.listdir(path):
            if(fname.__contains__('.txt')):
                with open(os.path.join(path,fname),'r') as f:
                    trainData.append(extractHCDData(f))
                    trainLabels.append(cname)

    # code to extract dev data from FOLDER/cname/dev/ and build dev data with their labels
    devData = []
    devTrueLabels = []
    for cname in classlist:
        path = fdr + '/' + cname + '/dev/'
        for fname in os.listdir(path):
            if(fname.__contains__('.txt')):
                with open(os.path.join(path,fname),'r') as f:
                    devData.append(extractHCDData(f))
                    devTrueLabels.append(cname)
    
    # we find the minimum length of the time series in both train data and dev data
    # and store the result in minlen
    minlen = np.PINF
    for t in trainData:
        minlen = min(minlen, len(t))
    for d in devData:
        minlen = min(minlen, len(d))

    # code to transform each train vector series to a fixed length supervector of dimension = minlen * 2
    # using the rolling window transform
    transformedTrainData = []
    for t in trainData:
        transformedTrainData.append(rollingWindowTransform(t, minlen))
    transformedTrainData = np.array(transformedTrainData)

    # code to transform each dev vector series to a fixed length supervector of dimension = minlen * 2
    # using the rolling window transform
    transformedDevData = []
    for d in devData:
        transformedDevData.append(rollingWindowTransform(d, minlen))
    transformedDevData = np.array(transformedDevData)

    # return the 5 tuple with train data, train labels, dev data, dev data labels, class names
    return (transformedTrainData, trainLabels, transformedDevData, devTrueLabels, classlist)
