# contains a class for KNN based classifier
# PRML A4 - Vikram CS19B021, Vedant Saboo CS19B074

import numpy as np

'''
    class for KNN classifier
'''
class KNN:
    '''
        initialize K = number of neighbours and seed the random module for determinism across runs
    '''
    def __init__(self, num_neighbours):
        self.num_neighbours = num_neighbours
        self.rng =  np.random.default_rng(12345)
    
    '''
        get train data with labels and the list of classes and store them
    '''
    def fit(self, trainvectors, trainlabels, classlist):
        self.trainvectors = trainvectors
        self.trainlabels = trainlabels
        self.classlist = classlist
        self.dim = trainvectors.shape[1]
    
    '''
        returns the list of classes of the K closest neighbours of a test vector
    '''
    def getNearestNeighbours(self, test):

        # code to find the distance of all train vectors from the test point and store the distances along with the class
        # in distances list
        distances = []
        count = len(self.trainvectors)
        for i in range(count):
            train = self.trainvectors[i]
            cls = self.trainlabels[i]
            distances.append([np.linalg.norm(test-train), cls])
        
        # sort the distances list and return the classes of the closest K train vectors
        distances.sort(key = lambda x : x[0])
        neighclasses = [distances[i][1] for i in range(self.num_neighbours)]
        return neighclasses
    
    '''
        predict the class for each test vector and return the prediction list
    '''
    def predict(self, testvectors):
        # list of predictions
        predictions = []
        for test in testvectors:
            # get neighbours of test vector
            neighbours = self.getNearestNeighbours(test)
            count = dict()
            # find the number of points from each class in test's K-neighbourhood
            for c in self.classlist:
                count[c] = 0
            for n in neighbours:
                count[n] += 1
            # get the largest number of representatives from a class
            large = max(count.values())
            # predclasses = list of classes which have large representatives in K-neighbourhood of test
            predclasses = []
            for c in self.classlist:
                if(count[c] == large):
                    predclasses.append(c)
            # predict the class randomly from predclasses
            idx = self.rng.integers(low=0,high=len(predclasses),size=1)[0]
            predictions.append(predclasses[idx])
        # return the predictions list
        return predictions

    '''
        predict the score (posterior P(Ci | X) for each class Ci, each test vector X) and return it.
    '''
    def predictscores(self,testvectors):
        # scores = dict from a class C to the scores of testvectors towards this class C
        scores = dict()
        # init scores for each class with an empty list
        for c in self.classlist:
            scores[c] = []
        # for each test vector
        for test in testvectors:
            # get neighbours of test vector
            neighbours = self.getNearestNeighbours(test)
            count = dict()
            # find the number of points from each class in test's K-neighbourhood
            for c in self.classlist:
                count[c] = 0
            for n in neighbours:
                count[n] += 1
            # score of test towards class C = count[c] / K = ratio of points from C in K-neighbourhood of test
            for c in self.classlist:
                scores[c].append(count[c]/self.num_neighbours)
        for c in self.classlist:
            scores[c] = np.array(scores[c])
        # return the scores dictionary
        return scores