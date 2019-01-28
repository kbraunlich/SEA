#!/usr/bin/python


import numpy as np
from time import time
overall_time = {"stim": 0.0, "learn": 0.0, "add": 0.0, "decide": 0.0}


# %%
def SoftMaxChoice(arrayV, d):
    '''Input:

    - arrayV[i] = choiceValuesD[i] = ComputeValueOfUnknownDim(... )
    - d: decision parameter'''
    tempV = np.exp(d * arrayV)
    tempV = tempV / np.add.reduce(tempV)
    tempV = np.cumsum(tempV)
    needle = np.random.uniform(0, 1)
    i = 0
    while tempV[i] < needle:
        i += 1
    return i


def ProbMatchChoice(arrayV, d):
    '''arrayV are the values, d is the decision parameter'''
    tempV = pow(arrayV, d)
    tempV = tempV / np.add.reduce(tempV)
    tempV = np.cumsum(tempV)
    needle = np.random.uniform(0, 1)
    i = 0
    while tempV[i] < needle:
        i += 1
    return i


def ComputeValueOfUnknownDim(model, theDim, itemA, knownL, recursion_level):
    '''given dim, item and known, returns the value of sampling the dim.

    - for dim i, iterate over values, j.
            - each time calculate feature value
            - return max(SituationValue, ComputeValueOfUnknownDim)

    this yields nDim vec, "outcomeValues"

    combine utility of each j with prob of each feature:
            - np.inner(self.Fpredictions[theDim],outcomeValues)
    '''
    recursion_level += 1
    if model.report:
        print(
            'recursion_level %d, ComputeValueOfUnknownDim%d' %
            (recursion_level, theDim))
    theKnown = np.zeros(model.NUM_DIMS)
    theKnown[knownL] = 1
    model.Stimulate(itemA, theKnown)

    # take value Prob of unobserved dimension (rat eq 2)
    valueProbs = np.copy(model.Fpredictions[theDim])
    outcomeValues = np.zeros(model.NUM_VALUES)
    tmpKnownL = list(knownL)
    tmpKnownL.append(theDim)

    for j in range(len(valueProbs)):
        tmpItemA = np.copy(itemA)
        tmpItemA[theDim] = j
        outcomeValues[j] = DetermineFeatureValue(
            model, tmpItemA, tmpKnownL, recursion_level)

    # combine utility of each feature value with probability of that value
    vUnknown = np.inner(valueProbs, outcomeValues)
    return vUnknown


def DetermineFeatureValue(model, stimulusA, observedL, recursion):
    '''Given stimulus, and observed features, F, recursively test \
    unsampled dimensions

    - get situationValue: max(self.Fpredictions[0]) * 100. - self.INFO_COST * \
    np.add.reduce(KNOWN_VEC)
    - for each unobserved dim, i,: ComputeValueOfUnknownDim, V_i(F)
    - if V_i(F) > [V(F) + C_i(F)]:

            - maxValue=V_i(F)
    - return maxValue
    '''
    knownA = np.zeros(model.NUM_DIMS)
    knownA[observedL] = 1

    maxValue = model.SituationValue(stimulusA, knownA)

    if model.report:
        print(' -- situationValue=%.2f' % maxValue)

    if recursion <= model.det_val_params['MAX_RECURSION_LEVEL']:
        for i in range(1, model.NUM_DIMS):  # (ignore category label)

            if i not in observedL:
                tmpStimulusA = np.copy(stimulusA)
                tmpObservedL = list(observedL)
                value = ComputeValueOfUnknownDim(
                    model, i, tmpStimulusA, tmpObservedL, recursion)
                if model.report:
                    print(' ---- unobserved dim%d value = %.2f' % (i, value))
                if value > maxValue:  # if V_i(F) >  V(F) + C_i(F)
                    maxValue = value
    return maxValue


def foo():
    foo.counter += 1

foo.counter = 0


def GetSampleSet(model, stimulus):
    '''Given a stimulus, returns an ordered list of dimensions to sample.
    If empty, the agent should respond immediately. Ccurrently assumes that
    all dimensions are initially unknown.

    input:

    - stimulus: the 1*nFt vector.

    calculates:

    - expected situation value:  max(self.Fpredictions[0]) * 100
    - exploration bonus : (100-expectSituationValue) / pow(model.totalNbyDimension+1,EXPLORATION_PARAM)
    - choiceValuesD dictionary =  key is the dimension, entry is value
    - choiceValuesD[i]: explorationBonus[i]+ ComputeValueOfUnknownDim(model, i, stimulus, sampleSet, 0) - expectSituationValue

    returns:

    - sampleSet:  ordered list of dimensions to sample. empty list means respond now
    '''
    if model.report:
        print('stim:', stimulus)
    foo()
    if (foo.counter > 1) and 0:
        raise ValueError('stim2')
    sampleSet = []

    # will build up known as features are sampled
    known = np.zeros(model.NUM_DIMS)

    chosen = -1  # dimension sampled
    while True:
        expectSituationValue = model.ExpectedValue(stimulus, known)

        explorationBonus = (
            100 - expectSituationValue) / pow(
            model.totalNbyDimension + 1,
            model.det_val_params['EXPLORATION_PARAM'])
        choiceValuesD = {}  # key = dimension, entry = value
        choiceValuesD[0] = 0  # current situation -  no exploration bonus
        for i in range(1, model.NUM_DIMS):  # leaving off category label
            if i not in sampleSet:
                choiceValuesD[i] = ComputeValueOfUnknownDim(
                    model, i, stimulus, sampleSet, 0) - expectSituationValue \
                        + explorationBonus[i]

        arrayV = np.array(np.zeros(len(choiceValuesD)))
        i = 0
        keys = list(choiceValuesD.keys())
        for key in keys:
            arrayV[i] = choiceValuesD[key]
            i += 1

        chosen = keys[SoftMaxChoice(
            arrayV, model.det_val_params['DECISION_PARAMETER'])]

        known[chosen] = 1

        if chosen == 0:
            break
        sampleSet.append(chosen)

    return sampleSet

# %%


class RMC:

    def __init__(self, param):
        self.param = param
        self.COUPLING = param['coupling']
        # first dimension of matrix indexes dimension, second indexes dimension
        # value.
        self.PRIOR_MAT = param['prior_matrix']
        self.D = param['d']
        self.report = param['report']
        self.NUM_DIMS = param['num_dims']
        self.NUM_VALUES = param['num_values']
        self.num_clus = 0
        self.totalN = 0  # n items experienced
        self.totalNbyDimension = np.zeros(
            self.NUM_DIMS)  # n times each dimension sampled
        # probability of every dimension for every value (2D structure).  0th
        # dimension is category label.
        self.Fpredictions = []
        # Float, number of times an item is assigned to each cluster.
        self.clusterN = []
        # Float, number of observations of each cluster/dimension/value combo +
        # prior.: 3 dimensional matrix.  first dimension indexes cluster, 2nd
        # the dimension, 3rd the dimension value.
        self.clusterF = []
        self.clusLike = []
        self.recog = []
        self.CreateCluster()
        self.det_val_params = {
            'MAX_RECURSION_LEVEL': param['MAX_RECURSION_LEVEL'],
            'EXPLORATION_PARAM': param['EXPLORATION_PARAM'],
            'DECISION_PARAMETER': param['DECISION_PARAMETER']}

    def getActionValue(self, action):
        '''
        .. math::
                 E(U(a|F_o)) = \sum_{s \in S} U(a|s)P(s|F_o)'''
        return np.sum([self.param['dfActionVals'].loc[state, 'a%d' % action] *
                       self.Fpredictions[0][action] for
                       state in self.param['dfActionVals'].index])

    def getStateVal_noCost(self):
        '''
        .. math::
                E(U(F_o)) = \max_{a \in A} (\mathbb{E}(U(a|F_o))) '''
        actionVals = [self.getActionValue(a) for a in range(
            len(self.param['dfActionVals']))]
        return np.max(actionVals)

    def getSituationCost(self, KNOWN_VEC):
        return self.param['getSituationCost'](KNOWN_VEC, self.param)

    def Reset(self):
        '''Reset model to init state'''
        self.clusterN = []
        self.clusterF = []
        self.clusLike = []
        self.recog = []
        self.Fpredictions = []
        self.totalN = 0
        self.num_clus = 0
        self.totalNbyDimension = np.zeros(self.NUM_DIMS)
        self.CreateCluster()  # set up the first cluster.

    def PresentStimulus(self, ITEM_VEC, KNOWN_VEC):
        '''returns KnownList: the ordered list of dims to sample '''
        KnownList = GetSampleSet(self, ITEM_VEC)
        theKnown = np.zeros(self.NUM_DIMS)
        theKnown[KnownList] = 1
        self.Stimulate(ITEM_VEC, theKnown)
        return KnownList

    def Stimulate(self, ITEM_VEC, KNOWN_VEC):
        '''calculates:

        - eq 4: priorProbOfClusters
        - probOfEachFeatureGivenCluster: (not eq6)
        - self.clusLike = probOfEachFeatureGivenCluster*priorProbOfClusters ft*dim
        - eq3 = self.clusLike=clusLike/sum(clusLike)
        - eq2 = self.Fpredictions'''
        start_time = time()

        self.priorProbOfClusters = (self.COUPLING * self.clusterN) / (
            (1 - self.COUPLING) + self.COUPLING * self.totalN)  # eq 4
        self.priorProbOfClusters[self.num_clus - 1] = (1.0 - self.COUPLING) / (
            (1.0 - self.COUPLING) + self.COUPLING * self.totalN)  # for hypothetical new cluster

        probOfEachFeatureGivenCluster = self.clusterF / np.reshape(
            np.sum(
                self.clusterF, 2), [
                self.num_clus, self.NUM_DIMS, 1])  # Includes new hypothetical cluster

        self.clusLike = np.array(np.ones(self.num_clus), np.float64)
        for i in np.arange(self.NUM_DIMS):  # prob(item | cluster)  Eq. 6
            if KNOWN_VEC[i]:
                # compute over clusters..
                self.clusLike *= probOfEachFeatureGivenCluster[:,
                                                               i, ITEM_VEC[i]]
        self.clusLike *= self.priorProbOfClusters

        self.recog = self.clusLike

        self.clusLike = self.clusLike / sum(self.clusLike)  # Equation 3

        # sum over all clusters:
        self.Fpredictions = sum(
            np.reshape(
                self.clusLike, [
                    self.num_clus, 1, 1]) * probOfEachFeatureGivenCluster, 0)  # eq 2

        for i in np.arange(self.NUM_DIMS):
            if KNOWN_VEC[i]:
                self.Fpredictions[i] = np.zeros(self.NUM_VALUES, np.float64)
                self.Fpredictions[i, ITEM_VEC[i]] = 1.0

        overall_time["stim"] += time() - start_time

    def ResponseCorrectProb(self, ITEM_VEC, QUERY_VEC):
        '''return Fpredictions (eq. 2)'''
        return self.Fpredictions[0][ITEM_VEC[0]]


    def SituationValue(self, ITEM_VEC, KNOWN_VEC):
        '''Takes a situation and computes its value by maximizing over the
        action.
        See eq. X.

        .. math::
                 \mathbb{E}(U(F_o)) = \max_{a}  [\sum_{s} U(a|s)P(s|F_o) ] -\sum_{d \in o} C_d

        '''
        self.Stimulate(ITEM_VEC, KNOWN_VEC)
        sv_nocost = self.getStateVal_noCost()
        cost = self.getSituationCost(KNOWN_VEC)
        sv = sv_nocost - cost

        if self.report:
            print(' -- Cost(%d known): %d' % (np.sum(KNOWN_VEC), cost))

        return sv

    def ExpectedValue(self, ITEM_VEC, KNOWN_VEC):
        '''ExpectedValue -- expected value of response, same as SituationValue,
        but without cost,

        - self.Stimulate
        - return max(self.Fpredictions[0] * state-value.)'''
        self.Stimulate(ITEM_VEC, KNOWN_VEC)

        return self.getStateVal_noCost()

    def CreateCluster(self):
        '''creates a new cluster and handles
                all learning for the trial.'''

        start_time = time()

        self.num_clus += 1  # increase cluster count.
        # setup novel hypothetical cluster
        self.clusterN = np.hstack([self.clusterN, np.array([0.0])])

        # initialize to prior, then add in observed dimensions
        newCluster = np.copy(self.PRIOR_MAT)

        if self.num_clus == 1:
            self.clusterF = np.array([newCluster])
        else:
            self.clusterF = np.vstack([self.clusterF, np.array([newCluster])])

        overall_time["add"] += time() - start_time

    def Learn(self, ITEM_VEC, KNOWN_VEC, QUERY_VEC):
        start_time = time()
        knownV = np.maximum(KNOWN_VEC, QUERY_VEC)
        self.Stimulate(ITEM_VEC, knownV)
        self.totalNbyDimension += knownV
        winner = np.argmax(self.clusLike)
        self.totalN += 1
        self.clusterN[winner] += 1.0
        for i in np.arange(self.NUM_DIMS):
            if knownV[i]:
                self.clusterF[winner, i, ITEM_VEC[i]] += 1.0
        if winner == self.num_clus - 1:
            # if the new hypothetical cluster won and is now an actual cluster,
            # create novel "hypothetical" cluster
            self.CreateCluster()

        overall_time["learn"] += time() - start_time

    def Report(self):
        '''	Prints various information about the status of a network'''
        print("Clusters:\n", self.clus_pos, "\n")
        print("Attention:\n", self.attention, "\n")
        print("Weights:\n", self.weights, "\n\n\n")
