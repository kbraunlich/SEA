#!/usr/bin/python

'''
 Davis, T., Love, B. C., & Preston, A. R. (2012). Striatal and hippocampal entropy and recognition signals in category learning: Simultaneous processes revealed by model-based fMRI. Journal of Experimental Psychology. Learning, Memory, and Cognition, 38(4)
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import sea

# %% simulation parameters
param = {
    # how many steps one looksahead? 0 is myopic (1 step ahead) and so on..
    'MAX_RECURSION_LEVEL':  5,
    'EXPLORATION_PARAM':    0.0,
    'DECISION_PARAMETER':   1.0,
    'TOTAL_BLOCKS':         30,
    'CONSEC_BLOCKS':        2,
    'prior_matrix':         np.array([
                                [.01, .01],
                                [1.0, 1.0],
                                [1.0, 1.0],
                                [1.0, 1.0],
                                [1.0, 1.0]]),
    'num_dims':             5,
    'num_values':           2,
    'coupling':             .3,
    'd':                    1.0,
    'report':               False,
    'NUM_TIMES':            50  # nreps of the six problems
}


LG_STIM = np.array([
    [0, 1, 1, 1, 1],  # exception A (first for later idx)
    [1, 0, 1, 1, 1],  # exception B (first for later idx)
    [0, 0, 0, 0, 1],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 0],])

LG_TRANS_STIM = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [1, 1, 0, 0, 1],])

# % functions for computing utility:


def getSituationCost(KNOWN_VEC, param):
    '''function to calculate costs for a given KNOWN_VEC (see RMC.SituationValue)

    This function assumes equal costs for each query. Can be modified for
    tasks in which some tests are more costly than others, or when sampling costs
    are not independent.'''

    return np.sum(KNOWN_VEC) * 10  # 10 is sampling cost used in paper


def dfActionVals(param):
    '''function to define utility table (e.g., table 1)

    When asym is True, this currently maximizes accuracy (i.e., correct=100
    utility units, incorrect=0 utility units).

    This function should be modified to reflect utility and cost associated
    with the final choice.'''

    m = np.diag([100] * param['num_values'])
    df = pd.DataFrame(index=['s%d' % a for a in range(param['num_values'])],
                      columns=['a%d' % a for a in range(param['num_values'])],
                      data=m)
    return df


param['getSituationCost'] = getSituationCost
param['dfActionVals'] = dfActionVals(param)


# %% STANDARD MODEL
model = sea.RMC(param)

LG_KNOWN = np.array([0, 1, 1, 1, 1])
LG_QUERY = np.array([1, 0, 0, 0, 0])
item_order = range(len(LG_STIM))

learningSamples = np.zeros(len(LG_STIM))
learningAcc = np.zeros(len(LG_STIM))

itemRecog = np.zeros(len(LG_STIM))
transRecog = np.zeros(len(LG_TRANS_STIM))
clusterPerSim = []

allSamples = []


for run_num in range(param['NUM_TIMES']):

    model.Reset()
    allSamples.append([])
    for num_block in range(
            param['TOTAL_BLOCKS']):        # there are 32 blocks of learning
        print(
            'run_num=%d, num_block=%d' %
            (run_num, num_block))
        np.random.shuffle(item_order)

        for item_num in item_order:
            tempN = model.PresentStimulus(
                LG_STIM[item_num], LG_KNOWN)  # ordered list to sample

            allSamples[run_num].append([item_num, tempN])

            SAMPLED_KNOWN = np.zeros(model.NUM_DIMS)
            SAMPLED_KNOWN[tempN] = 1

            correctProb = model.ResponseCorrectProb(
                LG_STIM[item_num], SAMPLED_KNOWN)

            learningSamples[item_num] += len(tempN)
            learningAcc[item_num] += correctProb

            model.Learn(LG_STIM[item_num], SAMPLED_KNOWN, LG_QUERY)

    clusterPerSim.append(len(model.clusterF) - 1)

    # Do Recognition
    for item in range(len(LG_STIM)):
        model.Stimulate(LG_STIM[item], LG_KNOWN)
        itemRecog[item] += sum(model.recog)

    for trans_item in range(len(LG_TRANS_STIM)):
        model.Stimulate(LG_TRANS_STIM[trans_item], LG_KNOWN)
        transRecog[trans_item] += sum(model.recog)


if 0:
    print("pickling allSamples to lgSample")
    pickle.dump(allSamples, open("past_samples/lgSample%d" % param['NUM_TIMES'], "w"))

stimTypes = ['exception', 'rule', 'trans']
dvs = ['learningSamples', 'learningAcc', 'itemRecog']
dfRes = pd.DataFrame(index=stimTypes, columns=dvs)
for stimType in stimTypes:
    if 'exception' in stimType:
        dfRes.loc[stimType, 'learningSamples'] = sum(
            learningSamples[0:2]) / (2. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'learningAcc'] = sum(
            learningAcc[0:2]) / (2. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'itemRecog'] = sum(
            itemRecog[0:2]) / (2. * param['NUM_TIMES'])
    elif 'rule' in stimType:
        dfRes.loc[stimType, 'learningSamples'] = sum(
            learningSamples[2:]) / (6. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'learningAcc'] = sum(
            learningAcc[2:]) / (6. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'itemRecog'] = sum(
            itemRecog[2:]) / (6. * param['NUM_TIMES'])
    elif 'trans' in stimType:
        dfRes.loc[stimType, 'itemRecog'] = sum(
            transRecog) / (8. * param['NUM_TIMES'])
print(dfRes)

print('clusterPerSim:', clusterPerSim)
plt.hist(clusterPerSim)
plt.show()


# %% YOKED
model = sea.RMC(param)

LG_KNOWN = np.array([0, 1, 1, 1, 1])
LG_QUERY = np.array([1, 0, 0, 0, 0])
item_order = range(len(LG_STIM))

learningSamples = np.zeros(len(LG_STIM))
learningAcc = np.zeros(len(LG_STIM))

itemRecog = np.zeros(len(LG_STIM))
transRecog = np.zeros(len(LG_TRANS_STIM))
clusterPerSim = []

pastSamples = pickle.load(open("past_samples/lgSample1000"))

for run_num in range(param['NUM_TIMES']):

    model.Reset()
    trial = 0
    for num_block in range(param['TOTAL_BLOCKS']):
        print(
            'run_num=%d, num_block=%d' %
            (run_num, num_block))
        np.random.shuffle(item_order)

        for item_num in item_order:
            tempN = pastSamples[run_num][trial][1]
            trial += 1
            SAMPLED_KNOWN = np.zeros(model.NUM_DIMS)
            SAMPLED_KNOWN[tempN] = 1
            model.Stimulate(LG_STIM[item_num], SAMPLED_KNOWN)

            correctProb = model.ResponseCorrectProb(
                LG_STIM[item_num], SAMPLED_KNOWN)

            learningSamples[item_num] += len(tempN)
            learningAcc[item_num] += correctProb

            model.Learn(LG_STIM[item_num], SAMPLED_KNOWN, LG_QUERY)

    # number of observations of each cluster/dimension/value combo + prior.
    # first dimension indexes cluster, 2nd the dimension, 3rd the dimension
    # value.
    print(np.around(model.clusterF, 2))
    clusterPerSim.append(len(model.clusterF) - 1)

    # Do Recognition
    for item in range(len(LG_STIM)):
        model.Stimulate(LG_STIM[item], LG_KNOWN)
        itemRecog[item] += sum(model.recog)

    for trans_item in range(len(LG_TRANS_STIM)):
        model.Stimulate(LG_TRANS_STIM[trans_item], LG_KNOWN)
        transRecog[trans_item] += sum(model.recog)


stimTypes = ['exception', 'rule', 'trans']
dvs = ['learningSamples', 'learningAcc', 'itemRecog']
dfRes = pd.DataFrame(index=stimTypes, columns=dvs)
for stimType in stimTypes:
    if 'exception' in stimType:
        dfRes.loc[stimType, 'learningSamples'] = sum(learningSamples[0:2])(
            2. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'learningAcc'] = sum(
            learningAcc[0:2]) / (2. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'itemRecog'] = sum(
            itemRecog[0:2]) / (2. * param['NUM_TIMES'])
    elif 'rule' in stimType:
        dfRes.loc[stimType, 'learningSamples'] = sum(
            learningSamples[2:]) / (6. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'learningAcc'] = sum(
            learningAcc[2:]) / (6. * param['TOTAL_BLOCKS'] * param['NUM_TIMES'])
        dfRes.loc[stimType, 'itemRecog'] = sum(
            itemRecog[2:]) / (6. * param['NUM_TIMES'])
    elif 'trans' in stimType:
        dfRes.loc[stimType, 'itemRecog'] = sum(
            transRecog) / (8. * param['NUM_TIMES'])
print(dfRes)

print(clusterPerSim)
plt.hist(clusterPerSim)
plt.show()
