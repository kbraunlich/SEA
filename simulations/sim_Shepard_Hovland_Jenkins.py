#!/usr/bin/python
'''
Shepard, Roger N., Carl I. Hovland, and Herbert M. Jenkins. (1961).
Learning & Memorization of Classifications. Psychological Monographs:
General and Applied 75(13).
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sea

# %%
param = {
    # how many steps one looksahead? 0 is myopic (1 step ahead) and so on..
    'MAX_RECURSION_LEVEL':  5,
    'EXPLORATION_PARAM':    0.0,
    'DECISION_PARAMETER':   1.0,
    'TOTAL_BLOCKS':         28,
    'CONSEC_BLOCKS':        2,
    'prior_matrix':         np.array([
                                [.01, .01],
                                [1.0, 1.0],
                                [1.0, 1.0],
                                [1.0, 1.0], ]),
    'num_dims': 4,
    'num_values': 2,
    'coupling': .3,
    'd': 1.0,
    'report': False,
    'NUM_TIMES'				: 100  # nreps of the six problems
}

# uses a binary feature (one unit) for the first three dims and two
# features for the output to facillitate choice.
SHEP_STIM = np.array((
                # type 1
                [[0, 0, 0, 0],  
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 1],
                 [1, 1, 1, 0],
                 [1, 1, 1, 1]],
                 
                 # type 2
                [[0, 0, 0, 0],  
                 [0, 0, 0, 1],
                 [1, 0, 1, 0],
                 [1, 0, 1, 1],
                 [1, 1, 0, 0],
                 [1, 1, 0, 1],
                 [0, 1, 1, 0],
                 [0, 1, 1, 1]],
                 
                 # type 3
                [[0, 0, 0, 0],  
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [1, 0, 1, 1],  # *
                 [1, 1, 0, 0],
                 [0, 1, 0, 1],  # *
                 [1, 1, 1, 0],
                 [1, 1, 1, 1]],
                 
                 # type 4
                [[0, 0, 0, 0],  
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [1, 0, 1, 1],  # *
                 [0, 1, 0, 0],  # *
                 [1, 1, 0, 1],
                 [1, 1, 1, 0],
                 [1, 1, 1, 1]],
                 
                 # type 5
                [[0, 0, 0, 0],  
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [1, 0, 1, 1],  # *
                 [1, 1, 0, 0],
                 [1, 1, 0, 1],
                 [1, 1, 1, 0],
                 [0, 1, 1, 1]],  # *
                 
                 # type 6
                [[0, 0, 0, 0],  
                 [1, 0, 0, 1],
                 [1, 0, 1, 0],
                 [0, 0, 1, 1],
                 [1, 1, 0, 0],
                 [0, 1, 0, 1],
                 [0, 1, 1, 0],
                 [1, 1, 1, 1]]))


SHEP_KNOWN = np.array([
                [0, 1, 1, 1],  # features known when stimulus is presented.
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 1, 1, 1]])

ALL_KNOWN = np.array([1, 1, 1, 1])


SHEP_QUERY = np.array([
                [1, 0, 0, 0],  # features queried
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]])


# human data from table 1 in Nosofsky 1994:
SHEP_HUMAN = pd.read_csv(
    'behavioral_results/nosofsky1994LearningCurves.csv',
    index_col=0).T.values

# % functions for computing utility:


def getSituationCost(KNOWN_VEC, param):
    '''function to calculate costs for a given KNOWN_VEC (see RMC.SituationValue)

    This function assumes equal costs for each query. Can be modified for
    tasks in which some tests are more costly than others, or when sampling costs
    are not independent.'''

    return np.sum(KNOWN_VEC) * 10  # 10 is sampling cost used in paper


def dfActionVals(param):
    '''function to define utility table (e.g., table 1)
    returns dataframe
    '''

    m = np.diag([100] * param['num_values'])
    df = pd.DataFrame(index=['s%d' % a for a in range(param['num_values'])],
                      columns=['a%d' % a for a in range(param['num_values'])],
                      data=m)
    return df


param['getSituationCost'] = getSituationCost
param['dfActionVals'] = dfActionVals(param)


# %% STANDARD MODEL

model = sea.RMC(param)

blocks_correct_consec = 0
overall_correct = np.zeros([6, param['TOTAL_BLOCKS']])
overall_sampled = np.zeros([6, param['TOTAL_BLOCKS']])
item_order = range(8)
lastFourBlocksSampled = np.zeros([6])
blocksToCriterion = np.zeros([6])


nDimSampledByStim = np.zeros((param['TOTAL_BLOCKS'], 6, len(item_order)))
nCorrectProbByStim = np.ones(
    (param['NUM_TIMES'], param['TOTAL_BLOCKS'], 6, len(item_order)))
mndim = np.zeros((param['NUM_TIMES'], 6, param['TOTAL_BLOCKS']))
dMod = {}
for problem_num in range(6):
    print "Running problem ", problem_num + 1
    dMod[problem_num] = {}
    for run_num in range(param['NUM_TIMES']):
        model.Reset()
        criterion = False
        sim_correct = np.zeros(param['TOTAL_BLOCKS'])
        numberDimSampled = np.zeros(param['TOTAL_BLOCKS'])
        for num_block in range(param['TOTAL_BLOCKS']):
                print(
                    'run_num=%d, num_block=%d' %
                    (run_num, num_block))
            np.random.shuffle(item_order)
            blockCorrectProb = 1.0

            for item_num in item_order:

                ITEM_VEC = SHEP_STIM[problem_num][item_num]
                KNOWN_VEC = SHEP_KNOWN[item_num]
                tempN = model.PresentStimulus(ITEM_VEC, KNOWN_VEC)

                nDimSampledByStim[num_block,
                                  problem_num, item_num] += len(tempN)
                SAMPLED_KNOWN = np.zeros(model.NUM_DIMS)
                SAMPLED_KNOWN[tempN] = 1

                numberDimSampled[num_block] += len(tempN)

                correctProb = model.ResponseCorrectProb(
                    SHEP_STIM[problem_num][item_num], SAMPLED_KNOWN)
                nCorrectProbByStim[run_num, num_block,
                                   problem_num, item_num] *= correctProb
                blockCorrectProb *= correctProb
                sim_correct[num_block] += correctProb
                model.Learn(
                    SHEP_STIM[problem_num][item_num],
                    SAMPLED_KNOWN,
                    SHEP_QUERY[item_num])

                mndim[run_num, problem_num, num_block] = model.num_clus

            if np.random.uniform(0, 1) < blockCorrectProb:
                blocks_correct_consec += 1
            else:
                blocks_correct_consec = 0

            if blocks_correct_consec == 4:
                sim_correct[(num_block + 1):] = 8.
                numberDimSampled[(num_block +
                                  1):] += float(numberDimSampled[num_block])
                criterion = True
                break

        dMod[problem_num][run_num] = model
        if criterion:
            blocksToCriterion[problem_num] += (num_block - 3)
        else:
            blocksToCriterion[problem_num] += 29
        overall_correct[problem_num] += sim_correct
        overall_sampled[problem_num] += numberDimSampled
        lastFourBlocksSampled[problem_num] += np.average(numberDimSampled[-4:])

# %%
write = 0
read = 0

mean_errors = (param['TOTAL_BLOCKS'] * 8 * param['NUM_TIMES'] -
               np.add.reduce(overall_correct, 1)) / (float(param['NUM_TIMES']))
print("TOTAL ERRORS", mean_errors)
print("last four blocks sampling", lastFourBlocksSampled /
      (8 * float(param['NUM_TIMES'])))

print("blocks to criterion", blocksToCriterion / float(param['NUM_TIMES']))
overall_correct /= (8. * param['NUM_TIMES'])
overall_correct = 1.0 - overall_correct
overall_sampled2 = overall_sampled / (8. * param['NUM_TIMES'])
df = pd.DataFrame(overall_correct, index=range(1, 7), columns=range(1, 29))
if write:
    df.to_csv('res/shj_acc_%dtimes.csv' % param['NUM_TIMES'])
if read:
    df = pd.read_csv(
        'res/shj_acc_%dtimes.csv' %
        param['NUM_TIMES'], index_col=0)


# -----------------------------------------------------------
# stimulus-specific effects:
mmndim = np.mean(mndim, axis=0)
mCorrectProbByStim = np.mean(nCorrectProbByStim, axis=0)
mn_nDimSampledByStim = nDimSampledByStim / param['NUM_TIMES']
bls = list(range(18, 27, 2))
for i, bl in enumerate(bls):
    plt.subplot(3, len(bls), i + 1)
    plt.imshow(mCorrectProbByStim[bl, :, :], vmin=.8, vmax=1.)
    plt.xticks(range(8))
    plt.xlabel('stim')
    plt.ylabel('prob#')
    plt.title('acc bl%.d' % bl)

    plt.subplot(3, len(bls), i + 1 + len(bls))
    plt.imshow(mn_nDimSampledByStim[bl, :, :], vmin=0, vmax=5)
    plt.xticks(range(8))
    plt.xlabel('stim')
    plt.ylabel('prob#')
    plt.title('sampled bl%.d' % bl)

    plt.subplot(3, len(bls), i + 1 + len(bls) * 2)
    plt.plot(mmndim[:, bl])
    plt.ylim((0, 10))
    plt.xticks(range(6))
    plt.title('nClstr bl%.d' % bl)

plt.tight_layout()

# overall correct
#         1      2    3    4    5     6
markers = ['x', 's', 'o', '+', 'D', 'o']
mfcs = ['none', 'k', 'none', 'none', 'none', 'k']
plt.figure(figsize=(4.5, 3))

# MODEL - - - - - - - - - - - - - - - -
dims = [1, 2, 3, 4, 5, 6]
for i, dim in enumerate(dims):
    plt.plot(np.arange(1, 29), df.loc[dim, :], label='Type %d' % dim,
             color='k', marker=markers[i], markersize=5, mfc=mfcs[i])

plt.title('Accuracy: Model')
plt.xlabel('Block')
plt.ylabel('Proportion Errors')
plt.ylim(-.05, .7)
plt.legend(bbox_to_anchor=(1, 1))
plt.yticks(np.arange(0., .7, .2))
plt.tight_layout()
plt.show()

if write:
    plt.savefig('images/shj_acc_mod_%dtimes.png' % param['NUM_TIMES'], dpi=300)
    plt.savefig('images/shj_acc_mod_%dtimes.svg' % param['NUM_TIMES'])
    plt.savefig(
        '/home/kurt/Dropbox/papers/paper_eye/figs/shj_acc_mod_%dtimes.png' %
        param['NUM_TIMES'], dpi=300)
    plt.savefig(
        '/home/kurt/Dropbox/papers/paper_eye/figs/shj_acc_mod_%dtimes.svg' %
        param['NUM_TIMES'])

# HUMAN (nosofsky1994b) - - - - - - - - - - - - - - - -
plt.figure(figsize=(4.2, 3))
for i, dim in enumerate(range(1, 7)):
    plt.plot(
        np.arange(
            1,
            26),
        np.transpose(SHEP_HUMAN)[
            :,
            dim -
            1],
        label='Type %d' %
        dim,
        color='k',
        marker=markers[i],
        markersize=5,
        mfc=mfcs[i])
plt.title('Accuracy: Behavior')
plt.xlabel('Block')
plt.ylabel('Proportion Errors')
plt.ylim(-.05, .7)
plt.xlim(0, 28)
plt.xticks(np.arange(0, 29, 5))
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()

if write:
    plt.savefig('images/shj_acc_nosBeh.png', dpi=300)
    plt.savefig('images/shj_acc_nosBeh.svg')
    plt.savefig(
        '/home/kurt/Dropbox/papers/paper_eye/figs/shj_acc_nosBeh.png',
        dpi=300)
    plt.savefig('/home/kurt/Dropbox/papers/paper_eye/figs/shj_acc_nosBeh.svg')

#  nosofksy1994b rational plot - - - - - - - - - - - - - - - -
plt.figure(figsize=(4.5, 3))
dfRat = pd.read_csv('behavioral_results/rational_shj.txt', header=None)
dfRat.columns = ['x', 'y']
dfRat['x'] = list(range(1, 17)) + list(range(1, 17)) + \
    list(range(1, 17)) + list(range(1, 17))
idxP = np.array([1] * 16 + [2] * 16 + [4] * 16 + [6] * 16)
m = np.vstack([dfRat.loc[idxP == 1, 'y'].values.T, dfRat.loc[idxP == 2, 'y'].values.T,
               dfRat.loc[idxP == 4, 'y'].values.T, dfRat.loc[idxP == 6, 'y'].values.T])
dfRat = pd.DataFrame(m)
for i, dim in enumerate([1, 2, 4, 6]):
    plt.plot(np.arange(1, 17), dfRat.loc[i, :], label='Type %d' % dim,
             color='b', marker=markers[dim - 1], markersize=5, mfc=mfcs[i])
plt.ylim(-.05, .7)
plt.xlim(0, 28)
plt.xticks(np.arange(0, 29, 5))
plt.legend(bbox_to_anchor=(1, 1))
plt.title('Accuracy: Rational (Nosofsky, 1994)')


df = pd.DataFrame(overall_sampled2, index=range(1, 7), columns=range(1, 29))
if write:
    df.to_csv('res/shep_nDimsSampled_%dtimes.csv' % param['NUM_TIMES'])
if read:
    df = pd.read_csv('res/shep_nDimsSampled_%dtimes.csv' % 1000, index_col=0)

plt.figure(figsize=(5, 3))
dims = [1, 2, 4, 6]

markers = ['x', 's', '+', 'o']
mfcs = ['none', 'k', 'none', 'k']
for i, dim in enumerate(dims):
    plt.plot(np.arange(1, 29), df.loc[dim, :], label='Type %d' % dim,
             color='k', marker=markers[i], markersize=5, mfc=mfcs[i])

plt.xlabel('Block')
plt.ylabel('n Dimensions')
plt.title('Sampling: Model')
plt.ylim([.75, 3.35])
plt.yticks([1, 2, 3])
plt.tight_layout()
plt.legend(bbox_to_anchor=(1, 1))

if write:
    plt.savefig(
        'images/shj_nDimsSampled_dims1246_%dtimes.png' %
        param['NUM_TIMES'], dpi=300)
    plt.savefig(
        'images/shj_nDimsSampled_dims1246_%dtimes.svg' %
        param['NUM_TIMES'])
    plt.savefig(
        '/home/kurt/Dropbox/papers/paper_eye/figs/shj_nDimsSampled_dims1246_%dtimes.png' %
        param['NUM_TIMES'], dpi=300)
    plt.savefig(
        '/home/kurt/Dropbox/papers/paper_eye/figsshj_nDimsSampled_dims1246_%dtimes.svg' %
        param['NUM_TIMES'])


# %% YOKED model

model = sea.RMC(param)

total_errors = np.zeros(6)
item_order = range(8)

pastSamples = pickle.load(open("past_samples/sample6"))

for problem_num in range(6):
    print "Running problem ", problem_num + 1
    for run_num in range(param['NUM_TIMES']):
        trial = 0
        model.Reset()

        blocks_for_run = len(pastSamples[problem_num][run_num]) / 8
        total_trials = range(8 * blocks_for_run)

        for num_block in range(blocks_for_run):
            np.random.shuffle(item_order)

            for item_num in item_order:
                print(
                    'run_num=%d, num_block=%d, problem=%d, item_num=%d' %
                    (run_num, num_block, problem_num + 1, item_num))

                tempN = pastSamples[problem_num][run_num][total_trials[trial]][1]

                SAMPLED_KNOWN = np.zeros(model.NUM_DIMS)
                SAMPLED_KNOWN[tempN] = 1
                itemN = item_num

                model.Stimulate(SHEP_STIM[problem_num][itemN], SAMPLED_KNOWN)
                trial += 1

                correctProb = model.ResponseCorrectProb(
                    SHEP_STIM[problem_num][itemN], SAMPLED_KNOWN)
                total_errors[problem_num] += 1 - correctProb

                model.Learn(
                    SHEP_STIM[problem_num][itemN],
                    SAMPLED_KNOWN,
                    SHEP_QUERY[itemN])

mean_errors = total_errors / param['NUM_TIMES']
print("TOTAL ERRORS", mean_errors)