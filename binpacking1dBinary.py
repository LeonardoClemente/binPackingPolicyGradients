# Author: Cesar Leonardo Clemente Lopez // clemclem1991@gmail.com

import tensorflow as tf
import numpy as np
import time
import math as mt
import matplotlib.pyplot as plt
import random
import pickle
nom = 'piecesBinary'
np.random.seed(2)
random.seed(2)
# Convention : parameters outside tensorflow fooBar. inside tensorflow foo_bar
sess = tf.InteractiveSession()
tensorboardDir = ''

# Hyper-parameters
seed = 1  # random seed

capacity = 10  # Bin capacity
sMin = 1
sMax = capacity-1  # Piece size limits
N = 5000


bins = []
bins.append(0)


# Data saving directoriesimg:

tensorboardDir ='/Users/leonardo/Desktop/terashima/tb'
imgDir = '/Users/leonardo/Desktop/terashima/img'
weightDir = '/Users/leonardo/Desktop/terashima/tb'
txtDir = '/Users/leonardo/Desktop/terashima/txt'

ss = open(txtDir +'/' + nom + '.txt', 'w')
pFile = open(txtDir +'/' + 'pieces.txt', 'w')
# Data graphing variables
globalAverage = []  # Averages over all games
localAverage = []  # Averages over games within the nEpisodes prior to backprop
totalHeuristicCounter = []
nBackprops = 0
graphLimit = 100
# See if there are any interesting heuristic 'combos'

# Heuristics

def nextFit(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    nBin = np.size(bins) - 1
    spaceUsed = bins[nBin] + piece
    if spaceUsed < capacity:
        bins[nBin] = spaceUsed
    else:
        bins.append(piece)
    return pieces, bins


def firstFit(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    n = np.size(bins)
    for i in range(0, n):
        spaceUsed = bins[i] + piece
        if spaceUsed < capacity:
            bins[i] = spaceUsed
            return pieces, bins
    bins.append(piece)
    return pieces, bins


def bestFit(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    b = np.vstack(bins)
    d = capacity-piece-np.vstack(bins)
    d[d < 0] = 0
    v = np.amin(d)
    if v == 0:
        bins.append(piece)
    else:
        indices = [i for i, value in enumerate(d) if value == v]
        bins[indices[0]] += piece
    return pieces, bins


def worstFit(bins, pieces, capacity):
    piece = pieces[0]
    pieces = np.delete(pieces, 0)
    b = np.vstack(bins)
    d = capacity-piece-np.vstack(bins)
    d[d < 0] = 0
    v = np.amax(d)
    if v == 0:
        bins.append(piece)
    else:
        indices = [i for i, value in enumerate(d) if value == v]
        bins[indices[0]] += piece
    return pieces, bins


def firstFitDecreasing(bins, pieces, capacity): # Modify
    piece = np.amax(pieces)
    firstFit(bins, piece, capacity)


def bestFitDecreasing(bins, pieces, capacity): # Modify
    piece = np.amax(pieces)
    bestFit(bins, piece, capacity)


def observation(bins, pieces, capacity):
    o = np.zeros([1, 10])
    space = [v for i, v in enumerate(bins) if v < capacity] # Only interested in non-full bins
    space = capacity - np.vstack(bins)
    o[0, 0] = pieces[0] # Next piece
    o[0, 1] = pieces.size # Pieces left
    o[0, 2] = np.sum(pieces) # Total space from pieces left
    o[0, 3] = round(np.mean(pieces)) # Piece average size
    o[0, 4] = np.amin(space) # Min space left in a bin
    o[0, 5] = np.amax(space) # Max space left in a bin
    o[0, 7] = np.sum(space)/(space.size*capacity) # space used in %
    o[0, 8] = np.mean(space) # Average space left from a bin
    o[0, 9] = space.size # Number of opened bins
    return o

def score(bins, capacity):
    # max score per closed bin 10
    score = 0
    for i in range(0, np.size(bins)):
        b = bins[i]*1.0/capacity
        if  b == 1:
            score += 2
        elif b > .7:
            score += b
        else:
            score -= (1-b)
    return score


def discount(r, gamesPerEpisode, rewardLimit, gamma):
    dr = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, epr.size)):
        running_add = running_add*gamma + r[t]
        dr[t] = running_add
        if t % gamesPerEpisode == 0:
            running_add = 0  # Reset reward for the next game
    return dr


def UBP(c, sMin, sMax, N, seed=1):
    pieces = np.random.randint(sMin, sMax, size=[N, 1])
    return pieces.flatten()

# Tensorflow functions


def tf_UBP(c, sMin, sMax, N, seed=1):
    # Check CHaMP : Creating Heuristics via Many parameters, section 3.
    pieces = tf.random_uniform([N, 1], minval=sMin,maxval=sMax, dtype=tf.int32, seed=seed)

    return pieces


def weight_variable(shape, nom):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=nom)

def bias_variable(shape, nom):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=nom)


tf_pieces = tf_UBP(capacity, sMin, sMax, N, seed=1)

#First network : Fully connected neural net
# Toy problem : 50 pieces instances,  capacity = 10
# Score function : max number of bins
# Input: piece,allPieces, averagePieceSize, minSpace, maxSpace, allSpace, averageSpace, openBins
# Output
first_epsilon = .004 # Learning rate
first_gamma = .99  # Discount factor
first_nObservations = 10
first_nEpisodes = 10
first_nGamesPerEpisode = 10
first_rewardLimit = 50
first_nPieces=50;
first_nGames=3*10**5
learning_rate = 1e-4
first_nHeuristics = 1
first_heuristics = {0: nextFit, 1: firstFit, 2:bestFit, 3:worstFit}
first_heuristicsNames = {0: 'nextFit', 1: 'firstFit', 2:'bestFit', 3:'worstFit'}
first_rmspropDecay = .99
randCounter = 0
randCounterLimit = 30000
# Model
model = {}
model['W1'] = np.random.randn(100,10)*.1 / np.sqrt(10) # "Xavier" initialization
model['W2'] =  np.random.randn(1,100) *.1/ np.sqrt(100)
model_buffer = {}
model_cache = {}
for k,v in model.iteritems():
    model_buffer[k] = np.zeros_like(v)
    model_cache[k] = np.zeros_like(v)

for i in range(0,2):
    totalHeuristicCounter.append(0)

# Karpathys functions

def policy_backward2(eph, epdlogp):
	dW2 = np.dot(eph.T, epdlogp).ravel()
	dh = np.outer(epdlogp, model['W2'])
	dh[eph <= 0] = 0 # backpro prelu
	dW1 = np.dot(dh.T, epx)
	return {'W1':dW1, 'W2':dW2}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # sigmoid "squashing" function to interval [0,1]

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def policy_forward(x):
    h = np.dot(model['W1'], x.T)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = sigmoid(logp)
    return p, h  # return probability of taking action 2, and hidden state

def policy_backward(eph, epdlogp):
  """ backward pass. (eph is array of intermediate hidden states) """
  dW2 = np.dot(epdlogp.T,eph)
  dh = np.dot(epdlogp, model['W2'])
  dh[eph <= 0] = 0 # backpro prelu
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}
  




# RL part
games = 0
ep = 0
spaceUsed  = 0
spaceUsedL = 0
while games < first_nGames:

    # Start episode
    rewards = []
    observations = []
    targets = []
    hiddenStates = []
    dlogps = []
    for i in range(0,first_nGamesPerEpisode):

        # Start game
        game = True
        pieces = UBP(capacity, sMin, sMax, first_nPieces)
        pFile.write('{0} \n'.format(pieces[:]))
        bins = [0]

        while game:

            x = observation(bins, pieces, capacity)

            # Take decision using an stochastic approach

            probs, h = policy_forward(x)
            r = np.random.random(1)
            j = 2
            a = 0
            if r < probs:
            	j = 3
            	a =1  
            

            # Take action
            totalHeuristicCounter[a] += 1
            pieces, bins = first_heuristics[j](bins, pieces, capacity) # (only heuristics 2 and 3)


            # Find reward
            if np.size(pieces) % first_rewardLimit == 0:
                r = score(bins, capacity)
            else:
                r = 0

            y = j
            rewards.append(r)
            observations.append(x)
            hiddenStates.append(h.T)
            targets.append(y)
            dlogps.append(y-np.vstack(probs).T)
            if np.size(pieces) == 0:
                game = False

        games +=1
        newAverage = np.sum(bins)*1.0/(capacity*np.size(bins))
        spaceUsed += newAverage
        spaceUsedL += newAverage
    # After finishing the desired number of games for the episode, we gather stack the gradients
    ep += 1
    epr = np.vstack(rewards)
    epx = np.vstack(observations)
    epy = np.vstack(targets)
    eph = np.vstack(hiddenStates)
    epdlogp = np.vstack(dlogps)
    discountedRewards = discount(epr , first_nGamesPerEpisode, first_rewardLimit, first_gamma)
    # Standardize rewards (see cs231n.github.io/neural-networks-2/)
    discountedRewards -= np.mean(discountedRewards)
    discountedRewards /= np.std(discountedRewards)

    # Get gradients and add them up!
    epdlogp *= discountedRewards

    gradss = policy_backward(eph, epdlogp)

    for k, v in model.iteritems():
        model_buffer[k] += gradss[k]

    if ep == first_nEpisodes:
        print 'entering backprop'
        # Backpropagate using  stacked gradients and rmsprop
        for k, v in model.iteritems():
            model_cache[k] = first_rmspropDecay*model_cache[k]+(1-first_rmspropDecay)*model_buffer[k]**2
            model[k] += learning_rate * model_buffer[k] / (np.sqrt(model_cache[k]) + 1e-5)
            model_buffer[k] = np.zeros_like(v) # Resetting grad buffer

        ep = 0 #Resetting episode run


        globalAverage.append(spaceUsed/games)
        localAverage.append(spaceUsedL/(first_nGamesPerEpisode*first_nEpisodes))
        ss.write('{0} \n'.format(globalAverage[-1]))
        print 'Resetting environment and backpropagating. average space filled = {0}'.format(globalAverage[-1])
        spaceUsedL = 0
        nBackprops += 1

print 'Updating graphs.'
x = [1, 4]
width = 1/1.5
normalizedCount = (np.vstack(totalHeuristicCounter)*1.0)/np.sum(totalHeuristicCounter)
plt.figure()
plt.bar(x, normalizedCount, color='blue')
plt.savefig(imgDir+'/'+nom+'bar.png')
plt.close()
plt.figure()
plt.plot(np.vstack(globalAverage), color='blue',label ='Global')
plt.plot(np.vstack(localAverage), color='green', label = 'Local')
plt.legend()
plt.savefig(imgDir+'/'+nom+'evolution.png')
plt.close()
nBackprops = 0
pickle.dump(model, open(weightDir+'/'+nom+'.pickle', 'wb'))

ss.close
pFile.close
# If the architecture generalizes but doesn't perform 'good' (average space filled)
