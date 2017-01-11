import tensorflow as tf
import numpy as np
import time
import math as mt
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.patches as mpatches
nom = 'testRun'
np.random.seed(1)
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


# Data saving directories
tensorboardDir ='/home/leon/Desktop/hyperHeuristics/tb'
imgDir = '/home/leon/Desktop/hyperHeuristics/img'
weightDir = '/home/leon/Desktop/hyperHeuristics/weights'


# Data graphing variables
globalAverage = []  # Averages over all games
localAverage = []  # Averages over games within the nEpisodes prior to backprop
totalHeuristicCounter = []
nBackprops = 0
graphLimit = 10
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
            score -= b
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
first_nHeuristics = 4
first_heuristics = {0: nextFit, 1: firstFit, 2:bestFit, 3:worstFit}
first_heuristicsNames = {0: 'nextFit', 1: 'firstFit', 2:'bestFit', 3:'worstFit'}

for i in range(0,first_nHeuristics):
    totalHeuristicCounter.append(0)


first_epsilon = .004 # Learning rate
first_gamma = .99  # Discount factor
first_nObservations = 10
first_nEpisodes = 10
first_nGamesPerEpisode = 10
first_rewardLimit = 50
first_nPieces=50;
first_nGames=1*10**5

first_hiddenLayer1 = 100 # Number of neurons on the first layer

first_fc1 = [first_nObservations, first_hiddenLayer1]
first_output = [first_hiddenLayer1, first_nHeuristics]
first_buffer = {0: np.zeros(first_fc1), 1: np.zeros(first_output)}
first_cache = {0: np.zeros(first_fc1), 1:np.zeros(first_output)}
first_rmspropDecay = .99
# Input
tf_first_observation = tf.placeholder(dtype=tf.float32, shape =[None, first_nObservations])
y_first = tf.placeholder(dtype=tf.float32, shape=[None, first_nHeuristics])
discounted_scores_first = tf.placeholder(dtype=tf.float32, shape=[None, 1])

# Hidden layer
W_first_fc1 = weight_variable(first_fc1,'first_fc1')
b_first_fc1 = bias_variable([first_fc1[1]], 'first_b')
h_first_fc1 = tf.nn.relu(tf.matmul(tf_first_observation, W_first_fc1) + b_first_fc1)

# Drop out
first_keep_prob = tf.placeholder(tf.float32)
drop_first_fc1 = tf.nn.dropout(h_first_fc1,first_keep_prob)

# Output layer
W_first_output = weight_variable(first_output, 'W_first_output')
b_first_output = bias_variable([first_output[1]],'b_first_output')
output_y_first = tf.nn.softmax(tf.matmul(drop_first_fc1,W_first_output)+b_first_output)

cross_entropy_first = tf.reduce_mean(-tf.reduce_sum(y_first*tf.log(output_y_first), reduction_indices=[1]))

weightedXent_first = tf.mul(cross_entropy_first, discounted_scores_first)

train_step_first = tf.train.GradientDescentOptimizer(first_epsilon)
grads_and_vars_first = train_step_first.compute_gradients(weightedXent_first, [W_first_fc1, W_first_output])

placeholder_gradients_first = []

for grad_var in grads_and_vars_first:
    placeholder_gradients_first.append((tf.placeholder('float', shape=grad_var[1].get_shape()), grad_var[1]))


applyGrads_first = train_step_first.apply_gradients(placeholder_gradients_first)

sess.run(tf.initialize_all_variables())

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
    for i in range(0,first_nGamesPerEpisode):

        # Start game
        game = True
        pieces = UBP(capacity, sMin, sMax, first_nPieces)
        bins = [0]

        while game:

            x = observation(bins, pieces, capacity)

            # Take decision using an stochastic approach

            netOutput = sess.run([output_y_first], feed_dict={tf_first_observation: x, first_keep_prob: 1})
            probs = np.cumsum(netOutput[0])
            r = np.random.random(1)
            j = 0
            bigger = True
            while bigger:
                if r < probs[j]:
                    action = j
                    bigger = False
                else:
                    j += 1
                    if j > 3:
                        print 'que pedo',probs, r
                        time.sleep(20)
            # Take action
            '''
            print 'Temp values step {0}, rand {1}, action {2}, probs {3}'.format(i,r,first_heuristicsNames[j], probs)
            time.sleep(3)
            print 'n pieces before game {0}, bins{1}'.format(pieces.size,bins)
            '''
            totalHeuristicCounter[j] += 1
            pieces, bins = first_heuristics[j](bins, pieces, capacity)

            '''
            print 'n pieces after game {0},bins {1}'.format(pieces.size,bins)
            time.sleep(5)
            '''

            # Find reward
            if np.size(pieces) % first_rewardLimit == 0:
                r = score(bins, capacity)
            else:
                r = 0

            y = np.zeros([1,first_nHeuristics])
            y[0,j] = 1
            rewards.append(r)
            observations.append(x)
            targets.append(y)

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
    discountedRewards = discount(epr , first_nGamesPerEpisode, first_rewardLimit, first_gamma)
    # Standardize rewards (see cs231n.github.io/neural-networks-2/)
    discountedRewards -= np.mean(discountedRewards)
    discountedRewards /= np.std(discountedRewards)

    # Get gradients and add them up!

    gradss = sess.run([g[0] for g in grads_and_vars_first], feed_dict ={tf_first_observation: epx, y_first: epy, discounted_scores_first: discountedRewards, first_keep_prob: 1})

    for k, v in enumerate(first_buffer):
        first_buffer[k] += gradss[k]
    if ep == first_nEpisodes:
        print 'entering backprop'
        # Backpropagate using  stacked gradients and rmsprop
        g1, g2 = sess.run([W_first_fc1, W_first_output])
        g=[g1, g2]
        feed_dict = {}
        for k, v in enumerate(first_buffer):
            first_cache[k] = first_rmspropDecay*first_cache[k]+(1-first_rmspropDecay)*g[k]**2
            feed_dict[placeholder_gradients_first[k][0]] = first_buffer[k]/(np.sqrt(first_cache[k])+1e-5)
            first_buffer[k] = np.zeros_like(first_buffer[k]) # Resetting grad buffer

        applyGrads_first.run(feed_dict = feed_dict)
        ep = 0 #Resetting episode run


        globalAverage.append(spaceUsed/games)
        localAverage.append(spaceUsedL/(first_nGamesPerEpisode*first_nEpisodes))

        print 'Resetting environment and backpropagating. average space filled = {0}'.format(globalAverage[-1])
        spaceUsedL = 0
        nBackprops += 1

        if nBackprops == graphLimit:
            print 'Updating graphs.'
            x = [k for k,v in first_heuristicsNames.iteritems()]
            width = 1/1.5
            normalizedCount = (np.vstack(totalHeuristicCounter)*1.0)/np.sum(totalHeuristicCounter)
            plt.figure()
            plt.bar(x, normalizedCount, color='blue')
            plt.savefig(imgDir+'/'+nom+'bar.png')
            plt.figure()
            plt.plot(np.vstack(globalAverage), color='blue',label ='Global')
            plt.plot(np.vstack(localAverage), color='green', label = 'Local')
            plt.legend()
            plt.savefig(imgDir+'/'+nom+'evolution.png')
            nBackprops = 0


# If the architecture generalizes but doesn't perform 'good' (average space filled)
