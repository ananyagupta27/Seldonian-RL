import numpy as np
import itertools
from scipy.stats import t
import math


# This function returns the inverse of Student's t CDF using the degrees of freedom in nu for the corresponding
# probabilities in p. It is a Python implementation of Matlab's tinv
# function: https://www.mathworks.com/help/stats/tinv.html
def tinv(p, nu):
    return t.ppf(p, nu)


# This function computes the sample standard deviation of the vector v, with Bessel's correction
def stddev(v):
    n = v.size
    variance = (np.var(v) * n) / (n - 1)  # Variance with Bessel's correction
    return np.sqrt(variance)  # Compute the standard deviation


# This function computes a (1-delta)-confidence upper bound on the expected value of a random
# variable using Student's t-test. It analyzes the data in v, which holds i.i.d. samples of the random variable.
# The upper confidence bound is given by
#    sampleMean + sampleStandardDeviation/sqrt(n) * tinv(1-delta, n-1),
#    where n is the number of points in v.
def ttestUpperBound(v, delta):
    n = v.size
    res = v.mean() + stddev(v) / math.sqrt(n) * tinv(1.0 - delta, n - 1)
    return res


# This function works similarly to ttestUpperBound, but returns a conservative upper bound. It uses
# data in the vector v (i.i.d. samples of a random variable) to compute the relevant statistics
# (mean and standard deviation) but assumes that the number of points being analyzed is k instead of |v|.
# This function is used to estimate what the output of ttestUpperBound would be if it were to
# be run on a new vector, v, containing values sampled from the same distribution as
# the points in v. The 2.0 factor in the calculation is used to double the width of the confidence interval,
# when predicting the outcome of the safety test, in order to make the algorithm less confident/more conservative.
def predictTTestUpperBound(v, delta, k):
    # conservative prediction of what the upper bound will be in the safety test for the a given constraint
    res = v.mean() + 2.0 * stddev(v) / math.sqrt(k) * tinv(1.0 - delta, k - 1)
    return res


# returns count lists in base order
def getCountlist(number_of_states, order):
    count_list = []
    for i in itertools.product(np.arange(0, order + 1), repeat=(number_of_states)):
        count_list.append(np.array(i))
    return np.array(count_list)


def fourierBasis(state, order_list):
    state_new = np.array(state).reshape(1, -1)
    scalars = np.einsum('ij, kj->ik', order_list,state_new)
    phi = np.cos(np.pi * scalars)
    return phi


def get_transformed_state(env, state, theta, order=3):

    state_copy = state.copy()
    for i, item in enumerate(env.observation_space.low):
        try:
            state_copy[i] = (state_copy[i] - env.observation_space.low[i]) / (
                env.observation_space.high[i] - env.observation_space.low[i])
        except OverflowError as e:
            print(state_copy)
    count_list = getCountlist(state_copy.shape[0], order)
    phi = fourierBasis(state_copy, count_list)
    return phi


def get_action(actPr):
    actions = actPr.shape[1]
    temp = np.random.uniform(0, 1)
    sum_pr = 0
    # print(actPr, actions)

    for i in range(actions):
        sum_pr += actPr[0][i]
        if temp <= sum_pr:
            return i
    return actions - 1
