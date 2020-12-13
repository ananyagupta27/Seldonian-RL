import numpy as np
from scipy.stats import t
import math
import sys
import os
from sklearn.model_selection import train_test_split
from utils_dir.fourier import get_transformed_state

sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.cartpole import Cartpole


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


# function that takes softmax taking care of numerical overflows
# https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def normalize(theta, axis=1):
    STATES = theta.shape[0]
    theta = theta - np.max(theta, axis=axis).reshape(STATES, 1)
    theta = np.exp(theta)
    norm_theta = np.sum(theta, axis=axis)
    return theta / norm_theta.reshape(STATES, 1)


# get action give action probabilities by sampling from uniform random distribution
# and selecting according to the range
def get_action(actPr, axis=1):
    actions = actPr.shape[axis]
    temp = np.random.uniform(0, 1)
    sum_pr = 0

    for i in range(actions):
        sum_pr += actPr[0][i]
        if temp <= sum_pr:
            return i
    return actions - 1



def split_discrete_dataset(dataset, split_ratio):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    p = dataset['p']
    R = dataset['R']
    d0 = dataset['d0']
    V = dataset['V']
    Q = dataset['Q']
    states_train, states_test, actions_train, actions_test, rewards_train, rewards_test, pi_b_train, pi_b_test, \
    p_train, p_test, R_train, R_test, d0_train, d0_test, V_train, V_test, Q_train, Q_test = \
        train_test_split(states, actions, rewards, pi_b, p, R, d0, V, Q, test_size=split_ratio, random_state=42)
    safetyDataset = {'states': states_test, 'actions': actions_test, 'rewards': rewards_test,
                     'pi_b': pi_b_test, 'p': p_test, 'R': R_test, 'd0': d0_test, 'V': V_test, 'Q': Q_test}
    candidateDataset = {'states': states_train, 'actions': actions_train, 'rewards': rewards_train,
                        'pi_b': pi_b_train, 'p': p_train, 'R': R_train, 'd0': d0_train, 'V': V_train, 'Q': Q_train}

    return candidateDataset, safetyDataset


def split_dataset(dataset, split_ratio):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    states_train, states_test, actions_train, actions_test, rewards_train, rewards_test, pi_b_train, pi_b_test = \
        train_test_split(states, actions, rewards, pi_b, test_size=split_ratio, random_state=42)
    safetyDataset = {'states': states_test, 'actions': actions_test, 'rewards': rewards_test,
                     'pi_b': pi_b_test}
    candidateDataset = {'states': states_train, 'actions': actions_train, 'rewards': rewards_train,
                        'pi_b': pi_b_train}
    return candidateDataset, safetyDataset



def test():
    get_transformed_state(Cartpole(), np.array([1, 2, 3, 4]), np.zeros((256, 2)))


if __name__ == "__main__":
    test()