import itertools
import numpy as np


"""
Refer to paper https://people.cs.umass.edu/~pthomas/papers/Konidaris2011a.pdf
for details on Fourier Basis
"""


def getCountlist(number_of_states, order):
    arr = []
    for i in itertools.product(np.arange(0,order + 1),repeat=(number_of_states)):
        arr.append(np.array(i))
    return np.array(arr)


def fourierBasis(state, order_list):
    state_new = np.array(state).reshape(1,-1)
    scalars = np.einsum('ij, kj->ik', order_list, state_new)
    # compute cosine to get the features phi
    phi = np.cos(np.pi*scalars)
    return phi


def get_transformed_state(env, state, theta, order=3):
    if env.discrete:
        # for discrete state environment simply return one hot vector for state
        # no need to use fourier basis
        state = env.getDiscreteState(state)
        discreteState = np.zeros(env.getNumDiscreteStates(), dtype=np.float32)
        discreteState[state] = 1
        return discreteState.reshape(-1,1)
    state_copy = state.copy()

    # normalize the state values before applying fourier basis
    for i, item in enumerate(env.observation_space.low):
        try:
            state_copy[i] = (state_copy[i] - env.observation_space.low[i]) / (
                env.observation_space.high[i] - env.observation_space.low[i])
        except OverflowError as e:
            print(state_copy)

    # get the count vectors
    count_list = getCountlist(state_copy.shape[0], order)

    # calculate the features given the current state and count vectors using the fourier basis
    phi = fourierBasis(state_copy, count_list)
    return phi