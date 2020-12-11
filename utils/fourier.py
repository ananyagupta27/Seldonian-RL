import itertools

import numpy as np


def getCountlist(number_of_states , order):
    arr = []
    for i in itertools.product(np.arange(0,order + 1),repeat=(number_of_states)):
        arr.append(np.array(i))
    return np.array(arr)


def fourierBasis(state, order_list):
    state_new = np.array(state).reshape(1,-1)
    scalars = np.einsum('ij, kj->ik', order_list, state_new)
    phi = np.cos(np.pi*scalars)
    return phi


def get_transformed_state(env, state, theta, order=3):
    if env.discrete:
        state = env.getDiscreteState(state)
        discreteState = np.zeros(env.getNumDiscreteStates(), dtype=np.float32)
        discreteState[state] = 1
        return discreteState.reshape(-1,1)
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