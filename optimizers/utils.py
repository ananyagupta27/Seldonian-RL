import numpy as np
import itertools


# returns count lists in base order
def getCountlist(number_of_states, order):
    count_list = []
    for i in itertools.product(np.arange(0, order + 1), repeat=(number_of_states)):
        count_list.append(np.array(i))
    return np.array(count_list)


def fourierBasis(state, order_list):
    '''
    Convert state to order-th Fourier basis
    '''

    state_new = np.array(state).reshape(1, -1)
    scalars = np.einsum('ij, kj->ik', order_list,
                        state_new)  # do a row by row dot product with the state. i = length of order list, j = state dimensions, k = 1
    phi = np.cos(3.14 * scalars)
    return phi


# print((getCountlist(2,4)).shape)
print(fourierBasis(np.array([3.1, 3.3]), getCountlist(2, 4)).shape)

# print(fourierBasis(np.array(3,3), (getCountlist(2,4)))
