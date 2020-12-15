import numpy as np
from scipy.stats import t


"""
This file includes several inequalities for confidence intervals around the given estimates,
The lower or upper bound in all cases is returned with confidence is 1 - delta
The factor parameter is for the training mode when we want to be conservative with the bound, e.g. to prevent overfitting
The size parameter is according to the mode if mode is candidate selection then assuming knowledge of the safety-
dataset size, the bound estimate is made.

These concentration inequalities are described in Phil's thesis
https://people.cs.umass.edu/~pthomas/papers/Thomas2015c.pdf
"""


"""
Function prototype
:param is_estimates: list of estimates
:param size: size of safety dataset
:param delta: bound returned with confidence 1-delta
:param factor: depending on mode - if mode is candidate selection then factor is 2, 
to be conservative with the bound, e.g. to prevent overfitting
                                    mode is safety test then factor is 1
:param b: upper limit for the possible values of estimates
:param a: lower limit for the possible values of estimates

:return: lower or upper bound
"""


# ttest
def ttestLB(is_estimates, size=None, delta=0.01, factor=1):

    if not size:
        size = len(is_estimates)
    lb = np.mean(is_estimates) - factor * (
            np.std(is_estimates, ddof=1) / np.sqrt(size)) * t.ppf(1 - delta, size - 1)
    return lb


def ttestUB(is_estimates, size=None, delta=0.01, factor=1):
    if not size:
        size = len(is_estimates)
    ub = np.mean(is_estimates) + factor * (
            np.std(is_estimates, ddof=1) / np.sqrt(size)) * t.ppf(1 - delta, size - 1)
    return ub


# Hoeffding
def HoeffdingLB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    return np.mean(is_estimates) - (b - a) * factor * (np.sqrt(np.log(1 / delta) / (2 * size)))


def HoeffdingUB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    return np.mean(is_estimates) + (b - a) * factor * (np.sqrt(np.log(1 / delta) / (2 * size)))


# Anderson
def AndersonLB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    is_estimates = np.append(is_estimates, a)
    is_estimates.sort()
    lb_n = [(is_estimates[i + 1] - is_estimates[i]) * min(1, (i / size) + np.sqrt(np.log(2 / delta) / (2 * size))) for i
            in range(0, int(size))]
    lb = (max(is_estimates) - sum(lb_n))
    return lb


def AndersonUB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    is_estimates = np.append(is_estimates, a)
    is_estimates.sort()
    is_estimates = np.append(is_estimates, b)

    ub_n = [(is_estimates[i+1]-is_estimates[i])*max(0, (i/size)-np.sqrt(np.log(2/delta)/(2*size))) for i in range(1,int(size+1))]
    ub = (1-sum(ub_n))
    return ub


# MPeB
def MPeBLB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    sample_mean = np.mean(is_estimates)
    sample_std = np.std(is_estimates, ddof=1)
    bound = (7 * (b - a) * (np.log(2 / delta))) / (3 * (size - 1)) + (
                np.sqrt((2 * np.log(2 / delta)) / size) * sample_std)
    lb = sample_mean - factor * bound
    return lb


def MPeBUB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    sample_mean = np.mean(is_estimates)
    sample_std = np.std(is_estimates, ddof=1)
    bound = (7 * (b - a) * (np.log(2 / delta))) / (3 * (size - 1)) + (
                np.sqrt((2 * np.log(2 / delta)) / size) * sample_std)
    ub = sample_mean + factor * bound
    return ub


# Refer to the paper for details
# https://people.cs.umass.edu/~pthomas/papers/Thomas2015.pdf

def PhilsAAAILB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    c = [b] * size

    c_inv_mean_inv = (np.sum([1 / i for i in c])) ** (-1)
    Y = [min(is_estimates[i], c[i]) for i in range(size)]

    ratio_sum = np.sum([Y[i] / c[i] for i in range(size)])
    ratio_std = np.std([Y[i] / c[i] for i in range(size)], ddof=1)
    empirical_mean = c_inv_mean_inv * ratio_sum

    bound = c_inv_mean_inv * ((7 * size * np.log(2 / delta)) / (3 * (size - 1))) + c_inv_mean_inv * np.sqrt(
        (np.log(2 / delta))) * ratio_std

    lb = (c_inv_mean_inv * ratio_sum - factor * bound)
    return lb



def PhilsAAAIUB(is_estimates, size=None, delta=0.01, factor=1, b=1, a=0):
    if not size:
        size = len(is_estimates)
    c = [b] * size

    c_inv_mean_inv = (np.sum([1 / i for i in c])) ** (-1)
    Y = [min(is_estimates[i], c[i]) for i in range(size)]

    ratio_sum = np.sum([Y[i] / c[i] for i in range(size)])
    ratio_std = np.std([Y[i] / c[i] for i in range(size)], ddof=1)
    empirical_mean = c_inv_mean_inv * ratio_sum

    bound = c_inv_mean_inv * ((7 * size * np.log(2 / delta)) / (3 * (size - 1))) + c_inv_mean_inv * np.sqrt(
        (np.log(2 / delta))) * ratio_std
    ub = (c_inv_mean_inv * ratio_sum + factor * bound)

    return ub