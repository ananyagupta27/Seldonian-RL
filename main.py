import numpy as np
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
import sys
import os
# import ray                  # To allow us to execute experiments in parallel
# ray.init()
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld import Gridworld
from environments.cartpole import Cartpole
from environments.mountaincar import Mountaincar

from optimizers.cem import CEM
from optimizers.powell import Powell
from optimizers.cmaes import CMAES
from helper import *
from is_estimates import *
from create_dataset import Dataset

# print(int(sys.argv[1]), int(sys.argv[2]))
param1 = int(sys.argv[1])
param2 = int(sys.argv[2])


class QSA:

    def __init__(self, env, episodes, fHat, gHats, deltas):
        self.env = env
        self.episodes = episodes
        self.fHat = fHat
        self.gHats = gHats
        self.deltas = deltas
        fourierBasisOrder = 4
        self.safetyDataSize = episodes

        datasetGenerator = Dataset(episodes, env)
        theta = np.zeros((env.getStateDims(), env.getNumActions()))
        self.candidateDataset = datasetGenerator.generate_dataset(theta)
        theta = np.zeros((env.getStateDims(), env.getNumActions()))
        self.safetyDataset = datasetGenerator.generate_dataset(theta)
        # print((self.candidateDataset['rewards'][0]))

    # , candidateDataset, fHat, gHats, deltas, safetyDataSize

    def candidateObjective(self, thetaToEvaluate):
        """
        This function takes in theta and dataset and safety constraints and returns the function value at
        the inputted theta using barrier function in case of violation of constraints

        :param thetaToEvaluate:
        :return: result
        """
        # importance sampling estimate
        result, estimates = self.fHat(thetaToEvaluate, self.candidateDataset, self.episodes, self.env)
        resf = result
        predictSafetyTest = True  # Prediction of what the safety test will return. Initialized to "True" = pass
        for i in range(len(self.gHats)):  # Loop over behavioral constraints, checking each
            g = self.gHats[i]  # The current behavioral constraint being checked
            delta = self.deltas[i]  # The confidence level of the constraint

            # This is a vector of unbiased estimates of g_i(thetaToEvaluate)
            g_samples = g(estimates)
            # print(g_samples)
            # Get the conservative prediction of what the upper bound on g_i(thetaToEvaluate) will be in the safety test
            upperBound = predictTTestUpperBound(g_samples, delta, self.safetyDataSize)

            # We don't think the i-th constraint will pass the safety test if we return this candidate solution
            if upperBound > 0.0:

                if predictSafetyTest:
                    # Set this flag to indicate that we don't think the safety test will pass
                    predictSafetyTest = False

                    # Put a barrier in the objective. Any solution that we think will fail the safety test will have a
                    # large negative performance associated with it
                    result = -100000.0

                # Add a shaping to the objective function that will push the search toward solutions that will pass
                # the prediction of the safety test
                result = result - upperBound

        # Negative because our optimizer (Powell) is a minimizer, but we want to maximize the candidate objective
        # print("result=",-result,"fhat=",resf, "upperboudn", upperBound)
        return -result

    def getCandidateSolution(self):
        """

        This function calls the optimizer with the evaluation function and dataset
        and gets back argmin of the evaluation function or returns the candidate solution

        :param candidataDataset:
        :param episodes:
        :param fHat:
        :param gHats: safety constraints
        :param deltas: failure rates for safety constraints
        :param safetyDataSize: size of the safety dataset
        :return: candidate solution
        """
        # theta = np.zeros((256 * 2))
        theta = np.zeros((self.env.getStateDims() * self.env.getNumActions()))
        optimizer = CEM(theta, self.candidateObjective)
        xMin = optimizer.run_optimizer()
        return xMin

    # def QSA(self, env, episodes, fHat, gHats, deltas):
    # fourierBasisOrder = 4
    # safetyDatasetSize = episodes
    #
    # datasetGenerator = Dataset(env, episodes)
    # theta = np.zeros((env.getStateDims() ** fourierBasisOrder, env.getNumActions()))
    # candidateDataset = datasetGenerator.generate_dataset(theta)
    # theta = np.zeros((env.getStateDims() ** fourierBasisOrder, env.getNumActions()))
    # safetyDataset = datasetGenerator.generate_dataset(theta)
    #
    # # Get the candidate solution
    # candidateSolution = self.getCandidateSolution(candidateDataset, episodes, fHat, gHats, deltas, safetyDatasetSize)
    #
    # # Run the safety test
    # passedSafety = self.safetyTest(candidateSolution, safetyDataset, fHat, gHats, deltas)
    #
    #
    #
    # if passedSafety:
    #     print("A solution was found: ", candidateSolution)
    #     print("fHat of solution (computed over all data, D):", fHat(candidateSolution, dataset))
    # else:
    #     print("No solution found")
    #
    # # Return the result and success flag
    # return [candidateSolution, passedSafety]

    def getTotalDataset(self):
        dataset = self.candidateDataset.copy()
        for k in dataset.keys():
            dataset[k].extend(self.safetyDataset[k])
        return dataset

    def safetyTest(self, candidateSolution):
        result, estimates = self.fHat(candidateSolution, self.safetyDataset, self.episodes, self.env)
        print("result in safety test ", result)
        for i in range(len(self.gHats)):  # Loop over behavioral constraints, checking each
            g = self.gHats[i]  # The current behavioral constraint being checked
            delta = self.deltas[i]  # The confidence level of the constraint

            # This is a vector of unbiased estimates of g(candidateSolution)
            g_samples = g(estimates)
            # print(g_samples)
            # Check if the i-th behavioral constraint is satisfied
            upperBound = ttestUpperBound(g_samples, delta)
            # print(upperBound)

            if upperBound > 0.0:
                print("Failed on ", i, "g hat", upperBound)# If the current constraint was not satisfied, the safety test failed
                return False

        # If we get here, all of the behavioral constraints were satisfied
        return True


def gHatCartpole(estimates):
    # return 100 - estimates
    return 170 - estimates
    # return 185 - estimates


def gHat1Gridworld(estimates):
    return -130 - estimates
    # return 50 - estimates


def gHat2Gridworld(estimates):
    return estimates + 1


def gHat1Mountaincar(estimates):
    # return 100 - estimates
    return -830 - estimates
    # return -760 - estimates


def gHat2Mountaincar(estimates):
    # return -830 - estimates
    return estimates + 1

# @ray.remote
def testCartpole(ms, numM, numTrials):
    for trial in range(numTrials):
        for m in [param2]:
            env = Cartpole()
            # np.random.seed(0)  # Create the random number generator to use, with seed zero
            episodes = m
            print("m=",m,"Number of episodes =", episodes)
            # Create the behavioral constraints - each is a gHat function and a confidence level delta
            gHats = [gHatCartpole]
            deltas = [0.1]
            fHat = PDIS

            qsa = QSA(env, episodes, fHat, gHats, deltas)  # Run the Quasi-Seldonian algorithm

            # Get the candidate solution
            candidateSolution = qsa.getCandidateSolution()

            # Run the safety test
            passedSafety = qsa.safetyTest(candidateSolution)

            if passedSafety:
                print("m=",m,"A solution was found: ", candidateSolution)
                print("m=",m,"fHat of solution (computed over all data, D):", fHat(candidateSolution, qsa.getTotalDataset(), episodes, env))
            else:
                print("m=",m,"No solution found")

    # Return the result and success flag
    return [candidateSolution, passedSafety]

# @ray.remote
def testMountaincar(ms, numM, numTrials):
    for trial in range(numTrials):
        for m in [param2]:
            env = Mountaincar()
            # np.random.seed(0)  # Create the random number generator to use, with seed zero
            episodes = m
            print("m=",m,"Number of episodes =", episodes)
            # Create the behavioral constraints - each is a gHat function and a confidence level delta
            gHats = [gHat1Mountaincar]
            deltas = [0.1]
            fHat = PDIS
            # fHat = total_return
            qsa = QSA(env, episodes, fHat, gHats, deltas)  # Run the Quasi-Seldonian algorithm

            # Get the candidate solution
            candidateSolution = qsa.getCandidateSolution()

            # Run the safety test
            passedSafety = qsa.safetyTest(candidateSolution)

            if passedSafety:
                print("m=",m,"A solution was found: ", candidateSolution)
                print("m=",m,"fHat of solution (computed over all data, D):",
                      fHat(candidateSolution, qsa.getTotalDataset(), episodes, env))
            else:
                print("m=",m,"No solution found")

    # Return the result and success flag
    return [candidateSolution, passedSafety]

# @ray.remote
def testGridworld(ms, numM, numTrials):
    for trial in range(numTrials):
        for m in [param2]:
            env = Gridworld()
            # np.random.seed(0)  # Create the random number generator to use, with seed zero
            episodes = m
            print("m=",m,"Number of episodes =", episodes)

            # Create the behavioral constraints - each is a gHat function and a confidence level delta
            gHats = [gHat1Gridworld, gHat2Gridworld]
            deltas = [0.1, 0.1]
            fHat = PDIS

            qsa = QSA(env, episodes, fHat, gHats, deltas)  # Run the Quasi-Seldonian algorithm

            # Get the candidate solution
            candidateSolution = qsa.getCandidateSolution()

            # Run the safety test
            passedSafety = qsa.safetyTest(candidateSolution)

            if passedSafety:
                print("m=",m,"A solution was found: ", candidateSolution)
                print("m=",m,"fHat of solution (computed over all data, D):",
                      fHat(candidateSolution, qsa.getTotalDataset(), episodes, env))
            else:
                print("m=",m,"No solution found")

    # Return the result and success flag
    return [candidateSolution, passedSafety]

# @ray.remote
def main():
    # print(int(sys.argv[1]), int(sys.argv[2]))
    env_map = {0:'Mountaincar', 1:'Gridworld', 2:'Cartpole'}
    env_choice = param1
    print("Running environment ", env_choice, " Name ", env_map[env_choice])
    if env_choice == 0:
        func = testMountaincar
    elif env_choice == 1:
        func = testGridworld
    else:
        func = testCartpole

    nWorkers = 5
    ms = [2 ** i for i in range(5, 10)]
    numM = len(ms)
    numTrials = 10
    # _ = ray.get([func.remote(ms, numM, numTrials) for worker_id in
    #              range(1, nWorkers + 1)])

    func(ms, numM, numTrials)

if __name__ == "__main__":
    main()
