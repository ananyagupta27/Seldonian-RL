import numpy as np

from bounds.confidence_intervals import ttestLB
from optimizers.optimizer_library import CMA


class QSA:

    def __init__(self, env, episodes, fHat, gHats, deltas, candidateDataset, safetyDataset):
        self.env = env
        self.episodes = episodes
        self.fHat = fHat
        self.gHats = gHats
        self.deltas = deltas
        self.safetyDataSize = episodes
        self.candidateDataset = candidateDataset
        self.safetyDataset = safetyDataset
        self.BENCHMARK_PERFORMANCE = env.threshold
        self.feval = 0


    def candidateObjective(self, thetaToEvaluate):
        """
        This function takes in theta and dataset and safety constraints and returns the function value at
        the inputted theta using barrier function in case of violation of constraints

        :param thetaToEvaluate:
        :return: result
        """
        self.feval = self.feval + 1
        # importance sampling estimate
        result, estimates = self.fHat(thetaToEvaluate, self.candidateDataset, self.episodes, self.env)
        resf = result


        passed, lb = self.safety_test(estimates, self.safetyDataSize, delta=0.01, factor=2)

        if not passed:

            result = -100000.0

            result = result + lb

        print("result=", -result, "fhat=", resf, "upperboudn", lb, "passed", passed, self.BENCHMARK_PERFORMANCE)
        return -result

    def getCandidateSolution(self):
        theta = np.zeros((self.env.getStateDims() * self.env.getNumActions()))
        optimizer = CMA(theta, self.candidateObjective)
        xMin = optimizer.run_optimizer()
        return xMin


    def getTotalDataset(self):
        dataset = self.candidateDataset.copy()
        for k in dataset.keys():
            dataset[k].extend(self.safetyDataset[k])
        return dataset

    def safety_test(self, is_estimates, size=None, delta=0.01, factor=1):
        if not size:
            size = len(is_estimates)
        lb = ttestLB(is_estimates, size, delta, factor)
        return lb >= self.BENCHMARK_PERFORMANCE, lb


    def objectiveWithoutConstraints(self, thetaToEvaluate):
        result, estimates = self.fHat(thetaToEvaluate, self.candidateDataset, self.episodes, self.env)
        return -result