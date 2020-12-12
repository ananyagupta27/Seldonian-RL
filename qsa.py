

from bounds.confidence_intervals import *
from optimizers.optimizer_library import *
from estimators.is_estimators import *



"""
Quasi-Seldonian RL Algorithm

Takes environment, episodes, importance sampling estimator, failure rate delta, and dataset
Return candidate solution after candidate selection optimization using an optimizer with barrier function
"""




class QSA:

    def __init__(self, env=None, episodes=100, fHat=PDIS, delta=0.01, candidateDataset=None, safetyDataset=None,
                 optimizer=CMA, cis=ttestLB):
        self.env = env
        self.episodes = episodes
        self.fHat = fHat
        self.delta = delta
        self.safetyDataSize = episodes
        self.candidateDataset = candidateDataset
        self.safetyDataset = safetyDataset
        self.BENCHMARK_PERFORMANCE = env.threshold
        self.feval = 0
        self.optimizer = optimizer
        self.cis = cis

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
        mean_estimate = result

        # check if the candidate solution passes the safety test
        passed, lb = self.safety_test(estimates, self.safetyDataSize, delta=self.delta, factor=2)

        if not passed:
            # add a barrier in case the safety constraint is violated
            result = -100000.0

            # adding the lower bound as we need to optimize the lower bound
            # just shifted by the barrier function
            result = result + lb

        # printing the current iteration results
        print("result=", -result, "fhat=", mean_estimate, "lowerbound=", lb, "passed=", passed,
              "benchmark performance=", self.BENCHMARK_PERFORMANCE)

        # negating the current function value as minimizing the function
        return -result



    def getCandidateSolution(self):


        """
        Candidate selection using the given optimizer finds the best value over the candidate dataset
        :return: the optimal value of parameter for the policy
        """


        # initializing parameter for the policy as all zeros which gives a random policy after softmax
        theta = np.zeros((self.env.getStateDims() * self.env.getNumActions()))

        # using given optimizer to get the best candidate solution which satisfies the safety constraint
        optimizer = self.optimizer(theta, self.candidateObjective)
        xMin = optimizer.run_optimizer()

        # return the candidate solution
        return xMin

    def getTotalDataset(self):
        """
        :return: total dataset => candidate + safety dataset
        """
        dataset = self.candidateDataset.copy()
        for k in dataset.keys():
            dataset[k].extend(self.safetyDataset[k])
        return dataset



    def safety_test(self, is_estimates, size=None, delta=0.01, factor=1):

        """
        performs safety test using high confidence lower bound
        checks if lower bound is above threshold with high confidence
        with a particular Cis e.g. ttest

        :param is_estimates: list of importance sampling estimates at the current value of policy parameter
        :param size: size of the safety dataset as we need to predict whether given candidate solution will
                    pass the safety test or not without the safety dataset but we have access to the safety dataset size
        :param delta: failure rate for the confidence interval
        :param factor: factor for increasing the confidence of the bound to be more conservative
                    possibly to prevent overfitting
        :return: passed: whether the safety test is passed or not
        :return: lower_bound: the lower bound for the interval
        """


        if not size:
            size = len(is_estimates)
        lb = self.cis(is_estimates, size, delta, factor)
        return lb >= self.BENCHMARK_PERFORMANCE, lb

    def objectiveWithoutConstraints(self, thetaToEvaluate):
        """
        Objective without taking into consideration the safety guarantees
        :param thetaToEvaluate: the current value of policy parameter
        :return: result from the importance sampling estimator for the return at the current theta
        """
        result, estimates = self.fHat(thetaToEvaluate, self.candidateDataset, self.episodes, self.env)
        return -result
