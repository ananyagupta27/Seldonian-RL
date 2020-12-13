import timeit
import sys
import os
from sklearn.model_selection import train_test_split
import ray  # To allow us to execute experiments in parallel
from arg_parser import parse_args
from environments.gridworldv2 import Gridworldv2
from qsa import QSA

ray.init()

from environments.gridworld687 import Gridworld687
from environments.gridworldv1 import Gridworldv1
from environments.mountaincar import Mountaincar

from optimizers.optimizer_library import *
from optimizers.cem import *
from optimizers.cmaes import *
from estimators.is_estimators import *
from data.create_dataset import Dataset
from data.create_model import Model
from bounds.confidence_intervals import *


bin_path = 'experiment_results/bin/'




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


# @ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, split_ratio, env, delta, fHat, cis, optimizer):
    # Results of the Seldonian algorithm runs
    seldonian_solutions_found = np.zeros((numTrials, numM))  # Stores whether a solution was found (1=True,0=False)
    seldonian_failures_g1 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1

    seldonian_fs = np.zeros((numTrials, numM))  # Stores the primary objective values (fHat) if a solution was found

    # Results of the Least-Squares (LS) linear regression runs
    LS_solutions_found = np.ones((numTrials, numM))  # Stores whether a solution was found. These will all be true (=1)
    LS_failures_g1 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1

    LS_fs = np.zeros((numTrials, numM))  # Stores the primary objective values (f) if a solution was found

    # Prepares file where experiment results will be saved
    experiment_number = worker_id
    outputFile = bin_path + 'results%d.npz' % experiment_number
    print("Writing output to", outputFile)

    for trial in range(numTrials):
        for (mIndex, m) in enumerate(ms):

            datasetGenerator = Dataset(m, env)
            theta = np.zeros((env.getStateDims(), env.getNumActions()))
            dataset = datasetGenerator.generate_dataset(theta)

            if args.discrete:
                datasetGenerator = Dataset(int(m * (1 - split_ratio)), env)
                candidateDataset = datasetGenerator.generate_dataset(theta)
                model = Model(env, candidateDataset, int(m * (1 - split_ratio)), env.getNumDiscreteStates(), env.getNumActions(), env.horizonLength)
                candidateDataset = model.makeMLEModel()
                datasetGenerator = Dataset(int(m), env)
                theta = np.zeros((env.getStateDims(), env.getNumActions()))
                safetyDataset = datasetGenerator.generate_dataset(theta)
                model = Model(env, safetyDataset, m, env.getNumDiscreteStates(), env.getNumActions(), env.horizonLength)
                safetyDataset = model.makeMLEModel()
            else:
                candidateDataset, safetyDataset = split_dataset(dataset, split_ratio)

            print("starting")

            # candidate dataset size
            m = int(m * (1 - split_ratio))

            qsa = QSA(env, m, fHat, delta, candidateDataset, safetyDataset,
                      cis=cis, optimizer= optimizer)  # the Quasi-Seldonian algorithm

            # Get the candidate solution
            solution = qsa.getCandidateSolution()

            # Perform safety test on safety dataset
            result, estimates = qsa.fHat(solution, safetyDataset, m, env)
            passedSafetyTest, lb = qsa.safety_test(estimates, m, delta=0.01, factor=1)

            # Run the Quasi-Seldonian algorithm

            if passedSafetyTest:
                seldonian_solutions_found[trial, mIndex] = 1
                seldonian_failures_g1[
                    trial, mIndex] = 1 if result < env.threshold else 0  # Check if the first behavioral constraint was violated

                seldonian_fs[trial, mIndex] = result  # Store the result on safety dataset
                print(
                    f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] A solution was found: [{solution[0]:.10f}, {solution[1]:.10f}]\tfHat over test data: {result:.10f}")
            else:
                seldonian_solutions_found[trial, mIndex] = 0  # A solution was not found
                seldonian_failures_g1[trial, mIndex] = 0  # Returning NSF means the first constraint was not violated
                seldonian_fs[
                    trial, mIndex] = None  # This value should not be used later. We use None and later remove the None values
                print(
                    f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] No solution found",
                    lb)

            theta = np.zeros((env.getStateDims() * env.getNumActions()))
            optimizer_function = optimizer(theta, qsa.objectiveWithoutConstraints)
            xMin = optimizer_function.run_optimizer()

            # # Run the Least Squares algorithm
            # theta = leastSq(trainX, trainY)  # Run least squares linear regression
            trueEstimate, totalEstimates = qsa.fHat(xMin, safetyDataset, m,
                                                    env)  # Get the "true" mean squared error using the testData
            LS_failures_g1[
                trial, mIndex] = 1 if trueEstimate < env.threshold else 0  # Check if the first behavioral constraint was violated
            LS_fs[trial, mIndex] = trueEstimate  # Store the "true" negative mean-squared error
            print(
                f"[(worker {worker_id}/{nWorkers}) LeastSq   trial {trial + 1}/{numTrials}, m {m}] LS fHat over test data: {trueEstimate:.10f}")
        print()

    np.savez(outputFile,
             ms=ms,
             seldonian_solutions_found=seldonian_solutions_found,
             seldonian_fs=seldonian_fs,
             seldonian_failures_g1=seldonian_failures_g1,
             LS_solutions_found=LS_solutions_found,
             LS_fs=LS_fs,
             LS_failures_g1=LS_failures_g1)


def main(args):
    # Map for the environment choice to environment
    env_map = {0: 'Mountaincar', 1: 'Gridworldv1', 2: 'Gridworldv2', 3: 'Gridworld687', 4: 'Cartpole'}

    env_choice = args.environment
    delta = args.delta

    print("Running environment ", env_choice, " Name ", env_map[env_choice])

    # selecting the IS estimator based on parameter
    if args.is_estimator == 'PDIS':
        fHat = PDIS
    elif args.is_estimator == 'IS':
        fHat = IS
    elif args.is_estimator == 'WIS':
        fHat = WIS
    elif args.is_estimator == 'DR':
        fHat = DR
        args.discrete = 1
    elif args.is_estimator == 'DR_hat':
        fHat = DR_hat
        args.discrete = 1
    else:
        print("Not supported estimator", args.is_estimator)
        exit(0)

    # choosing environment based on parameter
    if env_choice == 0:
        if args.discrete:
            env = Mountaincar(discrete=True)
        else:
            env = Mountaincar()
    elif env_choice == 1:
        env = Gridworldv1()
    elif env_choice == 2:
        env = Gridworldv2()
    elif env_choice == 3:
        env = Gridworld687()
    elif env_choice == 4:
        env = Cartpole()
    else:
        print("Wrong environment choice", env_choice)
        exit(0)

    # ttest, Anderson, MPeB, Phil, Hoeffding
    if args.cis == 'ttest':
        cis = ttestLB
    elif args.cis == 'Hoeffding':
        cis = HoeffdingLB
    elif args.cis == 'MPeB':
        cis = MPeBLB
    elif args.cis == 'Anderson':
        cis = AndersonLB
    elif args.cis == 'Phil':
        cis = PhilsAAAILB
    else:
        print("Wrong cis choice", args.cis)
        exit(0)

    if args.optimizer == 'CMA':
        optimizer = CMA
    elif args.optimizer == 'Powell':
        optimizer = Powell
    elif args.optimizer == 'BFGS':
        optimizer = BFGS
    elif args.optimizer == 'CEM':
        optimizer = CEM
    elif args.optimizer == 'CMAES':
        optimizer = CMAES
    else:
        print("Wrong optimizer choice", args.cis)
        exit(0)

    nWorkers = args.workers  # Workers is the number of threads running experiments in parallel

    print(f"Running experiments on {nWorkers} threads")

    # We will use different amounts of data, m. The different values of m will be stored in ms.
    # These values correspond to the horizontal axis locations in all three plots we will make.
    # We will use a logarithmic horizontal axis, so the amounts of data we use shouldn't be evenly spaced.
    ms = [32, 64, 128, 256, 512, 1024]
    numM = len(ms)

    # How many trials should we average over?
    numTrials = args.trials

    # ratio of test train split
    split_ratio = args.split_ratio

    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    # _ = ray.get(
    #     [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, env, gHats, deltas) for worker_id in
    #      range(1, nWorkers + 1)])
    run_experiments(0, nWorkers, ms, numM, numTrials, split_ratio, env, delta, fHat, cis, optimizer)
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time ellapsed: {time_parallel}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
