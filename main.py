import timeit
import sys
import os
import ray  # To allow us to execute experiments in parallel

from qsa import QSA

ray.init()
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'environments'))
sys.path.insert(1, os.path.join(os.path.dirname(sys.path[0]), 'optimizers'))

from environments.gridworld687 import Gridworld687
from environments.mountaincar import Mountaincar

from optimizers.optimizer_library import *
from estimators.is_estimates import *
from data.create_dataset import Dataset
from data.create_model import Model
from others.gHats import *
from bounds.confidence_intervals import *
# param1 = int(sys.argv[1])
param1 = 0
discrete = 1

bin_path = 'experiment_results/bin/'


# @ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest, env, gHats, deltas):
    # Results of the Seldonian algorithm runs
    seldonian_solutions_found = np.zeros((numTrials, numM))  # Stores whether a solution was found (1=True,0=False)
    seldonian_failures_g1 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1
    seldonian_failures_g2 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 2nd constraint, g_2
    seldonian_fs = np.zeros((numTrials, numM))  # Stores the primary objective values (fHat) if a solution was found

    # Results of the Least-Squares (LS) linear regression runs
    LS_solutions_found = np.ones((numTrials, numM))  # Stores whether a solution was found. These will all be true (=1)
    LS_failures_g1 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1
    LS_failures_g2 = np.zeros(
        (numTrials, numM))  # Stores whether solution was unsafe, (1=True,0=False), for the 2nd constraint, g_2
    LS_fs = np.zeros((numTrials, numM))  # Stores the primary objective values (f) if a solution was found

    # Prepares file where experiment results will be saved
    experiment_number = worker_id
    outputFile = bin_path + 'results%d.npz' % experiment_number
    print("Writing output to", outputFile)

    # Generate the data used to evaluate the primary objective and failure rates
    # np.random.seed((experiment_number + 1) * 9999)

    fHat = DR_hat
    print("importance sampling")
    # fHat = total_return



    for trial in range(numTrials):
        for (mIndex, m) in enumerate(ms):

            datasetGenerator = Dataset(m, env)
            theta = np.zeros((env.getStateDims(), env.getNumActions()))
            candidateDataset = datasetGenerator.generate_dataset(theta)
            if discrete:
                model = Model(env, candidateDataset, m, env.getNumDiscreteStates(), env.getNumActions(), env.horizonLength)
                candidateDataset = model.makeMLEModel()

            theta = np.zeros((env.getStateDims(), env.getNumActions()))
            datasetGenerator = Dataset(m, env)
            safetyDataset = datasetGenerator.generate_dataset(theta)
            if discrete:
                model = Model(env, safetyDataset, m, env.getNumDiscreteStates(), env.getNumActions(), env.horizonLength)
                safetyDataset = model.makeMLEModel()
            print("starting")

            # dataset, episodes, numStates, numActions, L

            # Generate the training data, D
            # base_seed = (experiment_number * numTrials) + 1
            # np.random.seed(base_seed + trial)  # done to obtain common random numbers for all values of m
            qsa = QSA(env, m, fHat, gHats, deltas, candidateDataset, safetyDataset)  # Run the Quasi-Seldonian algorithm

            # Get the candidate solution
            solution = qsa.getCandidateSolution()
            result, estimates = qsa.fHat(solution, safetyDataset, m, env)
            passedSafetyTest, lb = qsa.safety_test(estimates, m, delta=0.01, factor=1)



            # Run the Quasi-Seldonian algorithm
            # (result, passedSafetyTest) = QSA(trainX, trainY, gHats, deltas)
            if passedSafetyTest:
                seldonian_solutions_found[trial, mIndex] = 1
                trueEstimate, all_estimates = qsa.fHat(solution, safetyDataset, m,
                                                       env)  # Get the "true" mean squared error using the testData
                seldonian_failures_g1[
                    trial, mIndex] = 1 if trueEstimate < env.threshold else 0  # Check if the first behavioral constraint was violated
                # seldonian_failures_g2[
                # trial, mIndex] = 1 if trueEstimate < 1.25 else 0  # Check if the second behavioral constraint was violated
                seldonian_fs[trial, mIndex] = trueEstimate  # Store the "true" negative mean-squared error
                print(
                    f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] A solution was found: [{solution[0]:.10f}, {solution[1]:.10f}]\tfHat over test data: {trueEstimate:.10f}")
            else:
                seldonian_solutions_found[trial, mIndex] = 0  # A solution was not found
                seldonian_failures_g1[trial, mIndex] = 0  # Returning NSF means the first constraint was not violated
                # seldonian_failures_g2[trial, mIndex] = 0  # Returning NSF means the second constraint was not violated
                seldonian_fs[
                    trial, mIndex] = None  # This value should not be used later. We use None and later remove the None values
                print(
                    f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial + 1}/{numTrials}, m {m}] No solution found",lb)

            theta = np.zeros((env.getStateDims() * env.getNumActions()))
            optimizer = CMA(theta, qsa.objectiveWithoutConstraints)
            xMin = optimizer.run_optimizer()

            # # Run the Least Squares algorithm
            # theta = leastSq(trainX, trainY)  # Run least squares linear regression
            trueEstimate, totalEstimates = qsa.fHat(xMin, safetyDataset, m, env)  # Get the "true" mean squared error using the testData
            LS_failures_g1[
                trial, mIndex] = 1 if trueEstimate < env.threshold else 0  # Check if the first behavioral constraint was violated
            # LS_failures_g2[
            #     trial, mIndex] = 1 if trueEstimate < 1.25 else 0  # Check if the second behavioral constraint was violated
            LS_fs[trial, mIndex] = trueEstimate  # Store the "true" negative mean-squared error
            print(
                f"[(worker {worker_id}/{nWorkers}) LeastSq   trial {trial + 1}/{numTrials}, m {m}] LS fHat over test data: {trueEstimate:.10f}")
        print()

    np.savez(outputFile,
             ms=ms,
             seldonian_solutions_found=seldonian_solutions_found,
             seldonian_fs=seldonian_fs,
             seldonian_failures_g1=seldonian_failures_g1,
             seldonian_failures_g2=seldonian_failures_g2,
             LS_solutions_found=LS_solutions_found,
             LS_fs=LS_fs,
             LS_failures_g1=LS_failures_g1,
             LS_failures_g2=LS_failures_g2)


def main():
    env_map = {0: 'Mountaincar', 1: 'Gridworld', 2: 'Gridworldv2', 3: 'Cartpole'}
    env_choice = param1
    print("Running environment ", env_choice, " Name ", env_map[env_choice])
    if env_choice == 0:
        env = Mountaincar(discrete=True)
        gHats = [gHat1Mountaincar]
        deltas = [0.1]
    elif env_choice == 1:
        env = Gridworld687()
        gHats = [gHat1Gridworld]
        deltas = [0.1]
    elif env_choice == 2:
        env = Gridworldv2()
        gHats = [gHatGridworldv2]
        deltas = [0.1]
    elif env_choice == 3:
        env = Cartpole()
        gHats = [gHatCartpole]
        deltas = [0.1]
    else:
        print("Wrong environment choice")

    # Create the behavioral constraints: each is a gHat function and a confidence level delta
    # gHats = [gHat1, gHat2]
    # deltas = [0.1, 0.1]

    print("\nUsage: python main_plotting.py [number_threads]")
    print("       Assuming the default: 16")
    nWorkers = 4  # Workers is the number of threads running experiments in parallel

    print(f"Running experiments on {nWorkers} threads")

    # We will use different amounts of data, m. The different values of m will be stored in ms.
    # These values correspond to the horizontal axis locations in all three plots we will make.
    # We will use a logarithmic horizontal axis, so the amounts of data we use shouldn't be evenly spaced.
    # ms = [5,10,16,32, 128, 256, 512, 1024, 2048]  # ms = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ms = [512, 2048, 8192]
    numM = len(ms)

    # How many trials should we average over?
    numTrials = 5  # We pick 70 because with 70 trials per worker, and 16 workers, we get >1000 trials for each
    # value of m

    # How much data should we generate to compute the estimates of the primary objective and behavioral constraint
    # function values that we call "ground truth"? Each candidate solution deemed safe, and identified using limited
    # training data, will be evaluated over this large number of points to check whether it is really safe,
    # and to compute its "true" mean squared error.
    mTest = ms[-1] * 100

    # Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
    tic = timeit.default_timer()
    # _ = ray.get(
    #     [run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest, env, gHats, deltas) for worker_id in
    #      range(1, nWorkers + 1)])
    run_experiments(0, nWorkers, ms, numM, numTrials, mTest, env, gHats, deltas)
    toc = timeit.default_timer()
    time_parallel = toc - tic  # Elapsed time in seconds
    print(f"Time ellapsed: {time_parallel}")


if __name__ == "__main__":
    main()

