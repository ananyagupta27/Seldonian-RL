import numpy as np
from helper import *

class Model:

    def __init__(self, env, dataset, episodes, numStates, numActions, L):
        self.env = env
        self.dataset = dataset
        self.numStates = numStates - 1
        self.numActions = numActions
        self.L = L
        self.episodes = episodes
        self.pi = normalize(np.ones((env.getNumDiscreteStates(), env.getNumActions())))

    def makeMLEModel(self):
        states = self.dataset['states']
        actions = self.dataset['actions']
        rewards = self.dataset['rewards']
        pi_b = self.dataset['pi_b']

        # Set everything to zero
        stateActionCounts = np.zeros((self.numStates, self.numActions))
        stateActionStateCounts = np.zeros((self.numStates, self.numActions, self.numStates + 1))

        p = np.zeros((self.numStates, self.numActions, self.numStates + 1))
        R = np.zeros((self.numStates, self.numActions, self.numStates + 1))
        d0 = np.zeros(self.numStates)

        # Compute all of the counts and set R to the sum of rewards from [s][a][sprime] transitions
        for i in range(self.episodes):
            trajectoryLength = len(states[i])
            for j in range(trajectoryLength):
                # s = np.argmax(states[i][j])
                s = self.env.getDiscreteState(states[i][j])
                a = actions[i][j]
                r = rewards[i][j]
                if j == trajectoryLength - 1:
                    sPrime = self.numStates
                    # if sPrime ==0 :
                    #     print("s=",s,"a=",a,"sp",sPrime)
                else:
                    # sPrime = np.argmax(states[i][j + 1])
                    sPrime = self.env.getDiscreteState(states[i][j + 1])
                # print("state is", s, "action is", a, "sprime is", sPrime)

                if j != trajectoryLength - 1:
                    stateActionStateCounts[s][a][sPrime] += 1
                    stateActionCounts[s][a] += 1
                else:
                    if sPrime != self.numStates:
                        print("sPrime not equal to numStates: Error ", sPrime)
                        exit()

                R[s][a][sPrime] += r

        for i in range(self.episodes):
            # d0[np.argmax(states[i][0])] += 1.0 / self.episodes
            d0[self.env.getDiscreteState(states[i][0])] += 1.0 / self.episodes

        rMin = rewards[0][0]
        for i in range(self.episodes):
            trajectoryLength = len(states[i])
            for j in range(trajectoryLength):
                rMin = min(rewards[i][j], rMin)

        # Compute P and R
        print("p and R")
        for s in range(self.numStates):
            for a in range(self.numActions):
                for sPrime in range(self.numStates + 1):
                    if stateActionCounts[s][a] == 0:
                        p[s][a][sPrime] = 1 if sPrime == self.numStates else 0
                    else:
                        p[s][a][sPrime] = stateActionStateCounts[s][a][sPrime] / stateActionCounts[s][a]

                    if stateActionStateCounts[s][a][sPrime] == 0:
                        R[s][a][sPrime] = 0
                    else:
                        R[s][a][sPrime] /= stateActionStateCounts[s][a][sPrime]
                    # print("s ", s, " a ", a, " sprime ", sPrime, "P", p[s][a][sPrime], " R ", R[s][a][sPrime])
        Q, V = self.calculateValueFunctions(p, R, self.env)
        trajectories = {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b, 'p': p, 'R': R,
                        'd0': d0, 'V':V, 'Q':Q}
        print("done modeling")
        return trajectories

    def calculateValueFunctions(self, p, R, env):
        pi = self.pi
        numStates = env.getNumDiscreteStates()
        numActions = env.getNumActions()
        actionProbabilities = np.zeros((numStates, numActions))
        L = env.horizonLength

        numStates -= 1

        for s in range(numStates):
            actionProbabilities[s] = pi[s]

        Q = np.zeros((L, numStates + 1, numActions))
        V = np.zeros((L, numStates + 1))
        print(L, "L", numStates, numActions)

        for t in range(L - 1, -1, -1):
            for s in range(numStates):
                for a in range(numActions):
                    for sPrime in range(numStates + 1):
                        Q[t][s][a] += p[s][a][sPrime] * R[s][a][sPrime]
                        if sPrime != numStates and t != L - 1:
                            Q[t][s][a] += p[s][a][sPrime] * np.dot(actionProbabilities[sPrime], (Q[t + 1][sPrime]))

        for t in range(L):
            for s in range(numStates):
                V[t][s] = np.dot(actionProbabilities[s], (Q[t][s]))

        # for s in range(numStates):
        #     print("state:", s, " value func:", V[0][s])
        #
        # for s in range(numStates):
        #     for a in range(4):
        #         print("action q function", s, " ", a, " ", Q[0][s][a])
        print("calculated value functions")
        return Q, V