from agents.neural_network import *

from sklearn.model_selection import train_test_split
import numpy as np
import cma
from scipy.optimize import minimize
from scipy import stats
from data.create_dataset import *


"""
This is responsible for training and testing when optimization is done using lagrange mulitplier method
Here the experiments were performed on a given dataset
Note : Learning rates are sensitive for optimization and need tuning
The function approximation that maps states to actions is feed forward neural network
"""


STATES = 18
ACTIONS = 4
GAMMA = 0.95
BENCHMARK_PERFORMANCE = 1.41537
DUMMY = 0




def get_PDIS_estimate(dataset, out):
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']

    is_estimates = torch.zeros(states.size()[0])

    for episode in range(len(states)):
        is_current = torch.tensor(0).float()
        frac = torch.tensor(1).float()
        for timestep in range(len(states[episode])):

            s = states[episode][timestep]
            if s[0] == DUMMY:
                break
            a = actions[episode][timestep]
            frac *= out[episode][timestep][a] / pi_b[episode][timestep]
            is_current += (GAMMA ** timestep) * (rewards[episode][timestep] * frac)
        is_estimates[episode] = is_current
    return is_estimates


def lower_bound(is_estimates, factor=2, delta=0.01):
    size = is_estimates.size()[0]
    lb = torch.mean(is_estimates) - factor * (
            torch.std(is_estimates) / np.sqrt(size)) * stats.t.ppf(1 - delta, size - 1)
    return lb


# getting the value of the total objective function at the current values of theta and lambda
def lfn(out, l, dataset):
    list_estimates = (get_PDIS_estimate(dataset, out))
    mean_estimate = -torch.mean(list_estimates) + l * (BENCHMARK_PERFORMANCE - lower_bound(list_estimates))
    return mean_estimate


def train(net):
    horizon_length = 186
    batch_size = 64
    # using Adam optimizer to solve the minimization w.r.t policy parameter
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    # langrange multiplier
    l = torch.tensor(0.0, requires_grad=True)

    # training loop
    for epoch in range(5):
        epoch_loss = 0
        c = 0
        for states, actions, rewards, pi_b in dataloader:
            episodes = states.size()[0]
            optimizer.zero_grad()

            # forward pass to get the action probabilities from the states
            out = net(states.float())

            # calculate the objective function
            mean_estimate = lfn(out, l, {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b})

            # backprop w.r.t. policy parameter gradient descent since minimization w.r.t. theta
            mean_estimate.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += mean_estimate.item()
            c += 1

            # calculating the gradient w.r.t. lambda
            _, dl = torch.autograd.grad(mean_estimate, [out, l])

            # gradient ascent w.r.t. lambda since maximization w.r.t. lambda
            with torch.no_grad():
                l += 1e-5 * dl
        #         print("l is ", l)
        print("epoch loss----->", epoch_loss / c)
    return l


def test(net, dataloader, l):
    c = 0
    net.eval()
    for states, actions, rewards, pi_b in dataloader:
        with torch.no_grad():
            out = net(states.float())
            mean_estimate = lfn(out, l, {'states': states, 'actions': actions, 'rewards': rewards, 'pi_b': pi_b})
            print(mean_estimate)


if __name__ == '__main__':
    datasetCreator = Dataset()
    dataset = datasetCreator.get_dataset_from_file('data.csv')
    states = dataset['states']
    actions = dataset['actions']
    rewards = dataset['rewards']
    pi_b = dataset['pi_b']
    states_train, states_test, actions_train, actions_test, rewards_train, rewards_test, pi_b_train, pi_b_test = \
        train_test_split(states, actions, rewards, pi_b, test_size=0.5, random_state=42)

    dataset_test = {'states': states_test, 'actions': actions_test, 'rewards': rewards_test, 'pi_b': pi_b_test}
    dataset_test_size = len(states_test)

    from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler

    tensor_rewards = torch.tensor(rewards_train)
    tensor_states = torch.tensor(states_train)
    tensor_pib = torch.tensor(pi_b_train)
    tensor_actions = torch.tensor(actions_train)

    bs = 64
    dataset = TensorDataset(tensor_states, tensor_actions, tensor_rewards, tensor_pib)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=bs)
    net = Net(1, 4)
    print(net)

    lagrange = train(net)

    tensor_rewards = torch.tensor(rewards_test)
    tensor_states = torch.tensor(states_test)
    tensor_pib = torch.tensor(pi_b_test)
    tensor_actions = torch.tensor(actions_test)
    from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler

    bs = 64
    dataset = TensorDataset(tensor_states, tensor_actions, tensor_rewards, tensor_pib)
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=bs)

    test(net, dataloader, lagrange)
