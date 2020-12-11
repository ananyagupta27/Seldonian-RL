import argparse


def parse_args():
    parser = argparse.ArgumentParser('Seldonian RL Library')
    parser.add_argument('--environment', type=int, default=4, help='env_map = {0: Mountaincar, 1: Gridworldv1, 2: Gridworldv2, 3: Gridworld687, 4: Cartpole}] [default: 0]')
    parser.add_argument('--delta', type=float, default=0.01,
                        help='permissible failure rate for confidence bound [default: 0.01]')
    parser.add_argument('--discrete', type=int, default=0, help='Whether or not to discretize states of the enviroment supported for all Gridworld versions and Mountaincar [default: 0]')
    parser.add_argument('--workers',  default=4, type=int, help='Number of workers [default: 4]')
    parser.add_argument('--trials', default=5, type=int, help='Number of trials [default: 5]')
    parser.add_argument('--split_ratio', default=0.3, type=float, help='Split Ratio: train/test [default: 0.1]')
    parser.add_argument('--is_estimator', default='PDIS', type=str, help='PDIS, IS, WIS, DR, DR_hat supported [default: PDIS]')
    parser.add_argument('--cis', default='ttest', help='ttest, Anderson, MPeB, Phil, Hoeffding supported [default: ttest]')
    parser.add_argument('--optimizer', default='CMA', help='Optimizers - Powell, CMA, CMAES, BFGS, CEM supported [default: CMA]')

    return parser.parse_args()
