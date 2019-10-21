import argparse
import torch


def my_get_args(string=None):
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', default='ppo',
                        help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use-gae', action='store_true', default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--num-processes', type=int, default=256,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--num-val-processes', type=int, default=1280,
                        help='how many validation CPU processes to use (default: 16)')
    parser.add_argument('--num-steps', type=int, default=100,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo-epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--num-mini-batch', type=int, default=4,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip-param', type=float, default=0.1,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=5000,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--eval-interval', type=int, default=20,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--vis-interval', type=int, default=20,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--num-env-steps', type=int, default=25600000000,
                        help='number of environment steps to train (default: 10e6)')
    parser.add_argument('--env-name', default='test',
                        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--log-dir', default='logs',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--save-dir', default='./checkpoints',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--snapshot', default=None,
                        help='filename of snapshot to start from')    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--use-linear-clip-decay', action='store_true', default=False,
                        help='use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--port', type=int, default=8097,
                        help='port to run the server on (default: 8097)')

    parser.add_argument('--sim-sigma', type=float, default=0.03,
                        help='SIM noise (0.03)')
    parser.add_argument('--sim-momentum', type=float, default=0.9,
                        help='SIM momentum (0.9)')
    parser.add_argument('--sim-bins', type=int, default=3,
                        help='SIM pump bins (3)')
    parser.add_argument('--sim-rshift', type=float, default=1.0,
                        help='SIM reward shift for `cut` option (1.0)')
    parser.add_argument('--sim-scale', type=float, default=0.04,
                        help='SIM pump scale (0.04)')
    parser.add_argument('--sim-reward', default='rank',
                        help='SIM Reward kind: cut / (rank) / rank_orig')
    parser.add_argument('--sim-continuous', action='store_true', default=False,
                        help='Continuous actions in SIM')
    parser.add_argument('--sim-span', type=float, default=3.0,
                        help='SIM pump span for continuous actions (3.0)')
    parser.add_argument('--sim-percentile', type=float, default=99.0,
                        help='SIM ranked rewards percentile (99)')
    parser.add_argument('--sim-perc-len', type=int, default=100,
                        help='SIM: number of last runs to calculate percentiles (100)')
    parser.add_argument('--sim-no-linear', action='store_true', default=False,
                        help='SIM: remove linear pump baseline')
    parser.add_argument('--sim-start', type=float, default=1.0,
                        help='SIM start pump (1.0)')
    parser.add_argument('--sim-no-static', action='store_true', default=False,
                        help='SIM: remove static features')
    parser.add_argument('--sim-no-extra', action='store_true', default=False,
                        help='SIM: remove extra features')

    parser.add_argument('--sim-keep', type=int, default=5,
                        help='Number of runs after which to resample graph')
    parser.add_argument('--sim-nsim', type=int, default=1,
                        help='Number of graphs to keep simultaneously')
    parser.add_argument('--hidden-size', type=int, default=256,
                        help='Neural network hidden size')

    if string is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args([e for e in string.strip().split() if len(e)>0])

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args