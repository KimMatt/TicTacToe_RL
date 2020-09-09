import sys
import numpy as np

from src.algos.agent import Agent
from src.algos.a2c import ActorCritic
from src.algos.ppo import PPO 
from src.algos.td import TD

algo_types_to_args = {'ppo': PPO, 'td': TD, 'ac':ActorCritic}

help_cmds = ["-h","--help"]

def argparse(args):
    """Parse arguments into a named tuple"""
    params = {}

    for i in range(0,len(args),2):
        print(args[i][:2])
        assert args[i][:2] == "--", print("You have to specify parameters with --[param_name] [value]")
        param_name = args[i][2:]
        value = np.float32(args[i+1])
        params[param_name] = value

    return params

if __name__ == "__main__":

    cmd = "-h" if len(sys.argv) < 2 else sys.argv[1]
    algo_types = list(algo_types_to_args.keys())

    if cmd in help_cmds or cmd not in algo_types:
        print("""Train a model for Tic Tac Toe with\n\n
        python -m train_model ALGO_TYPE [PARAMS...]\n\n
        where ALGO_TYPE is a valid type\n
        Valid types are: {}
        """.format(", ".join(algo_types)))
    else:
        params = argparse(sys.argv[2:])
        algo = algo_types_to_args.get(cmd)
        agent = algo(params)
        agent.train_model()
        agent.save_model()
