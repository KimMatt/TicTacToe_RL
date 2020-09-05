import torch
import torch.nn as nn
import numpy as np

from src.enviro.tictactoe import TicTacToe
from src.utils.mlp import construct_mlp
from src.utils.logger import Logger


class PPO:

    def initialize_buffers(self):
        # create the buffers for trajectory values
        self.advantages = np.zeros(self.minibatch_size, dtype=np.float32)
        self.actions = np.zeros(self.minibatch_size, dtype=np.float32)
        self.values = np.zeros(self.minibatch_size, dtype=np.float32)
        self.rewards = np.zeros(self.minibatch_size, dtype=np.float32)
        self.states = np.zeros((self.minibatch_size, state_size), dtype=np.float32)
        self.logps = np.zeros(self.minibatch_size, dtype=np.float32)

    def __init__(self, params):
        # default values for params
        self.params = {'gamma': 0.99, 'lambda': 0.96, 'epochs': 10, 'minibatch_size':65, 
              'pi_lr': 3e-4, 'v_lr': 1e-3, 'clip':0.1, 'target_kl': 0.01,
              'pi_train_iters': 60, 'v_train_iters': 60}
        # parse the parameters 
        for param in params.keys():
            assert param in self.params.keys(), """Invalid parameter specified 
            (--{})\nValid params: [{}]""".format(param, ", ".join(list(self.params.keys())))
            self.params[param] = params[param]
        # assign them locally 
        for param in self.params.keys()
            exec('self.{}={}'.format(param, self.params[param]))
        del self.params

        # tic tac toe has 9 squares :)
        state_size = 9
        action_size = 1

        # create the buffers for trajectory values
        self.initialize_buffers()

        # initialize value and pi approximations 
        self.v = construct_mlp((state_size, 64, 64, 1), nn.Tanh)
        self.pi = construct_mlp((state_size + action_size, 64, 64, 1), nn.Tanh)


    def _get_trajectories(self):
        self.initialize_buffers()
        logger = Logger
        tictactoe = TicTacToe(logger)
        state = tictactoe.game_state[:]
        for t in range(len(self.minibatch_size)):
            action, logp = self.get_action(state)
            next_state, reward = tictactoe.play_move()


    def _update(self):
        pass


    def train_model(self):
        pass


    def get_action(self, state):
        p =  self.pi(state)
        pass


    def load_model(self):
        pass
