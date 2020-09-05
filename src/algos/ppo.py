from os import path
import os

import torch
import torch.nn as nn
from torch.optim import Adam
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
        # setup device for pytorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        # tic tac toe has 9 squares :)
        state_size = 9
        action_size = 1

        # create the buffers for trajectory values
        self.initialize_buffers()

        # initialize value and pi approximations 
        self.v = construct_mlp((state_size, 64, 64, 1), nn.Tanh)
        self.pi = construct_mlp((state_size, 64, 64, state_size), nn.Tanh, nn.SoftMax)


    def _produce_trajectories(self):

        self.initialize_buffers()

        logger = Logger
        tictactoe = TicTacToe(logger)
        state = tictactoe.game_state[:]
        self.states[0] = state

        t = 0
        ep_start = 0

        for t in range(len(self.minibatch_size)):
            if state is None:
                state = tictactoe.game_state[:]
            self.states[t] = state
            self.actions[t], self.logps[t] = self.get_action(state)
            self.values[t] = self.v(state)
            next_state, self.rewards[t] = tictactoe.play_move()
            # episode over
            if next_state is None:
                tictactoe.reset()
                ep_start = t + 1
                self.populate_buffers(ep_start, t, 0)
            state = next_state
        final_reward = 0 if state is None else self.v(state)
        self.populate_buffers(ep_start, len(self.minibatch_size)-1, final_reward)



    def _update(self):

        pi_optim = Adam(self.pi.parameters(), lr=self.pi_lr)
        v_optim = Adam(self.v.parameters(), lr=self.v_lr)

        # update pi values
        for i in range(self.pi_train_iters):
            pi_optim.zero_grad()

            new_logps = self.get_logps(self.states, self.actions)
            ratios = torch.exp(new_logps - self.logps)
            clipped_adv = torch.clamp(ratios, 1+self.clip, 1-self.clip) * self.advantages
            loss = -(torch.min(clipped_adv, self.advantages * ratios)).mean()

            kl = (self.logps - new_logps).mean().item()
            if kl > 1.5 * self.target_kl:
                print("Early stopping because target kl reached {}".format(kl))
                break

            loss.backwards()
            pi_optim.step()

        # update v values
        for i in range(self.v_train_iters):
            v_optim.zero_grad()

            loss = ((self.values - self.rewards)**2).mean()
            loss.backwards()

            v_optim.step()


    def train_model(self):
        for epoch in self.epochs:
            self._produce_trajectories()
            self._update()

    def get_logps(self, states, actions):
        pis = self.pi(states)
        return np.array([pis[i][action] for i, action in enumerate(actions)])


    def get_action(self, state):
        # .2 .3 .5
        pi_vals = self.pi(state)
        roll = np.random.random()
        cum_sum = 0
        for i in range(len(pi_vals)):
            cum_sum += pi_vals[i]
            if cum_sum >= roll:
                return i, pi_vals[i]
        return len(pi_vals) - 1, pi_vals[-1]


    def load_model(self, path):
        # assumes the path leads to the model's saved pi configurations
        v_path = path.split("_")
        v_path = v_path[1:].insert(0,"v")
        v_path = "_".join(v_path)

        if path.exists(v_path) and path.exists(path):
            self.v.load_state_dict(torch.load(v_path, map_location=self.device))
            self.pi.load_state_dict(torch.load(path, map_location=self.device))
        else:
            raise Exception('Couldn\'t load from paths {} and {}'.format(path, v_path))

    def save_model(self):
        model_path = './models/'
        if !(path.exists(model_path)):
            os.mkdir(model_path)
        model_name = "_".join(["{}={}".format(param, self.params[param]) for param in self.params.keys()])

        torch.save(self.v.state_dict, model_path + "v_" + model_name)
        torch.save(self.pi.state_dict, model_path + "pi_" + model_name)
