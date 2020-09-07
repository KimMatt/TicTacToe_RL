from os import path
import os

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import scipy.signal

from src.enviro.tictactoe import TicTacToe
from src.utils.mlp import construct_mlp
from src.utils.logger import Logger
from src.utils.utils import cum_sum


class PPO:

    def _initialize_buffers(self):
        # create the buffers for trajectory values
        self.advantages = torch.zeros(self.minibatch_size, dtype=torch.float)
        self.actions = torch.zeros(self.minibatch_size, dtype=torch.float)
        self.values = torch.zeros(self.minibatch_size, dtype=torch.float)
        self.rewards = torch.zeros(self.minibatch_size, dtype=torch.float)
        self.states = torch.zeros(
            (self.minibatch_size, self.state_size), dtype=torch.float)
        self.logps = torch.zeros(self.minibatch_size, dtype=torch.float)


    def __init__(self, params):
        # setup device for pytorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # default values for params
        self.params = {'gamma': 0.99, 'lam': 0.96, 'epochs': 10, 'minibatch_size':65,
              'pi_lr': 3e-4, 'v_lr': 1e-3, 'clip':0.1, 'target_kl': 0.01,
              'pi_train_iters': 60, 'v_train_iters': 60}
        # parse the parameters
        for param in params.keys():
            assert param in self.params.keys(), """Invalid parameter specified
            (--{})\nValid params: [{}]""".format(param, ", ".join(list(self.params.keys())))
            self.params[param] = params[param]
        # assign them locally
        for param in self.params.keys():
            exec('self.{}={}'.format(param, self.params[param]))

        # tic tac toe has 9 squares :)
        self.state_size = 9

        # create the buffers for trajectory values
        self._initialize_buffers()

        # initialize value and pi approximations
        self.v = construct_mlp((self.state_size, 64, 64, 1), nn.Tanh).to(self.device)
        self.pi = construct_mlp((self.state_size, 64, 64, self.state_size), nn.Tanh, nn.Softmax).to(self.device)


    def _populate_buffers(self, ep_start, ep_end, final_reward):
        values = torch.zeros((ep_end - ep_start) + 1, dtype=torch.float)
        values[0:(ep_end-ep_start)] = self.values[ep_start:ep_end]
        values[-1] = final_reward
        deltas = self.rewards[ep_start:ep_end] + (self.gamma * self.values[ep_start+1:ep_end+1]) - self.values[ep_start:ep_end]
        print(deltas)
        self.advantages[ep_start:ep_end] = cum_sum(deltas, self.gamma * self.lam)
        self.rewards[ep_start:ep_end] = cum_sum(self.rewards[ep_start:ep_end], self.gamma)


    def _produce_trajectories(self):

        self._initialize_buffers()
        logger = Logger()
        tictactoe = TicTacToe(logger, self)
        state = tictactoe.game_state[:]
        self.states[0] = torch.tensor(state, dtype=torch.float)

        t = 0
        ep_start = 0

        for t in range(self.minibatch_size):
            if state is None:
                state = tictactoe.game_state[:]
            self.states[t] = torch.tensor(state, dtype=torch.float)
            self.actions[t], self.logps[t] = self.get_action(state, tictactoe.possible_moves)
            self.values[t] = self.v(torch.tensor(state, dtype=torch.float).to(self.device))
            self.rewards[t], next_state = tictactoe.play_move(int(self.actions[t]))
            print("reward, actions", self.rewards[t], self.actions[t])
            print(t)
            # episode over
            if self.rewards[t].item() != 0.0:
                tictactoe.reset()
                print("tic tac toe reset", tictactoe.game_state, tictactoe.possible_moves)
                print("inside", ep_start, t)
                self._populate_buffers(ep_start, t, 0)
                ep_start = t + 1
            state = next_state
        # if the episode did not end on the last step
        if self.rewards[-1].item() == 0.0:
            final_reward = self.v(
                torch.tensor(state, dtype=torch.float).to(self.device))
            print("end of epoch", ep_start, self.minibatch_size-1)
            self._populate_buffers(ep_start, self.minibatch_size-1, final_reward)


    def _update(self):

        pi_optim = Adam(self.pi.parameters(), lr=self.pi_lr)
        v_optim = Adam(self.v.parameters(), lr=self.v_lr)

        # update pi values
        for i in range(self.pi_train_iters):
            pi_optim.zero_grad()

            new_logps = self._get_logps(self.states, self.actions)
            ratios = torch.exp(new_logps - self.logps)
            clipped_adv = torch.clamp(ratios, 1+self.clip, 1-self.clip) * self.advantages
            loss = -(torch.min(clipped_adv, self.advantages * ratios)).mean()

            kl = (self.logps - new_logps).mean().item()
            if kl > 1.5 * self.target_kl:
                print("Early stopping because target kl reached {}".format(kl))
                break

            loss.backward(retain_graph=True)
            pi_optim.step()

        # update v values
        for i in range(self.v_train_iters):
            v_optim.zero_grad()

            loss = ((self.values - self.rewards)**2).mean()
            loss.backward(retain_graph=True)

            v_optim.step()


    def train_model(self):
        for epoch in range(self.epochs):
            print("epoch: {}".format(epoch))
            self._produce_trajectories()
            self._update()


    def _get_logps(self, states, actions):
        pis = self.pi(torch.tensor(states, dtype=torch.float).to(self.device))
        return torch.tensor([torch.log(pis[i][int(action)]) for i, action in enumerate(actions)], dtype=torch.float)


    def get_action(self, state, possible_moves):
        # get the pi values
        pi_vals = self.pi(torch.tensor(state, dtype=torch.float).to(self.device))
        # get the sum of non available moves pi values
        distribute_sum = np.sum([pi_vals[i].item() for i in range(self.state_size) if i+1 not in possible_moves])
        addition_sum = distribute_sum / len(possible_moves)
        roll = np.random.random()
        cum_sum = 0
        # choose move from possible moves stochastically
        for move in possible_moves:
            cum_sum += pi_vals[move-1] + addition_sum
            if roll <= cum_sum:
                return move, torch.log(pi_vals[move-1])
        print(possible_moves)
        print(pi_vals)
        return possible_moves[-1], torch.log(pi_vals[possible_moves[-1]-1])


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
        if not path.exists(model_path):
            os.mkdir(model_path)
        model_name = "_".join(["{}={}".format(param, self.params[param]) for param in self.params.keys()])

        torch.save(self.v.state_dict, model_path + "v_" + model_name)
        torch.save(self.pi.state_dict, model_path + "pi_" + model_name)
