import torch

from src.algos.a2c import ActorCritic
from src.algos.ppo import PPO 
from src.algos.td import TD

algo_types_to_args = {'ppo': PPO, 'td': TD, 'ac':ActorCritic}

class Agent:

    def __init__(self, algo_type, params=None):
        algo = algo_types_to_args.get(algo_type)
        self.agent = algo(params)

    def load_model(self, model_name):
        self.agent.load_model(model_name)

    def train_model(self):
        self.agent.train_model()

    def save_model(self):
        self.agent.save_model()

    def get_action(self, state, available_moves):
        return self.agent.get_action(state, available_moves)
