import torch

from src.algos.ac import ActorCritic
from src.algos.ppo import PPO 
from src.algos.td import TD

algos = {'ppo': PPO, 'td': TD, 'ac':ActorCritic}

class Agent:

    def __init__(self, algo_type, params=None):
        algo = algo_type_to_imp.get(algo_type)
        self.agent = algo(params)

    def load_model(self, model_name):
        self.agent.load_model(model_name)

    def train_model(self):
        self.agent.train_model()

    def get_action(self, state):
        return self.agent.get_action(state)
