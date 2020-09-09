import torch

from src.enviro.tictactoe import TicTacToe
from src.utils.logger import Logger

class Agent:

    def __init__(self, algo_type, params=None):
        pass

    def load_model(self, model_name):
        raise NotImplementedError

    def train_model(self):
        raise NotImplementedError

    def save_model(self):
        raise NotImplementedError

    def get_action(self, state, available_moves):
        raise NotImplementedError

    def test_model(self, baseline_agent, num_trials):
        # Return a percentage of wins, ties, and losses against the baseline agent
        logger = Logger()
        test_game = TicTacToe(logger, baseline_agent)
        for i in range(num_trials):
            reward = 0
            while reward == 0:
                action, _ = self.get_action(test_game.game_state, test_game.possible_moves)
                reward, _ = test_game.play_move(int(action))
            test_game.reset()
        return logger.agent_1_wins/logger.total, logger.agent_2_wins/logger.total, logger.ties/logger.total
