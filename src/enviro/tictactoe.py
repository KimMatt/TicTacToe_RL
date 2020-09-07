import numpy as np
from src.utils.logger import Logger

class InvalidMoveException(Exception):

    def __init__(self):
        pass


class TicTacToe:

    in_progress = False
    game_state = None
    logger = None
    winner = None
    possible_moves = None
    moves = None

    def __init__(self, logger, opponent):
        """Opponent must have get_action(state, available_moves) -> move, logp implemented"""
        self.moves = 0
        self.logger = logger
        self.in_progress = True
        self.opponent = opponent
        # 0 for blank, 1 for x, 2 for o
        self.game_state = [0 for i in range(0, 9)]
        self.possible_moves = [0,1,2,3,4,5,6,7,8]


    @staticmethod
    def winning_state(game_state, player_sign, space):
        # Check if the space + player sign is a winning move in given state
        rows = [[1, 4, 7], [0, 3, 6], [2, 5, 8], # verticals
                [0, 1, 2], [3, 4, 5], [6, 7, 8], # horizontals
                [0, 4, 8], [2, 4, 6]] # diagonals
        for row in rows:
            if space in row:
                if all([game_state[each] == player_sign for each in row]):
                    return True
        return False


    def _play_move(self, player_n, move):
        # Plays the given move by player in given space
        # Returns whether the move is a winning move or not.
        if self.game_state[move] == 0:
            self.possible_moves.remove(move)
            self.game_state[move] = player_n
            if TicTacToe.winning_state(self.game_state, player_n, move):
                self.in_progress = False
                self.logger.log_agent_win(player_n)
                self.winner = player_n
                return 1
            elif not self.possible_moves:
                self.in_progress = False
                self.logger.log_tie()
                return 0.5
            self.moves += 1
            return 0
        else:
            raise InvalidMoveException


    def play_move(self, move):
        # Play player move
        reward = self._play_move(1, move)
        # If it results in the end of a game
        if reward == 1 or reward == 0.5:
            return reward, self.game_state
        # Play opponent's move
        opponent_move, _ = self.opponent.get_action(self.game_state, self.possible_moves)
        opponent_reward = self._play_move(2, opponent_move)
        if opponent_reward == 0.5:
            return opponent_reward, self.game_state
        elif opponent_reward == 1:
            return -1, self.game_state
        return reward, self.game_state


    def reset(self):
        self.__init__(self.logger, self.opponent)
