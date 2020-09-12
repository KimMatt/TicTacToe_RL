import numpy as np
from src.utils.logger import Logger

class InvalidMoveException(Exception):

    def __init__(self):
        pass

symmetric_transforms = [{0: 2, 1: 1, 2: 0, 3: 5, 4: 4, 5: 3, 6: 8, 7: 7, 8: 6},  # x mirror
                        {0: 6, 1: 7, 2: 8, 3: 3, 4: 4, 5: 5, 6: 0, 7: 1, 8: 2},  # y mirror
                        {0: 8, 1: 5, 2: 2, 3: 7, 4: 4, 5: 1, 6: 6, 7: 3, 8: 0},  # xy mirror
                        {0: 0, 1: 3, 2: 6, 3: 1, 4: 4, 5: 7, 6: 2, 7: 5, 8: 8},  # yx mirror
                        {0: 2, 1: 5, 2: 8, 3: 1, 4: 4, 5: 7, 6: 0, 7: 3, 8: 6},  # rotate 90
                        {0: 8, 1: 7, 2: 6, 3: 5, 4: 4, 5: 3, 6: 2, 7: 1, 8: 0},  # rotate 180
                        {0: 6, 1: 3, 2: 0, 3: 7, 4: 4, 5: 1, 6: 8, 7: 5, 8: 2}]  # rotate 270

forced_symmetric_mapping = {8:[0,2,6], 7:[1,3], 6:[0,2], 3:[1], 5:[1,3], 2:[0]}

symmetric_goals = [0,1,4]

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


    @staticmethod
    def check_symmetric(old_board, new_board):
        # Check if any of the symmetric transforms equate the
        # old and new boards
        new_board = np.array(new_board)
        for transform in symmetric_transforms:
            transformed_state = np.zeros(len(old_board))
            for key in transform.keys():
                transformed_state[transform[key]] = old_board[key]
            if np.array_equal(new_board, transformed_state):
                return True
        return False


    def transform_symmetric(self, last_move):
        if last_move not in symmetric_goals:
            # 1. get optimal positions to move to given last move
            symmetric_moves = forced_symmetric_mapping[last_move]
            # 2. while there are optimal positions...
            while len(symmetric_moves) > 0:
                # check if moving to optimal position maintains symmetry
                symmetric_move = symmetric_moves[0]
                if self.game_state[symmetric_move] == 0:
                    # 3. if it does then make the move and return
                    candidate_game_state = self.game_state[:]
                    candidate_game_state[symmetric_move] = self.game_state[last_move]
                    candidate_game_state[last_move] = 0
                    if self.check_symmetric(self.game_state, candidate_game_state):
                        self.possible_moves.remove(symmetric_move)
                        self.possible_moves.append(last_move)
                        self.game_state = candidate_game_state
                        break
                # 4. otherwise pop and repeat 2-4
                symmetric_moves.pop(0)
        # 5. when list is empty just keep the board the same


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
        self.transform_symmetric(move)
        # If it results in the end of a game
        if reward == 1 or reward == 0.5:
            return reward, self.game_state
        # Play opponent's move
        opponent_move, _ = self.opponent.get_action(self.game_state, self.possible_moves)
        opponent_reward = self._play_move(2, opponent_move)
        self.transform_symmetric(opponent_move)
        if opponent_reward == 0.5:
            return opponent_reward, self.game_state
        elif opponent_reward == 1:
            return -1, self.game_state
        return reward, self.game_state


    def reset(self):
        self.__init__(self.logger, self.opponent)

