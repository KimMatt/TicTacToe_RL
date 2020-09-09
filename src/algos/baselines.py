import numpy as np

from src.enviro.tictactoe import TicTacToe

class RandomAgent:

    def __init__(self):
        pass

    def get_action(self, state, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))], None


class ExpertAgent:

    def __init__(self):
        self.player_n = 2


    def winning_moves(self, player_n, state, possible_moves):
        """Returns list of moves [int] that would result
        in player_n winning if they took that position.
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of winning moves
        """

        moves = []

        for move in possible_moves:
            next_state = state[:]
            next_state[move] = player_n
            if TicTacToe.winning_state(next_state, player_n, move):
                moves.append(move)

        return moves


    def fork_moves(self, player_n, state, possible_moves):
        """Returns list of moves [int] where role has
        two opportunities to win (two non-blocked lines of 2) if
        they took that position.
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of fork moves
        """
        moves = []
        possible_moves = possible_moves
        # Note: This is used to test different positions so it may not be role's
        # actual turn so role-checking is turned off
        for move in possible_moves:
            next_state = state
            next_state[move] = player_n
            remaining_moves = possible_moves[:]
            remaining_moves.remove(move)
            winning_count = 0
            for move_2 in remaining_moves:
                test_state = next_state[:]
                test_state[move_2] = player_n
                if TicTacToe.winning_state(test_state, player_n, move_2):
                    winning_count += 1
            if winning_count >= 2:
                moves.append(move)
        return moves


    def opposite_corners(self, player_n, opponent_n, state):
        """Returns list of moves [int] opposite to an opponent's corner
        Args:
            player_n (int): Player to check for
        Returns:
            moves (list): List of opposite corner moves
        """
        moves = []
        opposite_corners = {0: 8, 2: 6, 8: 0, 6: 2}
        for k, v in opposite_corners.items():
            if state[k] == opponent_n and state[v] == 0:
                moves.append(v)
        return moves

    def get_action(self, state, possible_moves):

        corners = [0,2,6,8]
        center = 4

        opponent_n = 1 if self.player_n == 2 else 2

        winning_positions = self.winning_moves(self.player_n, state, possible_moves)
        blocking_positions = self.winning_moves(opponent_n, state, possible_moves)
        fork_positions = self.fork_moves(self.player_n, state, possible_moves)
        opponent_forks = self.fork_moves(opponent_n, state, possible_moves)
        opposite_corners = self.opposite_corners(self.player_n, opponent_n, state)
        available_corners = list(set(corners).intersection(set(possible_moves)))

        if len(possible_moves) == 9:
            # 1. If first move of the game, play a corner or center
            corners_and_center = corners + [center]
            return corners_and_center[random.randint(0, 4)], None
        if winning_positions:
            # 2. Check for winning moves
            return winning_positions[random.randint(0,len(winning_positions)-1)], None
        if blocking_positions:
            # 3. Check for blocking moves
            return blocking_positions[random.randint(0,len(blocking_positions)-1)], None
        if fork_positions:
            # 4. Check for fork positions
            return fork_positions[random.randint(0, len(fork_positions)-1)], None
        if opponent_forks:
            # 5. Prevent opponent from using a fork position
            return opponent_forks[random.randint(0, len(opponent_forks)-1)], None
        if center in available_moves:
            # 6. Try to play center
            return center, None
        if opposite_corners:
            # 7. Try to play a corner opposite to opponent
            return opposite_corners[random.randint(0, len(opposite_corners)-1)], None
        if available_corners:
            # 8. Try to play any corner
            return available_corners[random.randint(0, len(available_corners)-1)], None
        # 9. Play anywhere else - i.e. a middle position on a side
        return available_moves[random.randint(0,len(available_moves)-1)], None
