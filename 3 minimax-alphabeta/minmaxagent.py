import copy
import random

from exceptions import AgentException


class MinMaxAgent:
    def __init__(self, my_token='o', depth=2, heuristic=False):
        self.my_token = my_token
        self.search_depth = depth
        self.heuristic = heuristic

    def decide(self, connect4):
        if connect4.who_moves != self.my_token:
            raise AgentException('not my round')

        best_move = 0
        best_rating = -100
        moves = enumerate(connect4.possible_drops())
        for i, drop in moves:
            new_state = copy.deepcopy(connect4)
            new_state.drop_token(drop)
            score = self.minmax(new_state, False, self.search_depth)

            if score > best_rating:
                best_move = drop
                best_rating = score
        return best_move

    def rate_state(self, connect4):
        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins is None:
                return 0
            elif connect4.wins != self.my_token:
                return -1
        else:
            if self.heuristic:
                enemy_token = 'x' if self.my_token == 'o' else 'o'

                def rate_four(four):
                    my = 0
                    enemy = 0
                    empty = 0
                    for token in four:
                        if token == self.my_token:
                            my += 1
                        elif token == enemy_token:
                            enemy += 1
                        else:
                            empty += 1
                    return my, enemy, empty

                good_pos = 0
                all_positions = 0
                for four in connect4.iter_fours():
                    all_positions += 1
                    # if four == [self.my_token, self.my_token, self.my_token, '_'] or \
                    #         four == [self.my_token, self.my_token, '_', self.my_token]:
                    #     good_pos += 0.4
                    # elif four == [self.my_token, self.my_token, '_', '_']:
                    #     good_pos += 0.2
                    # elif four == [enemy_token, enemy_token, enemy_token, self.my_token] or \
                    #         four == [enemy_token, enemy_token, self.my_token, enemy_token]:
                    #     good_pos += 0.5
                    # elif four == [enemy_token, enemy_token, enemy_token, '_'] or \
                    #         four == [enemy_token, enemy_token, '_', enemy_token]:
                    #     good_pos -= 0.3
                    # elif four == [enemy_token, enemy_token, '_', '_']:
                    #     good_pos -= 0.1
                    my, enemy, empty = rate_four(four)
                    if my == 3 and empty == 1:
                        good_pos += 0.6
                    elif my == 2 and empty == 2:
                        good_pos += 0.2
                    elif enemy == 3 and my == 1:
                        good_pos += 0.5
                    elif enemy == 2 and my == 2:
                        good_pos += 0.3

                return good_pos / all_positions
            else:
                return 0

    def minmax(self, connect4, maximize, depth):
        if connect4.game_over or depth == 0:
            return self.rate_state(connect4)
        else:
            results = []
            for move in connect4.possible_drops():
                new_state = copy.deepcopy(connect4)
                new_state.drop_token(move)
                results.append(self.minmax(new_state, not maximize, depth - 1))

            if maximize:
                return max(results)
            else:
                return min(results)
