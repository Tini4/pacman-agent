# coding=utf-8
"""My team"""

import random
from time import perf_counter
from typing import List, Callable, Tuple, Set

import contest.util as util  # type: ignore
from contest.capture import GameState  # type: ignore
from contest.capture_agents import CaptureAgent  # type: ignore
from contest.game import Directions, GameStateData, Grid, AgentState, Configuration  # type: ignore
from contest.layout import Layout  # type: ignore
from contest.util import nearest_point  # type: ignore


class Timer:
    """
    A context manager and decorator for timing the execution of blocks of code or functions.
    """

    def __init__(self, name: str = 'Timer'):
        """Initializes the Timer object with a name."""

        self.name = name

    def __enter__(self) -> 'Timer':
        """Starts the timer."""

        self._ts = perf_counter()

        return self

    def __exit__(self, *_exc) -> None:
        """Stops the timer and prints the elapsed time."""

        te = perf_counter()

        print(f'{self.name}{" " if self.name else ""}took {te - self._ts:.3f} seconds')

    def __call__(self, f: Callable) -> Callable:
        """Decorates a function to measure its execution time."""

        def wrapper(*args, **kwargs) -> Callable:
            """Wrapper function."""
            with self:
                return f(*args, **kwargs)

        return wrapper


def create_team(first_index: int, second_index: int, is_red: bool, **_kwargs) -> List[CaptureAgent]:
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers. isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    gb = GameBoard(is_red)

    return [TiTAgent(first_index, gb), TiTAgent(second_index, gb)]


class GameBoard:
    """Game board class"""

    def __init__(self, is_red: bool) -> None:
        """Initializes the game board"""
        self.DISTS: List[List[List[List[int]]]] = []
        self.player_positions: List[List[Set[int]]] = []
        self.my_index: Tuple[int, int] = (-1, -1)
        self.enemy_index: Tuple[int, int] = (-1, -1)

        self.is_red = is_red

    @Timer('setup')
    def setup(self, gs: GameState) -> None:
        """Game board setup"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        layout: Layout = data.layout
        walls: Grid = layout.walls

        if self.is_red:
            self.my_index = tuple(gs.red_team)
            self.enemy_index = tuple(gs.blue_team)
        else:
            self.my_index = tuple(gs.blue_team)
            self.enemy_index = tuple(gs.red_team)

        self.DISTS = self.floyd_warshall(walls)

        self.player_positions = [[set() for _ in range(layout.height)] for _ in range(layout.width)]
        for i in range(4):
            p = agent_states[i].start.pos
            self.player_positions[p[0]][p[1]].add(i)

    @Timer('Floyd-Warshall')
    def floyd_warshall(self, walls: Grid) -> List[List[List[List[int]]]]:
        """Floyd-Warshall algorithm"""
        width = walls.width
        height = walls.height
        grid = walls.data

        N: int = width * height
        INF: int = 10 ** 9

        def idx(_x: int, _y: int) -> int:
            """Helper to translate coordinates to linear index"""
            return _y * width + _x

        # ----------------------------
        # STEP 1: Build flat N×N matrix
        # ----------------------------

        dist = [[INF] * N for _ in range(N)]

        for y in range(height):
            for x in range(width):
                if grid[x][y]:
                    continue
                i = idx(x, y)
                dist[i][i] = 0

                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and not grid[nx][ny]:
                        dist[i][idx(nx, ny)] = 1

        # ----------------------------------
        # STEP 2: Optimized Floyd–Warshall
        # ----------------------------------

        for k in range(N):
            dk = dist[k]
            for i in range(N):
                di = dist[i]
                via = di[k]
                if via == INF:
                    continue

                # Use local var lookup (fastest)
                for j in range(N):
                    alt = via + dk[j]
                    if alt < di[j]:
                        di[j] = alt

        # Replace INF with -1
        for i in range(N):
            row = dist[i]
            for j in range(N):
                if row[j] == INF:
                    row[j] = -1

        # ---------------------------------------
        # STEP 3: Convert N×N → 4-D distance grid
        # ---------------------------------------

        # Pre-allocate 4D structure
        d4 = [[[[0] * height for _ in range(width)] for _ in range(height)] for _ in range(width)]

        for y1 in range(height):
            for x1 in range(width):
                i = idx(x1, y1)
                row = dist[i]
                for y2 in range(height):
                    base = y2 * width
                    # slice assignment is fast:
                    for x2 in range(width):
                        d4[x1][y1][x2][y2] = row[base + x2]

        return d4

    @Timer('tick')
    def tick(self, ix: int, gs: GameState) -> None:
        """Game board tick"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        walls: Grid = data.layout.walls

        # ------------------------------------------------------
        # 1. Move *your own agent* to its exact observed position
        # ------------------------------------------------------
        my_state = agent_states[ix]
        my_x, my_y = map(round, my_state.configuration.pos)  # (x,y)

        # Remove you from every other square; place at exact position
        for x in range(walls.width):
            for y in range(walls.height):
                self.player_positions[x][y].discard(ix)
        self.player_positions[my_x][my_y].add(ix)

        # -----------------------------------------------------------------------
        # 2. Determine the enemy that moves BEFORE you (turn order relationship)
        # -----------------------------------------------------------------------
        enemy_ix = (ix - 1) % 4

        # Get enemy’s current GameState position
        enemy_state = agent_states[enemy_ix]
        enemy_config: Configuration = enemy_state.configuration

        # --------------------------------------------------------------
        # 3. If you SEE the enemy, collapse to the exact known position
        # --------------------------------------------------------------
        if enemy_config is not None:
            # exact collapse
            e_x, e_y = map(round, enemy_config.pos)  # (x,y)
            for x in range(walls.width):
                for y in range(walls.height):
                    self.player_positions[x][y].discard(enemy_ix)
            self.player_positions[e_x][e_y].add(enemy_ix)

            return

        # --------------------------------------------------------------
        # 4. If enemy NOT visible, propagate uncertainty
        # --------------------------------------------------------------
        new_positions: List[List[Set[int]]] = [[set() for _ in range(walls.height)] for _ in range(walls.width)]

        for x in range(walls.width):
            for y in range(walls.height):
                if enemy_ix in self.player_positions[x][y]:
                    # Expand to neighbors
                    for d_x, d_y in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                        n_x, n_y = x + d_x, y + d_y
                        if 0 <= n_x < walls.width and 0 <= n_y < walls.height and not walls.data[n_x][n_y]:
                            new_positions[n_x][n_y].add(enemy_ix)

        # Update positions
        # Keep other agents unchanged
        for x in range(walls.width):
            for y in range(walls.height):
                # Add the new positions
                self.player_positions[x][y].update(new_positions[x][y])

        if enemy_state.is_pacman:
            mid = walls.width // 2

            # Determine your home half
            if self.is_red:
                # Red home = left half (0 ... mid-1)
                for x in range(0, mid):
                    for y in range(walls.height):
                        self.player_positions[x][y].discard(enemy_ix)
            else:
                # Blue home = right half (mid ... width-1)
                for x in range(mid, walls.width):
                    for y in range(walls.height):
                        self.player_positions[x][y].discard(enemy_ix)


class TiTAgent(CaptureAgent):
    def __init__(self, index: int, gb: GameBoard) -> None:
        super().__init__(index, .9)

        self.gameboard: GameBoard = gb

    # @override
    @Timer('register_initial_state')
    def register_initial_state(self, gs: GameState) -> None:
        if self.index < 2:
            self.gameboard.setup(gs)

        # TODO A*

    # @override
    @Timer('choose_action')
    def choose_action(self, gs: GameState) -> str:
        self.gameboard.tick(self.index, gs)

        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states

        for as_ in agent_states:
            print(vars(as_))

        return Directions.STOP


class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.9):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.
        They can be either a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True, but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like. It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
