# coding=utf-8
"""My team"""

import random
from time import perf_counter
from typing import List, Callable, Tuple, Set, Optional

# TODO: REMOVE!!!!!
import matplotlib.pyplot as plt  # type: ignore

import contest.util as util  # type: ignore
from contest.capture import GameState  # type: ignore
from contest.capture_agents import CaptureAgent  # type: ignore
from contest.game import Directions, GameStateData, Grid, AgentState, Configuration  # type: ignore
from contest.layout import Layout  # type: ignore
from contest.util import nearest_point  # type: ignore


# TODO: REMOVE!!!!!
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


INF: int = 10 ** 9


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
        self.my_indexes: Tuple[int, int] = (-1, -1)
        self.enemy_indexes: Tuple[int, int] = (-1, -1)
        self.start = 0

        self.is_red = is_red

    @Timer('setup')
    def setup(self, gs: GameState) -> None:
        """Game board setup"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        layout: Layout = data.layout
        walls: Grid = layout.walls

        if self.is_red:
            self.my_indexes = tuple(gs.red_team)
            self.enemy_indexes = tuple(gs.blue_team)
        else:
            self.my_indexes = tuple(gs.blue_team)
            self.enemy_indexes = tuple(gs.red_team)

        self.DISTS = self.floyd_warshall(walls)

        self.player_positions = [[set() for _ in range(layout.height)] for _ in range(layout.width)]
        for i in range(4):
            p = agent_states[i].start.pos
            self.player_positions[p[0]][p[1]].add(i)

        self.start = data.timeleft

    @Timer('Floyd-Warshall')
    def floyd_warshall(self, walls: Grid) -> List[List[List[List[int]]]]:
        """Floyd-Warshall algorithm"""
        width = walls.width
        height = walls.height
        grid = walls.data

        N: int = width * height

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
        self.move_players(ix, gs)

        # self.draw_player_positions(gs)

    def draw_player_positions(self, gs: GameState) -> None:
        """Draw player positions"""
        data: GameStateData = gs.data
        layout: Layout = data.layout
        walls: Grid = layout.walls

        width, height = walls.width, walls.height
        colors = ['red', 'blue', 'green', 'orange']  # one per agent

        fig, ax = plt.subplots(figsize=(width / 2, height / 2))

        # Draw walls
        for x in range(width):
            for y in range(height):
                if walls.data[x][y]:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

        # Draw agents
        for x in range(width):
            for y in range(height):
                agents_here = list(self.player_positions[x][y])
                n = len(agents_here)
                if n == 0:
                    continue

                # Slight offsets to avoid overlap
                offsets = [(0, 0), (-0.2, 0.2), (0.2, 0.2), (-0.2, -0.2)]
                for i, agent in enumerate(agents_here):
                    dx, dy = offsets[i % 4]
                    ax.scatter(x + 0.5 + dx, y + 0.5 + dy,
                               c=colors[agent], s=200, edgecolors='black', zorder=5)

        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        ax.set_aspect('equal')
        ax.grid(True)
        plt.show()

    def move_players(self, ix: int, gs: GameState) -> None:
        """Move players"""
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

        # -------------------------------------------------
        # 5. Remove enemy from wrong halve
        # -------------------------------------------------
        mid = walls.width // 2
        if enemy_state.is_pacman:
            # Determine your home half
            if self.is_red:
                # Red home = left half (0 ... mid-1)
                for x in range(mid, walls.width):
                    for y in range(walls.height):
                        self.player_positions[x][y].discard(enemy_ix)
            else:
                # Blue home = right half (mid ... width-1)
                for x in range(0, mid):
                    for y in range(walls.height):
                        self.player_positions[x][y].discard(enemy_ix)
        else:
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

        # -------------------------------------------------
        # 6. Remove enemy from cells around friendly agents
        # -------------------------------------------------
        SAFE_RADIUS = 5  # Manhattan distance radius around friendly agents

        for friendly_ix in self.my_indexes:  # iterate over all friendly agents
            friendly_state = agent_states[friendly_ix]
            f_x, f_y = map(round, friendly_state.configuration.pos)

            # remove enemy_ix from all cells within SAFE_RADIUS
            for d_x in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                for d_y in range(-SAFE_RADIUS, SAFE_RADIUS + 1):
                    if abs(d_x) + abs(d_y) <= SAFE_RADIUS:  # Manhattan distance
                        n_x, n_y = f_x + d_x, f_y + d_y
                        if 0 <= n_x < walls.width and 0 <= n_y < walls.height:
                            self.player_positions[n_x][n_y].discard(enemy_ix)

        # TODO: ideas
        #     If enemy eats food, do we know where it is?
        #     Also if enemy eats capsule?

    def my_food(self, gs: GameState) -> List[Tuple[int, int]]:
        """Return list of (x,y) food coords that lie in our half only."""
        data: GameStateData = gs.data
        layout: Layout = data.layout
        food: Grid = layout.food

        mid = food.width // 2
        cells: List[Tuple[int, int]] = []
        for x in range(food.width):
            for y in range(food.height):
                if not food.data[x][y]:
                    continue
                # If we're red, our half is left (0 ... mid-1), else right (mid..width-1)
                if self.is_red:
                    if x < mid:
                        cells.append((x, y))
                else:
                    if x >= mid:
                        cells.append((x, y))

        if self.is_red:
            cells.extend(gs.get_red_capsules())
        else:
            cells.extend(gs.get_blue_capsules())

        return cells

    def min_player_dist_to(self, ix: int, x: int, y: int) -> int:
        """
        Return the minimum distance a player *could* be to (x, y)
        given current uncertainty in self.player_positions.
        If a player might be in multiple squares, we take min.
        If unreachable, returns a large number.
        """
        min_dist = INF
        # For each possible position where player may be:
        for p_x in range(len(self.player_positions)):
            for p_y in range(len(self.player_positions[0])):
                if ix in self.player_positions[p_x][p_y]:
                    d = self.DISTS[p_x][p_y][x][y]
                    if d < min_dist:
                        min_dist = d

        return min_dist

    def min_enemy_dist_to(self, x: int, y: int) -> int:
        """Return the minimum distance any enemy *could* be to (x, y)"""

        min_dist = INF
        for enemy_ix in self.enemy_indexes:
            d = self.min_player_dist_to(enemy_ix, x, y)
            if d < min_dist:
                min_dist = d

        return min_dist

    def eat(self, gs: GameState, e_x: int, e_y: int) -> None:
        """Remove enemy agent and respawn him"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        eaten = self.player_positions[e_x][e_y].intersection(set(self.enemy_indexes))
        for e_ix in eaten:
            for x in range(len(self.player_positions)):
                for y in range(len(self.player_positions[x])):
                    self.player_positions[x][y].discard(e_ix)
            p = agent_states[e_ix].start.pos
            self.player_positions[p[0]][p[1]].add(e_ix)


class TiTAgent(CaptureAgent):
    def __init__(self, index: int, gb: GameBoard) -> None:
        super().__init__(index, .9)

        self.gameboard: GameBoard = gb
        self.goal: Tuple[int, int] = (0, 0)
        self.position: Tuple[int, int] = (0, 0)

    # @override
    @Timer('register_initial_state')
    def register_initial_state(self, gs: GameState) -> None:
        if self.index < 2:
            self.gameboard.setup(gs)

            self.goal = (20, 9)
        else:
            self.goal = (20, 2)

        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states

        my_state = agent_states[self.index]
        self.position = tuple(map(round, my_state.configuration.pos))  # (x,y)

    # @override
    @Timer('choose_action')
    def choose_action(self, gs: GameState) -> str:
        self.gameboard.tick(self.index, gs)

        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states

        my_state = agent_states[self.index]
        self.position = tuple(map(round, my_state.configuration.pos))  # (x,y)

        if data.timeleft > self.gameboard.start - 120:
            return self.move_toward(gs, self.goal[0], self.goal[1])

        target = self.eat(gs)
        if target is not None:
            self.gameboard.eat(gs, target[0], target[1])
        else:
            target = self.defend(gs)
            if target is None:
                target = self.attack(gs)

        return self.move_toward(gs, target[0], target[1])

    def move_toward(self, gs: GameState, t_x: int, t_y: int) -> str:
        """
        Return the best single action to move toward (tx, ty),
        using the Floyd–Warshall precomputed DISTS.
        """

        # get current position
        my_x, my_y = self.position

        # If already there
        if (my_x, my_y) == (t_x, t_y):
            return Directions.STOP

        # If target is unreachable
        if self.gameboard.DISTS[my_x][my_y][t_x][t_y] >= INF:
            return Directions.STOP

        # Try each possible move
        best_action = Directions.STOP
        best_dist = INF

        legal = gs.get_legal_actions(self.index)

        # Movement deltas
        directions = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0),
        }

        for direct, (d_x, d_y) in directions.items():
            if direct not in legal:
                continue

            n_x, n_y = my_x + d_x, my_y + d_y

            # Distance from neighbor toward target
            dist = self.gameboard.DISTS[n_x][n_y][t_x][t_y]

            if dist < best_dist:
                best_dist = dist
                best_action = direct

        return best_action

    def defend(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """ TODO: do not defend if teammate is near enough?
        Determine the best tile to move toward to defend food.
        Returns (x, y) of the target defensive position.
        """
        my_food_list = self.gameboard.my_food(gs)

        if not my_food_list:
            # No food left, stay put
            return None

        best_food = None
        worst_risk = -INF  # risk = enemy distance - my distance (negative = dangerous)

        for fx, fy in my_food_list:
            enemy_dist = self.gameboard.min_enemy_dist_to(fx, fy)
            my_dist = self.gameboard.min_player_dist_to(self.index, fx, fy)
            risk = my_dist - enemy_dist  # positive if you are farther than enemy

            if risk < worst_risk:
                continue  # safer than the previous worst food
            worst_risk = risk
            best_food = (fx, fy)

        if best_food is None:
            # All food is safe, stay put
            return None

        # STEP 2: Find defensive tile (closest to you while still intercepting)
        fx, fy = best_food
        width = gs.data.layout.walls.width
        height = gs.data.layout.walls.height
        mid = width // 2

        candidates = []

        for x in range(width):
            for y in range(height):
                # Must be in your half
                if self.gameboard.is_red and x >= mid:
                    continue
                if not self.gameboard.is_red and x < mid:
                    continue

                # Only consider tiles that let you reach food no later than enemy
                my_d = self.gameboard.min_player_dist_to(self.index, x, y)
                enemy_d = self.gameboard.min_enemy_dist_to(fx, fy)
                # You can be at most as far as enemy from food
                if self.gameboard.DISTS[x][y][fx][fy] <= enemy_d:
                    candidates.append((x, y))

        if not candidates:
            # Fallback: just move toward the food itself
            return best_food

        # Pick the candidate closest to current position
        cx, cy = self.position
        best_tile = min(candidates, key=lambda t: self.gameboard.DISTS[cx][cy][t[0]][t[1]])

        return best_tile

    def eat(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """ TODO
        Return a tile to attack an enemy if one is nearby on your half.
        Otherwise, return None.
        """
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states

        walls: Grid = gs.data.layout.walls
        width, height = walls.width, walls.height
        mid = width // 2
        my_x, my_y = self.position

        # Check all enemy positions
        for enemy_ix in self.gameboard.enemy_indexes:
            enemy_state = agent_states[enemy_ix]
            enemy_config: Configuration = enemy_state.configuration
            if enemy_config is None:
                continue

            e_x, e_y = map(round, enemy_config.pos)  # (x,y)

            # Only consider enemies on your half
            if self.gameboard.is_red and e_x >= mid:
                continue
            if not self.gameboard.is_red and e_x < mid:
                continue

            dist = abs(my_x - e_x) + abs(my_y - e_y)
            if dist <= 1:
                return e_x, e_y

        # No nearby enemy, no attack
        return None

    def attack(self, gs: GameState) -> Tuple[int, int]:
        """
        Return a food tile to attack if it's safe: the agent can reach the food
        and return to its home half without being intercepted by enemies.
        Returns the target food tile (x, y), or None if no safe attack exists.
        """
        my_x, my_y = self.position
        width = gs.data.layout.walls.width
        mid = width // 2
        my_food_list = gs.data.layout.food

        # Determine home half x-range
        if self.gameboard.is_red:
            home_x_range = range(0, mid)
        else:
            home_x_range = range(mid, width)

        # Candidate foods on enemy half
        enemy_food = [
            (x, y) for x in range(my_food_list.width)
            for y in range(my_food_list.height)
            if my_food_list.data[x][y] and (x not in home_x_range)
        ]

        safe_foods = []

        for fx, fy in enemy_food:
            # Distance to food
            my_dist_to_food = self.gameboard.DISTS[my_x][my_y][fx][fy]

            # Distance from enemies to food
            enemy_dist_to_food = self.gameboard.min_enemy_dist_to(fx, fy)

            if my_dist_to_food >= enemy_dist_to_food:
                # Can't reach food before enemy → unsafe
                continue

            # Check if path back to home half is safe
            # We'll consider all tiles in home half reachable from food
            safe_return = False
            for hx in home_x_range:
                for hy in range(gs.data.layout.walls.height):
                    my_dist_back = self.gameboard.DISTS[fx][fy][hx][hy]
                    enemy_dist_back = self.gameboard.min_enemy_dist_to(hx, hy)
                    # Only safe if we reach home tile before any enemy
                    if my_dist_back < enemy_dist_back:
                        safe_return = True
                        break
                if safe_return:
                    break

            if safe_return:
                safe_foods.append((fx, fy))

        if not safe_foods:
            # No safe attack available
            return my_x, my_y

        # Pick the closest safe food
        target = min(safe_foods, key=lambda t: self.gameboard.DISTS[my_x][my_y][t[0]][t[1]])
        return target


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
