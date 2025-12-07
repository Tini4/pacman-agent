# coding=utf-8
"""My team"""

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

        # print(f'{self.name}{" " if self.name else ""}took {te - self._ts:.3f} seconds')

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
        self.is_red = is_red

        self.DISTS: List[List[List[List[int]]]] = []
        self.player_positions: List[List[Set[int]]] = []
        self.my_indexes: Tuple[int, int] = (-1, -1)
        self.enemy_indexes: Tuple[int, int] = (-1, -1)
        self.start = 0

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
        data: GameStateData = gs.data
        if data.timeleft < self.start:
            self.move_players(ix, gs)

        # TODO: remove!
        # plt.close()
        # self.draw_player_positions(gs)

    def draw_player_positions(self, gs: GameState) -> None:
        """Draw player positions"""
        data: GameStateData = gs.data
        layout: Layout = data.layout
        walls: Grid = layout.walls

        width, height = walls.width, walls.height
        colors = ['red', 'blue', 'green', 'orange']  # one per agent

        _fig, ax = plt.subplots(figsize=(width / 2, height / 2))

        # Draw walls
        for x in range(width):
            for y in range(height):
                if walls.data[x][y]:
                    ax.add_patch(plt.Rectangle((x, y), 1, 1, color='black'))

        # Draw agents
        for x in range(width):
            for y in range(height):
                agents_here = list(self.player_positions[x][y])
                if not agents_here:
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
        food: Grid = data.food

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

    def enemy_food(self, gs: GameState) -> List[Tuple[int, int]]:
        """Return list of (x,y) food coords that lie in our half only."""
        data: GameStateData = gs.data
        food: Grid = data.food

        mid = food.width // 2
        cells: List[Tuple[int, int]] = []
        for x in range(food.width):
            for y in range(food.height):
                if not food.data[x][y]:
                    continue
                # If we're red, our half is left (0 ... mid-1), else right (mid..width-1)
                if not self.is_red:
                    if x < mid:
                        cells.append((x, y))
                else:
                    if x >= mid:
                        cells.append((x, y))

        # if not self.is_red:  # TODO: do something special with this!!!
        #     cells.extend(gs.get_red_capsules())
        # else:
        #     cells.extend(gs.get_blue_capsules())

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

    def min_my_dist_to(self, gs: GameState, x: int, y: int) -> int:
        """Return the minimum distance from my agents to (x, y)"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        min_dist = INF
        for my_ix in self.my_indexes:
            my_state = agent_states[my_ix]
            my_x, my_y = map(round, my_state.configuration.pos)  # (x,y)
            d = self.DISTS[my_x][my_y][x][y]
            if d < min_dist:
                min_dist = d

        return min_dist

    def min_ally_dist_to(self, gs: GameState, ix: int, x: int, y: int) -> int:
        """Return the minimum distance from my other agents to (x, y)"""
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        min_dist = INF
        for my_ix in self.my_indexes:
            if my_ix == ix:
                continue

            my_state = agent_states[my_ix]
            my_x, my_y = map(round, my_state.configuration.pos)  # (x,y)
            d = self.DISTS[my_x][my_y][x][y]
            if d < min_dist:
                min_dist = d

        return min_dist

    def move_eaten(self, gs: GameState, e_x: int, e_y: int) -> None:
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

        # TODO: remove (fucked if different layout)?
        # if data.timeleft > self.gameboard.start - 120:
        #     return self.move_toward(gs, self.goal[0], self.goal[1])

        target = self.eat(gs)
        if target is not None:
            self.gameboard.move_eaten(gs, target[0], target[1])
        else:
            target = self.defend(gs)

        if target is None:
            target = self.attack(gs)

        if target is None:
            target = self.position

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
        """Defend my food"""
        data: GameStateData = gs.data
        layout: Layout = data.layout

        my_food = self.gameboard.my_food(gs)
        if not my_food:
            return None

        my_x, my_y = self.position

        legal = gs.get_legal_actions(self.index)

        # Movement deltas
        directions = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0),
            Directions.STOP: (0, 0),
        }

        best_action_worst_risk = INF
        best_action_target = None
        for direct, (d_x, d_y) in directions.items():
            if direct not in legal:
                continue

            n_x, n_y = my_x + d_x, my_y + d_y
            action_worst_risk = -INF
            action_target = None
            for f_x, f_y in my_food:
                if self.index < 2:
                    if self.gameboard.is_red:
                        if f_y > (layout.height * 2) // 3 + 1:
                            continue
                    else:
                        if f_y < (layout.height * 1) // 3 + 1:
                            continue
                else:
                    if self.gameboard.is_red:
                        if f_y < (layout.height * 1) // 3 + 1:
                            continue
                    else:
                        if f_y > (layout.height * 2) // 3 + 1:
                            continue

                enemy_dist = self.gameboard.min_enemy_dist_to(f_x, f_y)

                if (layout.height * 1) // 3 + 1 < f_y < (layout.height * 2) // 3 + 1:
                    my_dist_after = min(self.gameboard.DISTS[n_x][n_y][f_x][f_y],
                                        self.gameboard.min_ally_dist_to(gs, self.index, f_x, f_y))
                else:
                    my_dist_after = self.gameboard.DISTS[n_x][n_y][f_x][f_y]

                risk = my_dist_after - enemy_dist
                if risk > action_worst_risk:
                    action_worst_risk = risk
                    action_target = (n_x, n_y)

            if action_worst_risk < best_action_worst_risk:
                best_action_worst_risk = action_worst_risk
                best_action_target = action_target

        print(best_action_worst_risk)

        return best_action_target

    def eat(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """
        Return spot to eat an enemy.
        Otherwise, return None.
        """
        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states
        layout: Layout = data.layout
        mid = layout.width // 2
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

    def attack(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """
        Return a food tile to attack if it's safe: the agent can reach the food
        and return to its home half without being intercepted by enemies.
        Returns the target food tile (x, y), or None if no safe attack exists.
        """
        data: GameStateData = gs.data
        layout: Layout = data.layout
        my_x, my_y = self.position
        enemy_food = self.gameboard.enemy_food(gs)

        safe_foods = []
        for f_x, f_y in enemy_food:
            # Distance to food
            my_dist_to_food = self.gameboard.DISTS[my_x][my_y][f_x][f_y]

            # Distance from enemies to food
            enemy_dist_to_food = self.gameboard.min_enemy_dist_to(f_x, f_y)

            if my_dist_to_food >= enemy_dist_to_food:
                # Can't reach food before enemy
                continue

            # Check if path back to home half is safe
            # We'll consider all tiles in home half reachable from food
            if self.gameboard.is_red:
                safe_x = layout.width // 2 - 1
            else:
                safe_x = layout.width // 2
            safe_return = False
            for safe_y in range(layout.height):
                my_dist_back = self.gameboard.DISTS[f_x][f_y][safe_x][safe_y]
                enemy_dist_back = self.gameboard.min_enemy_dist_to(safe_x, safe_y)
                if my_dist_back < enemy_dist_back:
                    safe_return = True
                    break

            if safe_return:
                safe_foods.append((f_x, f_y))

        if not safe_foods:
            # No safe attack available
            return None

        # TODO: both agents shouldn't go for the same food
        # TODO: capsule!

        # Pick the closest safe food
        return max(safe_foods, key=lambda f: self.gameboard.DISTS[my_x][my_y][f[0]][f[1]])
