# coding=utf-8
"""My team"""
from typing import List, Tuple, Set, Optional

from contest.capture import GameState  # type: ignore
from contest.capture_agents import CaptureAgent  # type: ignore
from contest.game import Directions, GameStateData, Grid, AgentState, Configuration  # type: ignore
from contest.layout import Layout  # type: ignore

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

    def tick(self, ix: int, gs: GameState) -> None:
        """Game board tick"""
        data: GameStateData = gs.data
        if data.timeleft < self.start:
            self.move_players(ix, gs)

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

    def min_enemy_dist_to_if_dangerous(self, gs: GameState, x: int, y: int):
        """
        Returns minimum enemy distance to (x, y), but ignores enemies whose scared_timer > 1.
        This means they cannot eat us even if they reach us.
        """
        min_dist = INF
        for enemy_ix in self.enemy_indexes:
            st = gs.data.agent_states[enemy_ix]
            if st.scared_timer > 0:
                # Enemy is scared, cannot eat us
                continue

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
    def choose_action(self, gs: GameState) -> str:
        self.gameboard.tick(self.index, gs)

        data: GameStateData = gs.data
        agent_states: List[AgentState] = data.agent_states

        my_state = agent_states[self.index]
        self.position = tuple(map(round, my_state.configuration.pos))  # (x,y)

        target = self.eat(gs)
        if target is not None:
            self.gameboard.move_eaten(gs, target[0], target[1])

        if target is None:
            target = self.run_home(gs)

        if target is None:
            target = self.defend_capsule(gs)

        if target is None:
            intruder = False
            for e_ix in self.gameboard.enemy_indexes:
                enemy_state = agent_states[e_ix]
                if enemy_state.is_pacman:
                    intruder = True

            if intruder:
                target = self.defend(gs)
            else:
                target = self.attack(gs)

                if target is None:
                    target = self.defend(gs)

        if target is None:
            # print('cry!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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
        agent_states: List[AgentState] = data.agent_states

        my_x, my_y = self.position

        if self.gameboard.is_red:
            safe_x = layout.width // 2 - 1
        else:
            safe_x = layout.width // 2

        safe_spots = [(safe_x, safe_y) for safe_y in range(layout.height)]
        safe_spots.sort(key=lambda s: self.gameboard.DISTS[my_x][my_y][s[0]][s[1]])
        if self.gameboard.is_red:
            if my_x > safe_x:
                for safe_x, safe_y in safe_spots:
                    my_dist_back = self.gameboard.DISTS[my_x][my_y][safe_x][safe_y]
                    enemy_dist_back = self.gameboard.min_enemy_dist_to(safe_x, safe_y)
                    if my_dist_back < enemy_dist_back:
                        return safe_x, safe_y

                # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        else:
            if my_x < safe_x:
                for safe_x, safe_y in safe_spots:
                    my_dist_back = self.gameboard.DISTS[my_x][my_y][safe_x][safe_y]
                    enemy_dist_back = self.gameboard.min_enemy_dist_to(safe_x, safe_y)
                    if my_dist_back < enemy_dist_back:
                        return safe_x, safe_y

                # print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')

        for e_ix in self.gameboard.enemy_indexes:
            enemy_state = agent_states[e_ix]
            if not enemy_state.is_pacman:
                continue

            enemy_config: Configuration = enemy_state.configuration
            if enemy_config is None:
                continue

            e_x, e_y = map(round, enemy_config.pos)  # (x,y)

            my_dist = self.gameboard.DISTS[my_x][my_y][e_x][e_y]
            ally_dist = self.gameboard.min_ally_dist_to(gs, self.index, e_x, e_y)
            if my_dist > ally_dist:
                continue

            return e_x, e_y

        my_food = self.gameboard.my_food(gs)
        if not my_food:
            return None

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

        return best_action_target

    def defend_capsule(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """Defend my capsule"""
        my_x, my_y = self.position

        if self.gameboard.is_red:
            capsules = gs.get_red_capsules()
        else:
            capsules = gs.get_blue_capsules()

        if not capsules:
            return None

        best_capsule = None
        worst_risk = -INF
        for cap_x, cap_y in capsules:
            # Distance from my agents
            my_dist = self.gameboard.min_my_dist_to(gs, cap_x, cap_y)

            # Distance from the closest enemy
            enemy_dist = self.gameboard.min_enemy_dist_to(cap_x, cap_y)

            # Risk = enemy closer than us
            risk = my_dist - enemy_dist

            # Prefer capsule that is most threatened (lowest or negative risk)
            if risk > worst_risk:
                worst_risk = risk
                best_capsule = (cap_x, cap_y)

        if best_capsule is None:
            return None

        if worst_risk < -1:
            return None

        my_dist = self.gameboard.DISTS[my_x][my_y][best_capsule[0]][best_capsule[1]]
        ally_dist = self.gameboard.min_ally_dist_to(gs, self.index, best_capsule[0], best_capsule[1])
        if my_dist > ally_dist:
            return None

        return best_capsule

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

            st = gs.data.agent_states[enemy_ix]
            if st.scared_timer == 0:
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

        if not self.gameboard.is_red:
            capsules = gs.get_red_capsules()
        else:
            capsules = gs.get_blue_capsules()

        for cap_x, cap_y in capsules:
            enemy_dist = self.gameboard.min_enemy_dist_to_if_dangerous(gs, cap_x, cap_y)
            my_dist = self.gameboard.DISTS[my_x][my_y][cap_x][cap_y]

            if my_dist <= enemy_dist:
                return cap_x, cap_y

        enemy_food = self.gameboard.enemy_food(gs)

        if not enemy_food:
            return None

        eat_food = []
        for f_x, f_y in enemy_food:
            if self.index < 2:
                if self.gameboard.is_red:
                    if f_y >= layout.height // 2:
                        continue
                else:
                    if f_y < layout.height // 2:
                        continue
            else:
                if self.gameboard.is_red:
                    if f_y < layout.height // 2:
                        continue
                else:
                    if f_y >= layout.height // 2:
                        continue

            eat_food.append((f_x, f_y))

        if not eat_food:
            return None

        f_x, f_y = min(eat_food, key=lambda f: self.gameboard.DISTS[my_x][my_y][f[0]][f[1]])

        test = self.move_toward(gs, f_x, f_y)
        directions = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0),
            Directions.STOP: (0, 0),
        }
        my_x += directions[test][0]
        my_y += directions[test][1]

        if self.gameboard.is_red:
            if my_x < layout.width // 2:
                return f_x, f_y
        else:
            if my_x >= layout.width // 2:
                return f_x, f_y

        d = self.gameboard.min_enemy_dist_to_if_dangerous(gs, my_x, my_y)

        if d <= 1:
            return None

        # Check if path back to home half is safe
        if self.gameboard.is_red:
            safe_x = layout.width // 2 - 1
        else:
            safe_x = layout.width // 2

        safe_spots = [(safe_x, safe_y) for safe_y in range(layout.height)]
        safe_spots.sort(key=lambda s: self.gameboard.DISTS[my_x][my_y][s[0]][s[1]])

        safe_return = False
        for safe_x, safe_y in safe_spots:
            my_dist_back = self.gameboard.DISTS[my_x][my_y][safe_x][safe_y] + 2
            enemy_dist_back = self.gameboard.min_enemy_dist_to_if_dangerous(gs, safe_x, safe_y)
            if my_dist_back < enemy_dist_back:
                safe_return = True
                break

        if not safe_return:
            return None

        return f_x, f_y

    def run_home(self, gs: GameState) -> Optional[Tuple[int, int]]:
        """At the end, run home"""
        data: GameStateData = gs.data
        layout: Layout = data.layout

        if data.timeleft > 100:
            return None

        my_x, my_y = self.position

        if self.gameboard.is_red:
            safe_x = layout.width // 2 - 1
        else:
            safe_x = layout.width // 2

        safe_spots = [(safe_x, safe_y) for safe_y in range(layout.height)]
        # safe_spots.sort(key=lambda s: self.gameboard.DISTS[my_x][my_y][s[0]][s[1]])

        target = None
        if self.gameboard.is_red:
            if my_x > safe_x - 1:
                for safe_x, safe_y in safe_spots:
                    my_dist_back = self.gameboard.DISTS[my_x][my_y][safe_x][safe_y]
                    enemy_dist_back = self.gameboard.min_enemy_dist_to_if_dangerous(gs, safe_x, safe_y)
                    if my_dist_back < enemy_dist_back:
                        target = (safe_x, safe_y)

                # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        else:
            if my_x < safe_x + 1:
                for safe_x, safe_y in safe_spots:
                    my_dist_back = self.gameboard.DISTS[my_x][my_y][safe_x][safe_y]
                    enemy_dist_back = self.gameboard.min_enemy_dist_to_if_dangerous(gs, safe_x, safe_y)
                    if my_dist_back < enemy_dist_back:
                        target = (safe_x, safe_y)

                # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')

        if target is None:
            # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
            return None

        return target
