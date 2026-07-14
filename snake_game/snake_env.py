import random
from collections import deque


class SnakeEnv:
    """
    Improved Snake environment for tabular RL and DQN.

    Actions are relative:
        0 = go straight
        1 = turn right
        2 = turn left

    Main improvements:
        - stronger death penalty
        - stronger food reward
        - loop/spin penalty
        - distance-to-food reward shaping
        - action-based open-space features
        - action-based tail-reach features
    """

    ACTIONS = (0, 1, 2)

    def __init__(self, width=10, height=10, seed=1):
        self.width = width
        self.height = height
        self.random = random.Random(seed)
        self.seed = seed
        self.reset()

    def reset(self):
        cx = self.width // 2
        cy = self.height // 2

        self.snake = [
            (cx, cy),
            (cx - 1, cy),
            (cx - 2, cy),
        ]

        self.direction = (1, 0)
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.done = False
        self.recent_actions = deque(maxlen=8)
        self.food = self._spawn_food()

        return self.get_state()

    def _spawn_food(self):
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in self.snake
        ]

        if not empty:
            return None

        return self.random.choice(empty)

    def _turn(self, action, direction=None):
        if direction is None:
            direction = self.direction

        dx, dy = direction

        if action == 0:
            return (dx, dy)

        if action == 1:
            return (-dy, dx)

        if action == 2:
            return (dy, -dx)

        raise ValueError("Action must be 0, 1 or 2.")

    def _next_position(self, direction, head=None):
        if head is None:
            head = self.snake[0]

        return (head[0] + direction[0], head[1] + direction[1])

    def _is_collision(self, position, snake_body=None):
        if snake_body is None:
            snake_body = self.snake

        x, y = position

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True

        return position in snake_body

    def _distance_to_food(self, position=None):
        if self.food is None:
            return 0

        if position is None:
            position = self.snake[0]

        return abs(position[0] - self.food[0]) + abs(position[1] - self.food[1])

    def _reachable_space(self, start, snake_body):
        if self._is_collision(start, snake_body):
            return 0

        queue = deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = (x + dx, y + dy)

                if nxt in visited:
                    continue

                if self._is_collision(nxt, snake_body):
                    continue

                visited.add(nxt)
                queue.append(nxt)

        return len(visited)

    def _can_reach_tail(self, start, snake_body):
        if not snake_body:
            return True

        tail = snake_body[-1]

        if start == tail:
            return True

        queue = deque([start])
        visited = {start}

        while queue:
            x, y = queue.popleft()

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nxt = (x + dx, y + dy)

                if nxt in visited:
                    continue

                if nxt != tail and self._is_collision(nxt, snake_body):
                    continue

                if nxt == tail:
                    return True

                visited.add(nxt)
                queue.append(nxt)

        return False

    def _simulate_action(self, action):
        new_direction = self._turn(action)
        new_head = self._next_position(new_direction)
        will_eat = new_head == self.food

        body_for_collision = self.snake if will_eat else self.snake[:-1]
        collision = self._is_collision(new_head, body_for_collision)

        if collision:
            predicted_snake = [new_head] + self.snake
        else:
            predicted_snake = [new_head] + self.snake
            if not will_eat:
                predicted_snake = predicted_snake[:-1]

        return {
            "action": action,
            "direction": new_direction,
            "new_head": new_head,
            "will_eat": will_eat,
            "collision": collision,
            "predicted_snake": predicted_snake,
            "open_space": 0 if collision else self._reachable_space(new_head, predicted_snake),
            "can_reach_tail": False if collision else self._can_reach_tail(new_head, predicted_snake),
        }

    def _space_bucket(self, open_space):
        """
        Bucket open space so tabular state stays small.

        0 = dangerous/blocked
        1 = very small
        2 = enough for current body
        3 = comfortably open
        """
        if open_space == 0:
            return 0
        if open_space < len(self.snake):
            return 1
        if open_space < len(self.snake) * 2:
            return 2
        return 3

    def _food_relative_to_action(self, action):
        """
        Whether taking this relative action points generally toward the food.
        """
        simulated = self._simulate_action(action)
        new_head = simulated["new_head"]

        return int(self._distance_to_food(new_head) < self._distance_to_food())

    def get_state(self):
        """
        Compact state for tabular methods and DQN.

        It includes action-based planning features:
            open space if straight/right/left
            tail reach if straight/right/left
        """
        sims = [self._simulate_action(action) for action in self.ACTIONS]

        danger_straight = int(sims[0]["collision"])
        danger_right = int(sims[1]["collision"])
        danger_left = int(sims[2]["collision"])

        moving_up = int(self.direction == (0, -1))
        moving_down = int(self.direction == (0, 1))
        moving_left = int(self.direction == (-1, 0))
        moving_right = int(self.direction == (1, 0))

        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

        food_straight = self._food_relative_to_action(0)
        food_turn_right = self._food_relative_to_action(1)
        food_turn_left = self._food_relative_to_action(2)

        open_space_straight = self._space_bucket(sims[0]["open_space"])
        open_space_right = self._space_bucket(sims[1]["open_space"])
        open_space_left = self._space_bucket(sims[2]["open_space"])

        can_reach_tail_straight = int(sims[0]["can_reach_tail"])
        can_reach_tail_right = int(sims[1]["can_reach_tail"])
        can_reach_tail_left = int(sims[2]["can_reach_tail"])

        return (
            danger_straight,
            danger_right,
            danger_left,
            moving_up,
            moving_down,
            moving_left,
            moving_right,
            food_up,
            food_down,
            food_left,
            food_right,
            food_straight,
            food_turn_right,
            food_turn_left,
            open_space_straight,
            open_space_right,
            open_space_left,
            can_reach_tail_straight,
            can_reach_tail_right,
            can_reach_tail_left,
        )

    def step(self, action):
        if self.done:
            return self.get_state(), 0, True, {"score": self.score}

        self.steps += 1
        self.steps_since_food += 1
        self.recent_actions.append(action)

        old_distance = self._distance_to_food()
        sim = self._simulate_action(action)

        if sim["collision"]:
            self.done = True
            return self.get_state(), -100.0, True, {
                "score": self.score,
                "death": True,
                "reason": "collision",
            }

        reward = -0.10

        # Loop/spin penalty. A snake that keeps turning without food should be discouraged.
        if action != 0:
            reward -= 0.05

        if len(self.recent_actions) == self.recent_actions.maxlen:
            turning_ratio = sum(1 for a in self.recent_actions if a != 0) / len(self.recent_actions)
            if turning_ratio > 0.75:
                reward -= 0.5

        self.direction = sim["direction"]
        self.snake.insert(0, sim["new_head"])

        if sim["will_eat"]:
            self.score += 1
            reward += 50.0
            self.steps_since_food = 0
            self.food = self._spawn_food()

            if self.food is None:
                self.done = True
                return self.get_state(), reward, True, {"score": self.score, "won": True}
        else:
            self.snake.pop()

            new_distance = self._distance_to_food()
            if new_distance < old_distance:
                reward += 1.0
            elif new_distance > old_distance:
                reward -= 1.0

        # Planning-style reward shaping after the move.
        open_space = self._reachable_space(self.snake[0], self.snake)

        if open_space < len(self.snake):
            reward -= 10.0
        elif open_space < len(self.snake) * 2:
            reward -= 3.0
        else:
            reward += 0.3

        if self._can_reach_tail(self.snake[0], self.snake):
            reward += 0.5
        else:
            reward -= 5.0

        # Stop very long wandering episodes.
        max_steps_without_food = 80 + len(self.snake) * 10
        if self.steps_since_food > max_steps_without_food:
            self.done = True
            reward -= 30.0
            return self.get_state(), reward, True, {
                "score": self.score,
                "timeout": True,
                "reason": "too long without food",
            }

        return self.get_state(), reward, self.done, {"score": self.score}

    def clone_for_render(self):
        return {
            "width": self.width,
            "height": self.height,
            "snake": list(self.snake),
            "food": self.food,
            "score": self.score,
            "steps": self.steps,
        }
