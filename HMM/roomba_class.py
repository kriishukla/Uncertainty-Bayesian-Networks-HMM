import random

class Roomba:
    """
    A class to simulate the movement of a Roomba robot on a grid with different movement policies.

    Attributes:
    - MOVEMENTS (dict): A dictionary mapping headings to (dx, dy) movement vectors.
    - HEADINGS (list): A list of possible headings (directions) for the Roomba.
    - is_obstacle (function): A function that checks if a given position is an obstacle.
    - position (tuple): The current position of the Roomba on the grid, represented as (x, y).
    - heading (str): The current heading (direction) of the Roomba, which can be one of the values in HEADINGS.
    - movement_policy (str): The movement policy of the Roomba, which can be 'straight_until_obstacle' 
                             or 'random_walk'.

    Methods:
    - __init__(self, MOVEMENTS, HEADINGS, is_obstacle, start_pos, start_heading, movement_policy):
        Initializes the Roomba with movement rules, headings, obstacle detection, 
        starting position, heading, and movement policy.

    - move(self):
        Moves the Roomba based on the specified movement policy. Calls the appropriate method 
        for the selected policy. Raises a ValueError if an unknown movement policy is provided.

    - straight_until_obstacle_move(self):
        Implements the 'straight_until_obstacle' movement policy:
        - Moves the Roomba in the current heading direction until it encounters an obstacle.
        - If an obstacle is encountered, the Roomba chooses a new random heading and attempts to move.
        - If the new position is also an obstacle, the Roomba stays in place; otherwise, it moves.

    - random_walk_move(self):
        Implements the 'random_walk' movement policy:
        - Chooses a new random heading and attempts to move.
        - If the new position is an obstacle, the Roomba stays in place; otherwise, it moves.
    """
    def __init__(self, MOVEMENTS, HEADINGS,is_obstacle,start_pos, start_heading, movement_policy='straight_until_obstacle'):
        self.MOVEMENTS = MOVEMENTS
        self.HEADINGS = HEADINGS
        self.is_obstacle = is_obstacle
        self.position = start_pos
        self.heading = start_heading
        self.movement_policy = movement_policy

    def move(self):
        if self.movement_policy == 'straight_until_obstacle':
            return self.straight_until_obstacle_move()
        elif self.movement_policy == 'random_walk':
            return self.random_walk_move()
        else:
            raise ValueError('Unknown movement policy')

    def straight_until_obstacle_move(self):
        dx, dy = self.MOVEMENTS[self.heading]
        new_position = (self.position[0] + dx, self.position[1] + dy)

        if self.is_obstacle(new_position):
            self.heading = random.choice(self.HEADINGS)
            dx, dy = self.MOVEMENTS[self.heading]
            new_position = (self.position[0] + dx, self.position[1] + dy)
            if self.is_obstacle(new_position):
                return self.position
            else:
                self.position = new_position
                return self.position
        else:
            self.position = new_position
            return self.position

    def random_walk_move(self):
        self.heading = random.choice(self.HEADINGS)
        dx, dy = self.MOVEMENTS[self.heading]
        new_position = (self.position[0] + dx, self.position[1] + dy)
        if self.is_obstacle(new_position):
            return self.position
        else:
            self.position = new_position
            return self.position
