######################################
#        Tóth Bálint (PME4BQ)        #
# Artificial Intelligence Laboratory #
#               2023                 #
######################################

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gym_maze.envs.mazer import Maze

class MazeWorld(gym.Env):

    metadata = {'render_modes': ['ascii']}    

    def __init__(self, node_space_width: int = 13, node_space_height: int = 13, seed: int = 20231120, loops: bool = False, num_loop: int = 2):
        self.seed = seed
        self.height = node_space_height * 2 + 1
        self.width = node_space_width * 2 + 1
        # generate a maze
        self.maze = Maze(self.height, self.width, self.seed, loops, num_loop)

        # make the states
        # we will make the environment fully observable for now
        self.observation_space = spaces.Dict({
            # starting position of the agent in upper left corner
            "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.height-1, self.width-1]), dtype=np.int8),
            "target": spaces.Box(low=np.array([0,0]), high=np.array([self.height-1, self.width-1]), dtype=np.int8), 
            "environment": spaces.MultiBinary([self.height, self.width])
        })

        # action space
        self.action_space = spaces.Discrete(1)
        self.action_to_direction = {
            "up": np.array([-1, 0]),
            "right": np.array([0, 1]),
            "down": np.array([1, 0]),
            "left": np.array([0, -1]),
        }

        self._agent_location = np.array([1,1])
        self._target_location = np.array([self.height-2, self.width-2])

    def _observations(self) -> dict:
        return {"agent": self._agent_location, "target": self._target_location, "environment": self.maze.get_grid()}
    
    def _get_info(self) -> dict:
        return {"distance": np.abs(self._agent_location[0] - self._target_location[0]) + 
                                np.abs(self._agent_location[1] - self._target_location[1])}

    def reset(self, seed: int=None, options=None) -> (dict, str):
        # super().reset(seed=None, options=options)

        self._agent_location = np.array([1,1])
        self._target_location = np.array([self.height-2, self.width-2])

        info = self._get_info()
        obs = self._observations()

        self.render()

        return obs, info
    
    def render(self):
        self.maze.pretty_print(
            {"i": self._agent_location[0], "j": self._agent_location[1]},
            {"i": self._target_location[0], "j": self._target_location[1]}
        )
    
    def step(self, action: str):
        direction = self.action_to_direction[action]
        prev_location = self._agent_location
        self._agent_location = np.clip(
            self._agent_location + direction, 0, [self.height - 1, self.width - 1]
        )
        if self.maze.get_grid()[self._agent_location[0], self._agent_location[1]] == 0:
            self._agent_location = prev_location
        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = self._reward(terminated)

        info = self._get_info()
        obs = self._observations()
        self.render()
        return obs, reward, terminated, False, info
    
    def _reward(self, found: bool) -> int | None:
        i = self._agent_location[0]
        j = self._agent_location[1]
        # arrived to the target
        if found:
            return 0
        # in a wall
        if self.maze._grid[i,j] == 1:
            return -1
        # just chilling (based on the stepping rules, this is unreachable)
        else:  
            return -1000

