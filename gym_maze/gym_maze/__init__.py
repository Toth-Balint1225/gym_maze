######################################
#        Tóth Bálint (PME4BQ)        #
# Artificial Intelligence Laboratory #
#               2023                 #
######################################

from gym_maze.envs.maze_world import MazeWorld
from gym_maze.envs.maze_world import Maze

from gymnasium.envs.registration import register 


register(
    id="gym_maze/MazeWorld-v0",
    entry_point="gym_maze.envs:MazeWorld"
)
