import numpy as np
from gym_maze import MazeWorld
import os
from time import sleep

class SimpleAgent:
    # moves available (should query it from the env, but this is ok for now)
    moveset = {"up":    {"i": -1, "j": 0}, 
               "down":  {"i": 1, "j": 0},
               "left":  {"i": 0, "j": -1},
               "right": {"i": 0, "j": 1}}

    def __init__(self, env: MazeWorld):
        # store the environment just in case
        # this agent is specifically designed for the 
        self.env: MazeWorld

        # state / grid transformation and state management
        self.state_coord_map: dict = {}
        self.state_count: int

        # Fully Observable MDP descriptors
        self.reward: np.ndarray
        self.policy: dict = {}
        self.probabilities: dict = {}
        # state values 
        self.current: np.ndarray
        self.next: np.ndarray

        # target state index for the matrix later
        self.target_state: int
        # the agent i and j position on the start
        self.agent: dict = {}

        # set basic members and reset the environment
        self.env = env
        initial_observation, info = env.reset(seed=None)
        print(f"Environment reset\ninfo: {info}")
        grid = initial_observation["environment"]
        self.height, self.width = grid.shape
        target = initial_observation["target"]
        
        self.agent["i"] = initial_observation["agent"][0]
        self.agent["j"] = initial_observation["agent"][1]

        
        # create state maps
        self._make_state_maping(grid, self.height, self.width)

        # set up basic reward system for the navigating in the maze
        self.reward = np.ndarray([self.state_count], dtype=float)
        self.reward.fill(-1)
        self.target_state = self._state_idx(target[0], target[1])
        self.reward[self.target_state] = 0 

        # state transitions (also sets the starting policy)
        self._make_state_transition_tables(grid)
        self.current = np.ndarray([self.state_count], dtype=float)
        self.current.fill(0)

        self.next = np.ndarray([self.state_count], dtype=float)
        self.next.fill(0)


    #######################################################
    # public interface 

    def solve(self, discount: float, eval_max: int, iter_max: int):
        self._policy_iter(discount, eval_max, iter_max)
        input("Press [ENTER] to continue...")
        os.system("clear")
        # print(f"{self.current}\n{self.policy}")
        terminated = False
        i = 0
        acc: float = 0.0
        observation = {"agent": [self.agent["i"], self.agent["j"]]}
        while not terminated and i < self.width * self.height:
            # print(f"Agent index {self._state_idx(agent_i, agent_j,self._state_coord_map)} at position {agent_i}, {agent_j}, step {i}")
            move = "up"
            agent_i = observation["agent"][0]
            agent_j = observation["agent"][1]
            for m in self.moveset:
                if self.policy[m][self._state_idx(agent_i, agent_j)] > 0:
                    move = m
                    break

            observation, reward, terminated, _, info = self.env.step(move)
            acc += reward
            print(f"info: {info}, reward: {reward}, terminal: {terminated}\nTotal reward: {acc}")
            i += 1
            if not terminated:
                sleep(0.25)
                os.system("clear")

    #######################################################
    # inner interface
    def _make_state_maping(self, grid: np.ndarray, height: int, width: int):
        self.state_count = 0
        for i in range(1, height-1):
            for j in range(1, width-1):
                if grid[i,j]:
                    self.state_coord_map[self.state_count] = {"i": i, "j": j}
                    self.state_count += 1

    def _make_state_transition_tables(self, grid: np.ndarray):
        # set up the action based values 
        for move in self.moveset:
            self.probabilities[move] = np.ndarray([self.state_count, self.state_count], dtype=float)
            # set up with zeroes
            self.probabilities[move].fill(0)

            # transition directions
            di = self.moveset[move]["i"]
            dj = self.moveset[move]["j"]
            for state in self.state_coord_map:
                to_i = self.state_coord_map[state]["i"] + di
                to_j = self.state_coord_map[state]["j"] + dj
                if state != self.target_state:
                    if grid[to_i,to_j] == 1:
                        # at this point, we have a transition from 
                        to_state = self._state_idx(to_i, to_j)
                        self.probabilities[move][state, to_state] = 1
                    else:
                        # if the cell in the direction is 0, we add a default transition back into itself
                        self.probabilities[move][state, state] = 1
            # set up the starting uniform random policy
            self.policy[move] = np.ndarray([self.state_count], dtype=float)
            # start with uniform random policy
            self.policy[move].fill(0.25)

    def _policy_eval(self, discount: float, iter_count: int):
        for i in range(iter_count):
            # a single iteration step
            for move in self.moveset:
                # use the policy vector instead of the hardcoded ones
                # note that this way of multiplying vectors is mathematically illegal
                # a state x state matrix with the policy in the diagonal would've been 
                # more elegant, but python can do this operation, so I'm leaving it like this
                self.next += self.policy[move] * (self.reward + discount * np.dot(self.probabilities[move], self.current))
            for idx in range(self.state_count):
                self.current[idx] = self.next[idx]

    def _policy_iter(self, discount: float, eval_count: int, iter_count: int):
        i: int = 0
        while not self._found_optimal() and i < iter_count:
            self._policy_eval(discount, eval_count)
            self._policy_improve()
            i += 1
        print(f"Found optimal policy after {i} iterations")

    def _found_optimal(self) -> bool:
        for state in range(self.state_count):
            # quick and dirty check: if it's not the final state, there can only be zeros
            # and ones in the policy matrix
            if state != self.target_state:
                for move in self.moveset:
                    if not (self.policy[move][state] == 1.0 or self.policy[move][state] == 0.0):
                        return False
        return True

    def _policy_improve(self):
        for state in range(self.state_count):
            for move in self.moveset:
                if np.dot(self.probabilities[move][state], self.current) > self.current[state]:
                    # improve:
                    self.policy[move][state] = 1.0
                    for m in self.moveset:
                        if m != move:
                            self.policy[m][state] = 0.0
    
    def _state_idx(self,i: int, j: int) -> int:
        # find map entry, where i,j are of the target
        for state in self.state_coord_map:
            if self.state_coord_map[state]['i'] == i and self.state_coord_map[state]['j'] == j:
                return state
        return 0
