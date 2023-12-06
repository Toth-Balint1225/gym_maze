######################################
#        Tóth Bálint (PME4BQ)        #
# Artificial Intelligence Laboratory #
#               2023                 #
######################################

import numpy as np
import random as rng

class WrongSize(Exception):
    def __init__(self):
        super().__init__("Height and width must be odd")


class Maze:
    def __init__(self, height: int, width: int, seed: int, loops: bool = False, num_loop: int = 2):
        if (width % 2 == 0 or height % 2 == 0):
            raise WrongSize()
        self._height = height
        self._width = width
        self._grid = np.ndarray([height, width], dtype=np.int8)
        self._grid.fill(0)
        # generate the grid
        # border

        #for i in range(height):
        #    self._grid[i, width-1] = 0
        #    self._grid[i, 0] = 0
        #for j in range(width):
        #    self._grid[height-1, j] = 0
        #    self._grid[0, j] = 0

        # stack will be a list
        # visited will also be a list
        self._stack = []
        self._visited = 1
        self._stack.insert(0, {"i": 1, "j": 1})

        # seed the RNG
        rng.seed(seed)

        # iterate
        while self._visited < ((width -1) * (height - 1)) / 4:
            next = self._get_next()
            if not next:
                # print(f"backtrack on {self._stack[0]}")
                self._stack.pop(0)
            else:
                # print(f"visiting {next}")
                self._visit(next)
                # print(f"Visited: {self._visited}")
        
        # introduce some loops
        if loops:
            # get some random walls, that can be made into loops
            wall_count: int = 0
            iter_count: int = 0
            while wall_count < num_loop and iter_count < 100:
                iter_count += 1
                # get a random position
                i: int = rng.randrange(1,height-1)
                j: int = rng.randrange(1,width-1)
                if self._is_good_for_gap(i, j):
                    #print(f"gap {i}, {j}")
                    self._grid[i,j] = 1
                    wall_count += 1


    def _is_good_for_gap(self, i: int, j: int) -> bool:
        # if it's not a wall, we fail
        if self._is_visited(i, j):
            return False

        # has exactly 2 neighbouring paths
        nc: int = 0
        if self._is_visited(i+1, j):
            nc += 1
        if self._is_visited(i, j+1):
            nc += 1
        if self._is_visited(i-1, j):
            nc += 1
        if self._is_visited(i, j-1):
            nc += 1
        
        if nc == 2:
            # need to check if the paths are on the opposite side
            if self._is_visited(i+1, j) and self._is_visited(i-1, j) or self._is_visited(i, j+1) and self._is_visited(i, j-1):
                return True
        else: 
            return False



    def _is_visited(self, i: int, j: int) -> bool:
        return self._grid[i, j] == 1
            
    
    def _valid_cell(self, i: int, j: int) -> bool:
        if i < 0 or i >= self._height:
            return False
        if j < 0 or j >= self._width:
            return False
        return True
    
    def _valid_nexts(self, tos: dict) -> list:
        i: int = tos["i"]
        j: int = tos["j"]
        nexts = []
        if (self._check(i, j+2)):
            nexts.append({"i": i, "j": j+2})
        if (self._check(i+2, j)):
            nexts.append({"i": i+2, "j": j})
        if (self._check(i-2, j)):
            nexts.append({"i": i-2, "j": j})
        if (self._check(i, j-2)):
            nexts.append({"i": i, "j": j-2})
        return nexts
    
    def _check(self, i: int, j: int) -> bool:
        if self._valid_cell(i, j) and (not self._is_visited(i, j)):
            return True
        else:
            return False

    def _get_next(self) -> dict:
        valids = self._valid_nexts(self._stack[0])
        if (len(valids) == 0):
            return {}
        if (len(valids) == 1):
            return valids[0]
        return valids[rng.randrange(0, len(valids))]

    def _link(a: int, b: int) -> int:
        delta: int = a - b
        if delta == 0:
            return a
        elif delta > 0:
            return b + (delta - 1)
        else:
            return a + ((-1*delta) - 1) 
    
    def _visit(self, next_cell: dict):
        base_cell: dict = self._stack[0]
        # get the deltas and set both to 1
        self._grid[next_cell["i"], next_cell["j"]] = 1
        self._grid[base_cell["i"], base_cell["j"]] = 1
        self._grid[Maze._link(base_cell["i"],next_cell["i"]), Maze._link(base_cell["j"], next_cell["j"])] = 1
        # add the link
        self._stack.insert(0, next_cell)
        self._visited += 1
    
        


    def get_grid(self) -> np.ndarray[np.int8]:
        return self._grid

    def __str__(self) -> str:
        res = ""
        # top line
        res += "+"
        for i in range(self._width * 2):
            res += "-"
        res += "+\n"

        for i in range(self._height):
            res += "|"
            for j in range(self._width):
                if self._grid[i,j] == 1:
                    res += "  "
                else:
                    res += "XX"
            res += "|\n"

        # bottom line
        res += "+"
        for i in range(self._width * 2):
            res += "-"
        res += "+" # last newline is implicit
        return res

    def pretty_print(self, agent: dict, target: dict):
        # top line
        print("+", end="")
        for i in range(self._width * 2):
            print("-", end="")
        print("+")

        for i in range(self._height):
            print("|", end="")
            for j in range(self._width):
                if self._grid[i,j] == 1:
                    if i == agent["i"] and j == agent["j"]:
                        print("AA",end="")
                    elif i == target["i"] and j == target["j"]:
                        print("TT",end="")
                    else:
                        print("  ", end="")

                else:
                    print("XX", end="")
            print("|")

        # bottom line
        print("+", end="")
        for i in range(self._width * 2):
            print("-", end="")
        print("+")
    
    # valid moves as up down left or right from that cell
    def valid_moves(self, pos: dict) -> list:
        # if we're in a wall, can't go anywhere
        i: int = pos["i"]
        j: int = pos["j"]
        if not self._is_visited(i,j):
            return []
        res = []
        # check for all four directions, return the list of valid ones
        if self._is_visited(i+1, j):
            res.append("down")
        if self._is_visited(i, j+1):
            res.append("right")
        if self._is_visited(i, j-1):
            res.append("left")
        if self._is_visited(i-1, j):
            res.append("up")
        
        return res
        

    
    # give back a list of positions that the agent can "see" from its position 
    def visual_range(self, pos: dict) -> list:
        pass

# 
# import time
# def main():
#     maze = Maze(25, 25, time.time(), loops=True, num_loop=5)
#     print(maze) 
# 
# if __name__ == '__main__':
#     main()
