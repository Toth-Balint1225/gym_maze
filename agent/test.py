
from simple_agent import SimpleAgent
import gymnasium as gym

def main():
    gym.logger.set_level(40)
    env = gym.make("gym_maze/MazeWorld-v0", 
                   node_space_width=10, 
                   node_space_height=10, 
                   loops=True, num_loop=5)
    agent = SimpleAgent(env)
    agent.solve(1.0, 10, 100)

if __name__ == '__main__':
    main()
