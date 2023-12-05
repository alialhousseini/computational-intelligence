from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
import numpy as np
from gym import spaces
import gym
import gymnasium
import math
import random
import time


class NimEnv(gym.Env):
    def __init__(self, num_piles=5):
        super(NimEnv, self).__init__()

        self.max_objects_per_pile = 2*(num_piles-1) - 1

        # Define the observation space and action space
        self.observation_space = gymnasium.spaces.Box(low=0, high=self.max_objects_per_pile,
                                                      shape=(num_piles,), dtype=np.int64)
        self.action_space = gymnasium.spaces.MultiDiscrete(
            [num_piles, self.max_objects_per_pile])

        # Initialize state
        self.state = [2 * i - 1 for i in range(num_piles)]
        self.num_piles = num_piles
        self.player = 0  # 0 or 1 to indicate which player's turn
        self.info = {}

    def switch_player(self):
        return 0 if self.player == 1 else 1

    def calculate_nim_sum(self):
        nim_sum = 0
        for objects in self.state:
            nim_sum ^= objects
        return nim_sum

    def best_move(self):
        nim_sum = self.calculate_nim_sum(self.state)

        if nim_sum == 0:
            # No winning move, return a random valid move
            valid_moves = [(pile, objects)
                           for pile, objects in enumerate(self.state) if objects > 0]
            return valid_moves[np.random.randint(len(valid_moves))]

        # Find a move that makes the nim sum 0
        for pile, objects in enumerate(self.state):
            target = objects ^ nim_sum
            if target < objects:
                return (pile, objects - target)

    def check_end_game(self):
        return all(self.state == 0)

    def step(self, action):  # player 0 AI, player 1 human with nimsum
        # Execute one time step within the environment
        pile, num_objects = action
        reward = 0
        done = False

        # Check if action is valid
        if self.state[pile] < num_objects or num_objects <= 0:
            # Penalize for invalid actions
            print("Invalid action\n")
            reward = -5
            self.info['action'] = action
            self.info['state'] = self.state
            return self.state, reward, done, self.info

        # Perform action player 0
        self.state[pile] -= num_objects

        if self.check_end_game():
            reward = -10  # player 0 loses
            done = True
            print("Player 0 loses\n")
            self.info = {'action_made': action, 'best_move': self.best_move()}
            return self.state, reward, done, self.info

        # Perform action player 1
        self.switch_player()
        pile, n_obj = self.best_move()
        self.state[pile] -= n_obj

        # Check for end game condition
        if self.check_end_game():
            reward = 15  # Win
            done = True
            print("Player 1 Wins!\n")
        else:
            reward = 2
            done = False

        self.info = {'action_made': action, 'best_move': self.best_move()}
        return self.state, reward, done, self.info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = [2 * i - 1 for i in range(self.num_piles)]
        self.player = 0
        self.info = {}
        return self.state

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f"State of the piles: {self.state}")
        print(f'info:{self.info}')
        print(f'####################')


def main():

    # Instantiate the environment
    env = NimEnv(num_piles=5)
    monitored_env = Monitor(env, "C:\\Users\\a_h9\\Desktop\\Monitor")

    ################################# Training#####################################
    # Train the model
    model = A2C('MlpPolicy', monitored_env,
                tensorboard_log="./vnrs/", verbose=1)

    timesteps_per_epoch = 80000

    model.learn(total_timesteps=timesteps_per_epoch)

    # Save the trained model
    model.save("C:\\Users\\a_h9\\Desktop\\Training")

    print("Training Complete")

    ################################# Testing  #####################################

    model = A2C.load("C:\\Users\\a_h9\\Desktop\\Training.zip", env)
    num_episodes = 100
    testing_episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break


if __name__ == "__main__":
    main()
