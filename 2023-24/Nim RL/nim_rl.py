from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.monitor import Monitor
import numpy as np
from gym import spaces
import gym
import math
import random
import time


class NimEnv(gym.Env):
    def __init__(self, num_piles=5):
        super(NimEnv, self).__init__()

        self.max_objects_per_pile = 2*(num_piles-1) - 1

        self.state = [2 * i + 1 for i in range(num_piles)]

        # Define the observation space and action space
        self.observation_space = spaces.Box(low=0, high=self.max_objects_per_pile,
                                            shape=(num_piles,), dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(
            [num_piles, self.max_objects_per_pile])

        # Initialize
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
        nim_sum = self.calculate_nim_sum()

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
        return all(element == 0 for element in self.state)

    def step(self, action, skip_second_player=0):  # player 0 AI, player 1 human with nimsum
        # Execute one time step within the environment
        pile, num_objects = action
        reward = 0
        done = False

        # Check if action is valid
        if self.state[pile] < num_objects or num_objects <= 0:
            # Penalize for invalid actions
            print(f"Invalid action {(pile, num_objects)}\n")
            reward = -5
            return self.state, reward, done, {}

        # Perform action player 0
        self.state[pile] -= num_objects
        if skip_second_player == 0:
            print(f"Player 0 played {(pile, num_objects)}\n")

        if self.check_end_game():
            reward = 15  # player 0 WINS
            done = True
            print("Player 0 Wins!\n")
            return self.state, reward, done, {}

        if skip_second_player == 0:
            # Perform action player 1
            self.switch_player()
            valid = False
            while not valid:
                pile = random.choice(range(len(self.state)))  # Random move
                n_obj = random.choice(range(self.state[pile]+1))
                if n_obj > 0 and self.state[pile] >= n_obj:
                    valid = True
            # pile, n_obj = self.best_move()
            print(f"Player 1 played {(pile, n_obj)}\n")
            self.state[pile] -= n_obj

        # Check for end game condition
        if self.check_end_game():
            reward = -10  # Win player 1
            done = True
            print("Player 0 LOSES!\n")
        else:
            reward = 1
            done = False

        return self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = [2 * i + 1 for i in range(self.num_piles)]
        self.player = 0
        self.info = {}
        return self.state

    def render(self, mode='human'):
        # Render the environment to the screen
        print(f"State of the piles: {self.state}")
        print(f'####################')


def main():

    # Instantiate the environment
    env = NimEnv(num_piles=5)
    monitored_env = Monitor(env, "C:\\Users\\a_h9\\Desktop\\Monitor5A")

    ################################# Training#####################################
    # Train the model
    # model = A2C('MlpPolicy', monitored_env,
    #             tensorboard_log="./vnrs/", verbose=1)

    # timesteps_per_epoch = 240000

    # model.learn(total_timesteps=timesteps_per_epoch)

    # # Save the trained model
    # model.save("C:\\Users\\a_h9\\Desktop\\Training5A")

    # print("Training Complete")

    ################################# Testing  #####################################

    model = A2C.load("C:\\Users\\a_h9\\Desktop\\Training5A.zip", env)
    num_episodes = 500
    data = []  # to be used for processing
    for episode in range(num_episodes):
        temp_data = []
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            temp_data.append({"state": np.array(obs), "action": action})
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
            if done:
                break
        if reward == 15:
            data.extend(temp_data)

    ################################# XAI  #####################################

    import pandas as pd
    df = pd.DataFrame(data)
    df.to_csv("C:\\Users\\a_h9\\Desktop\\data5AA.csv", index=False)

    ################################# Playing (Not completed)  #####################################

    # model = A2C.load("C:\\Users\\a_h9\\Desktop\\Training.zip",
    #                  env)
    # obs = env.reset()
    # who_starts = input("Do you want to start? (y/n): ")
    # print("\n Your are player 1")
    # done = False
    # if who_starts == 'y':
    #     while not done:
    #         action = input("Enter your move (pile, num_objects): ")
    #         action = tuple(map(int, action.split(',')))
    #         obs[action[0]] -= action[1]

    #         env.render()

    #         action, _ = model.predict(obs)

    #         print("Machine Turns: Please wait...\n")
    #         print(f"Machine Played {action}\n")
    #         obs, reward, done, _ = env.step(action, skip_second_player=1)
    #         env.render()

    # else:
    #     player = 0


if __name__ == "__main__":
    main()
