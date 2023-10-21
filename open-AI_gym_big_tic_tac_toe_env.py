import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random 
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from Kryds_Bolle_I_Kryds_Bolle import Board

class Big_Tic_tac_Toe_Env(Env):
    def __init__(self):
        self.action_space = Discrete(81)
        self.observation_space = Box(-1,1,(81,))
        self.board = Board((0,0), 0, "black", has_children=True, is_clickable=False)

    def step(self, action):
        self.board.place_piece_on_board(action)
        self.board.get_board_stat()

    def render(self):
        pass

    def reset(self):
        pass