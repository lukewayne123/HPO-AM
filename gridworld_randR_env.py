"""
Gridworld is simple 4 times 4 gridworld from example 4.1 in the book: 
    Reinforcement Learning: An Introduction
@author: pinghsieh
@Most of the code was originally borrowed from: https://github.com/podondra/gym-gridworlds

"""

import gym
from gym import spaces
import numpy as np

class Gridworld_RandReward_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_Env, self).__init__()
        
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(15)
        
        self.gridworld = np.arange(
                self.observation_space.n + 1
                ).reshape((4, 4))
        self.gridworld[-1, -1] = 0
        
        # state transition matrix
        
        # 4 *N=15 ��N=15 state?
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:-1]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                #d means direction
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 0

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        #action should pick one of  [(-1, 0), (0, 1), (1, 0), (0, -1)]
        # from self.observation_space.n=15 draw 1 if some state is inaccessible, then p must be 0.
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -2.4 + 4.4 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        self.obs = next_obs
        info ={}
        # return next_obs, reward, done, None
        return next_obs, reward, done, info
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs
        
    def render(self):
        return None
    
class Gridworld_RandReward_3x3_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_3x3_Env, self).__init__()
        
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(9)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((3, 3))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 2))
                next_col = max(0, min(col + d[1], 2))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -2.4 + 4.4 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        self.obs = next_obs
        return next_obs, reward, done, None
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs
        
    def render(self):
        return None
    
class Gridworld_RandReward_4x4_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_4x4_Env, self).__init__()
        self.distance = 0
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(16)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((4, 4))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            # a = 0 1 2 self.action_space.n-1
            # d = (-1, 0), (0, 1), (1, 0), (0, -1)
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        
        
        
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # print("self.observation_space.n",self.observation_space.n)
        # print("self.P[action, self.obs, :].flatten():",self.P[action, self.obs, :].flatten())
        # print("self.obs",self.obs)
        # print("next_obs",next_obs)
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        # reward = self.R[action,self.obs]
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        # next_obs = 0 if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        info = {}
        return next_obs, reward, done, info
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        self.distance = ( int(self.obs/4) + self.obs%4 )
        return self.obs
        
    def render(self):
        return None
    
class Gridworld_RandReward_5x5_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_5x5_Env, self).__init__()
        
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # observation is between 0 and 24
        self.observation_space = spaces.Discrete(25)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((5, 5))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 4))
                next_col = max(0, min(col + d[1], 4))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        return next_obs, reward, done, None
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        return self.obs
        
    def render(self):
        return None    
class Gridworld_FixedReward_4x4_Env(gym.Env):

    def __init__(self):
        super(Gridworld_FixedReward_4x4_Env, self).__init__()
        self.distance = 0 
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(16)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((4, 4))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            # a = 0 1 2 self.action_space.n-1
            # d = (-1, 0), (0, 1), (1, 0), (0, -1)
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 3))
                next_col = max(0, min(col + d[1], 3))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5
        self.R[0, 4] = 5
        self.R[3, 1] = 5
        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # print("self.observation_space.n",self.observation_space.n)
        # print("self.P[action, self.obs, :].flatten():",self.P[action, self.obs, :].flatten())
        # print("self.obs",self.obs)
        # print("next_obs",next_obs)
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        # reward = -1.2 + 2.2 * np.random.randint(0, 2)
        reward = float (self.R[action,self.obs])
        dist_old = ( int(self.obs/4) + self.obs%4 )
        dist_new = ( int(next_obs/4) + next_obs%4 )
        if dist_old == dist_old:
            # reward = float(-10)
            reward = float(-1.0)
        else:
            reward = float( -( dist_new - dist_old ) )
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        # next_obs = 0 if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        info = {}
        return next_obs, reward, done, info
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        self.distance = ( int(self.obs/4) + self.obs%4 )
        return self.obs
        
    def render(self):
        return None

class Gridworld_RandReward_8x8_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_8x8_Env, self).__init__()
        self.distance = 0
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(64)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((8, 8))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            # a = 0 1 2 self.action_space.n-1
            # d = (-1, 0), (0, 1), (1, 0), (0, -1)
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 7))
                next_col = max(0, min(col + d[1], 7))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        
        
        
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # print("self.observation_space.n",self.observation_space.n)
        # print("self.P[action, self.obs, :].flatten():",self.P[action, self.obs, :].flatten())
        # print("self.obs",self.obs)
        # print("next_obs",next_obs)
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        # reward = self.R[action,self.obs]
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        # next_obs = 0 if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        info = {}
        return next_obs, reward, done, info
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        self.distance = ( int(self.obs/8) + self.obs%8 )
        return self.obs
        
    def render(self):
        return None
class Gridworld_RandReward_6x6_Env(gym.Env):

    def __init__(self):
        super(Gridworld_RandReward_6x6_Env, self).__init__()
        self.distance = 0
        self.reward_range = (-1, 0)
        self.action_space = spaces.Discrete(4)
        # although there are 2 terminal squares in the grid
        # they are considered as 1 state
        # therefore observation is between 0 and 14
        self.observation_space = spaces.Discrete(36)
        
        self.gridworld = np.arange(
                self.observation_space.n
                ).reshape((6, 6))
        #self.gridworld[-1, -1] = 0
        
        # state transition matrix
        self.P = np.zeros((self.action_space.n,
                              self.observation_space.n,
                              self.observation_space.n))
        
        # any action taken in terminal state has no effect
        self.P[:, 0, 0] = 1

        for s in self.gridworld.flat[1:self.observation_space.n]:
            row, col = np.argwhere(self.gridworld == s)[0]
            # a = 0 1 2 self.action_space.n-1
            # d = (-1, 0), (0, 1), (1, 0), (0, -1)
            for a, d in zip(
                    range(self.action_space.n),
                    [(-1, 0), (0, 1), (1, 0), (0, -1)]
                    ):
                next_row = max(0, min(row + d[0], 5))
                next_col = max(0, min(col + d[1], 5))
                s_prime = self.gridworld[next_row, next_col]
                self.P[a, s, s_prime] = 1

        self.R = np.full((self.action_space.n,
                             self.observation_space.n), -1)
        self.R[:, 0] = 5

        # Initialize the state arbitrarily 
        self.obs = 1
        
    def step(self, action):
        
        
        
        next_obs = np.random.choice(self.observation_space.n, 1, p=self.P[action, self.obs, :].flatten())
        # print("self.observation_space.n",self.observation_space.n)
        # print("self.P[action, self.obs, :].flatten():",self.P[action, self.obs, :].flatten())
        # print("self.obs",self.obs)
        # print("next_obs",next_obs)
        #if next_obs == 0:
        #    reward = 1.0
        #else:
        reward = -1.2 + 2.2 * np.random.randint(0, 2)
        # reward = self.R[action,self.obs]
        done = True if next_obs == 0 else False
        next_obs = None if next_obs == 0 else next_obs    # Terminal state == None
        # next_obs = 0 if next_obs == 0 else next_obs    # Terminal state == None
        self.obs = next_obs
        info = {}
        return next_obs, reward, done, info
        
    def reset(self):
        # Reset the state uniformly at random
        #self.obs = 6
        self.obs = np.random.randint(1, self.observation_space.n, size=1)
        self.distance = ( int(self.obs/6) + self.obs%6 )
        return self.obs
        
    def render(self):
        return None