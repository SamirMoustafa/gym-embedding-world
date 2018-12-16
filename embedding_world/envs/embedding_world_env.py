import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from embedding_world.envs.embedding_world_handler import SpaceHandler


class EmbeddingEnv(gym.Env):
    # TODO : Set the goal in this scope

    metadata = {'render.modes': ['human', "rgb_array"]}

    ACTION = []

    def __init__(self, embedding_file=None, epsilon=None):

        if embedding_file and epsilon:
            # load the corpus to gensim model as word to vector
            self.space = SpaceHandler(space_file_path=embedding_file, epslion=0.0015)
            # get the embedding dimension
            self.emb_dim = self.space.emb_dim

            # define pickup and drop down actions
            self.ACTION.append(["pickup", "dropout"])
            # make two action for every dimension example : up and down == dim(0)+1 dim(0)-1
            [self.ACTION.append(["dim(%s)+1" % dim, "dim(%s)-1" % dim]) for dim in range(self.emb_dim)]
            # flatten 2D list of action to 1D list : len len(ACTION) = 2N + 2
            self.ACTION = [j for sub in self.ACTION for j in sub]
            # store epsilon inverse as a space size
            self.space_size = int(epsilon ** (-1))

            # define the n-dimension space size
            self.n_dim_space = tuple([self.space_size for i in range(self.emb_dim)])

            # forward or backward in each dimension
            self.action_space = spaces.Discrete(2 * len(self.n_dim_space) + 2)

            low_i = []
            high_i = []
            # observation is the n-space
            for i in range(self.emb_dim):
                low_i.append(np.zeros(len(self.n_dim_space), dtype=int))
                high_i.append(np.array(self.n_dim_space, dtype=int) - np.ones(len(self.n_dim_space), dtype=int))

            self.observation_space = spaces.Box(np.array(low_i), np.array(high_i))

            # initial condition
            self.state = None
            self.steps_beyond_done = None

            # Simulation related variables.
            self.seed()
            self.reset()

            # Just need to initialize the relevant attributes
            # self._configure()

            print('space size is epsilon inverse %i' % self.space_size)
            print('list of all actions %s' % self.ACTION)
            print(self.n_dim_space)
            print(self.action_space)
            print(self.observation_space)

        else:
            if epsilon is None:
                raise AttributeError("must supply epsilon as float")
            if embedding_file is None:
                raise AttributeError("must supply embedding_file path as (str)")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if isinstance(action, int):
            self.space.move_robot(self.ACTION[action])
        else:
            self.space.move_robot(action)

        # TODO : Handle goal in the handler class
        if np.array_equal(self.space.robot, self.space.goal):
            reward = 1
            done = True
        else:

            # TODO : Handle punishment
            reward = -0.1 / (self.space_size[0] * self.space_size[1])
            done = False

        self.state = self.space.robot

        info = {}

        return self.state, reward, done, info

    def reset(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass


'''
    def take_action(self, action):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError

    def get_seed(self):
        raise NotImplementedError

    def get_close(self):
        raise NotImplementedError
'''


class EmbeddingEnvExample(EmbeddingEnv):
    def __init__(self):
        print(os.getcwd())
        super(EmbeddingEnvExample, self).__init__(
            embedding_file="embedding_world/envs/world_sample/mini.wiki.multi.en.vec", epsilon=0.006)
