import os

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from embedding_world.envs.embedding_world_handler import SpaceHandler


class EmbeddingEnv(gym.Env):

    metadata = {'render.modes': ['human', "rgb_array"]}
    threshold = 0.01
    ACTION = []

    def __init__(self, embedding_file=None, epsilon=None, goals_as_words=None):

        if embedding_file and epsilon:
            # load the corpus to gensim model as word to vector
            self.space = SpaceHandler(space_file_path=embedding_file, epslion=0.0015)
            # get the embedding dimension
            self.emb_dim = self.space.emb_dim

            # make two action for every dimension example : up and down == dim(0)+1 dim(0)-1
            [self.ACTION.append(["dim(%s)+1" % dim, "dim(%s)-1" % dim]) for dim in range(self.emb_dim)]
            # flatten 2D list of action to 1D list : len len(ACTION) = 2N + 2
            self.ACTION = [j for sub in self.ACTION for j in sub]
            # store epsilon inverse as a space size
            self.space_size = int(epsilon ** (-1))

            # define the n-dimension space size
            self.n_dim_space = tuple([self.space_size for i in range(self.emb_dim)])

            # forward or backward in each dimension
            self.action_space = spaces.Discrete(2 * len(self.n_dim_space))

            low_i = []
            high_i = []
            # observation is the n-space
            for i in range(self.emb_dim):
                low_i.append(np.zeros(len(self.n_dim_space), dtype='float64'))
                high_i.append(np.array(self.n_dim_space, dtype='float64') - np.ones(len(self.n_dim_space), dtype='float64'))

            self.observation_space = spaces.Box(np.array(low_i), np.array(high_i))

            # initial condition
            self.state = None
            self.steps_beyond_done = None

            # Simulation related variables.
            self.seed()
            self.reset()
            '''
            # Just need to initialize the relevant attributes
            self._configure()

            print('space size is epsilon inverse %i' % self.space_size)
            print('list of all actions %s' % self.ACTION)
            print(self.n_dim_space)
            print(self.action_space)
            print(self.observation_space)
            
            '''
        else:
            if epsilon is None:
                raise AttributeError("must supply epsilon as float")
            if embedding_file is None:
                raise AttributeError("must supply embedding_file path as (str)")

    def handle_goals(self,phrase):
        # set goals
        self.space.set_goals(phrase.split())
        # get goal
        self.goals_as_vetors = self.space.get_goals()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        print(self.space.robot[0])

        if isinstance(action, int):
            self.space.move_robot(self.ACTION[action])
        else:
            self.space.move_robot(action)

        print(self.space.robot[0])
        #print(self.goals_as_vetors[0])

        print(self.space.robot)

        # check if the vector of robot is near to the target(desired) vector
        if (np.abs(self.space.robot - self.goals_as_vetors[0]) <= self.threshold).all():
            print("I found suitable vector")
            if len(self.goals_as_vetors) is 1:

                # the phrase end
                reward = 1
                done = True
            else:
                # match one word
                self.goals_as_vetors.pop(0)
                reward = .5
                done = False
        else:
            print("I made a bad step.. I'm sorry")
            reward = -0.1 / self.space_size
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
