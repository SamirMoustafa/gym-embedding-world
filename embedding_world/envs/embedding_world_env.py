import gym
import os
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding

from gensim.models import KeyedVectors

class EmbeddingEnv(gym.Env):
    metadata = {'render.modes': ['human', "rgb_array"]}

    ACTION = ["N", "S", "E", "W"]

    def __init__(self,embedding_file=None):

        # load the corpus to gensim model as word to vector
        word2vec = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
        # get the embedding dimension
        self.emb_dim = word2vec.wv.vector_size


        '''
        if embedding_file:
            self.embedding = MazeView2D(maze_name="OpenAI Gym - Maze (%s)" % maze_file,
                                        maze_file_path=maze_file,
                                        screen_size=(640, 640))
        elif maze_size:
            if mode == "plus":
                has_loops = True
                num_portals = int(round(min(maze_size) / 3))
            else:
                has_loops = False
                num_portals = 0

            self.maze_view = MazeView2D(maze_name="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                        maze_size=maze_size, screen_size=(640, 640),
                                        has_loops=has_loops, num_portals=num_portals)
        else:
            raise AttributeError("One must supply either a maze_file path (str) or the maze_size (tuple of length 2)")

        self.maze_size = self.maze_view.maze_size

        #raise FileNotFoundError("Cannot find %s." % embedding_file)
        '''
        pass

    def step(self, action):
        """
        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        raise NotImplementedError

    def seed(self, **kwargs):
        pass

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
        super(EmbeddingEnvExample, self).__init__(embedding_file="embedding_world/envs/world_sample/mini.wiki.multi.en.vec")