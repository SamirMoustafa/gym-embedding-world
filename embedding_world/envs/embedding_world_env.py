import gym
from gym import error, spaces, utils
from gym.utils import seeding


class EmbeddingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
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
        raise NotImplementedError

    def render(self, mode='human', close=False):
        raise NotImplementedError

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