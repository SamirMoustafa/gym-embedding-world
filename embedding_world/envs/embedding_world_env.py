import atari_py
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from embedding_world.envs.embedding_world_handler import SpaceHandler


class EmbeddingEnv(gym.Env):
    def __init__(self):

        self.metadata = {'render.modes': ['human', "rgb_array"]}

        # simulate environment to be like a game(pong)
        self.ale = atari_py.ALEInterface()
        self.game_path = atari_py.get_game_path('pong')

        # initializing
        self.number_of_words_to_trans = 0
        self.ACTION = [['pick-up']]
        self.phrase, self.target = None, None
        self.in_production_mood = False
        self.initial_for_reset = None
        self.done = False

    def set_normalization(self, from_normalize, to_normalize):
        self.from_normalize = from_normalize
        self.to_normalize = to_normalize

    def set_paths(self, embedding_from_file, embedding_to_file):

        # load the corpus to gensim model as word to vector
        self.space = SpaceHandler(space_file_path_from=embedding_from_file,
                                  space_file_path_to=embedding_to_file)

        # get epsilon
        self.epsilon = self.space.get_epsilon()

        # get the embedding dimension
        self.emb_dim = self.space.emb_dim

        # make two action for every dimension example : up and down == dim(0)+1 dim(0)-1
        [self.ACTION.append(["dim(%s)+1" % dim, "dim(%s)-1" % dim]) for dim in range(self.emb_dim)]
        # flatten 2D list of action to 1D list : len len(ACTION) = 2N + 1
        self.ACTION = [j for sub in self.ACTION for j in sub]

        # store epsilon inverse as a space size
        self.space_size = int(self.epsilon ** (-1))

        # define the n-dimension space size
        self.n_dim_space = tuple([self.space_size for i in range(self.emb_dim)])

        # forward or backward in each dimension
        self.action_space = spaces.Discrete(2 * len(self.n_dim_space) + 1)

        # observation is the n-space
        low_i, high_i = [], []
        for i in range(self.emb_dim):
            low_i.append(np.zeros(len(self.n_dim_space), dtype='float64'))
            high_i.append(np.array(self.n_dim_space, dtype='float64') - np.ones(len(self.n_dim_space), dtype='float64'))

        self.observation_space = spaces.Box(np.array(low_i), np.array(high_i), dtype='float64')

    def set_sentences(self, phrase, target):
        # Check that handler is loaded
        if self.space is None:
            raise ValueError("use set_paths(embedding_from_file, embedding_to_file) to set your corpses paths.")

        self.phrase, self.target = phrase, target
        self.space.handle_initial_and_goal(self.from_normalize(self.phrase).split(),
                                           self.to_normalize(self.target).split())

        # initial condition
        self.initial_for_reset = self.space.initial_matrix[0]
        self.state = None
        self.steps_beyond_done = None

        # get goal
        self.goals_as_vectors = self.space.get_goals()

        # get initial
        self.initial_as_vetors = self.space.get_initial()

        # Put the number of words that will translated
        self.number_of_words_to_trans = len(self.phrase.split())

        # Simulation related variables.
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.ale.loadROM(self.game_path)
        return [seed]

    def step(self, action):
        """Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the environment

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        if self.in_production_mood:
            return self.__step_in_production__(action)
        else:
            # Handle the error input
            if self.target is None or self.target == '':
                raise ValueError("use set_sentnce(phrase, target) to set your sentences or use production_is_on()")
            return self.__step_in_training__(action)

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

                Returns: observation (object): the initial observation of the
                    space.
        """
        self.state = self.initial_for_reset
        self.done = False
        return self.state

    def render(self, mode='human', close=False):
        # TODO : Make PCA on the N-dimension embedding and show them in a grid world
        pass

    def close(self):
        del self.space

    def production_is_on(self):
        self.in_production_mood = True

    def production_is_off(self):
        self.in_production_mood = False

    def get_initial_state(self):
        return self.space.current_pos

    def __step_in_production__(self, action):
        # default values for production
        info = {"ale.lives": self.ale.lives()}
        reward = 0

        if self.done:
            return self.space.current_pos, 0, self.done, info

        if action == 0:
            self.number_of_words_to_trans -= 1
            info['trans'] = self.space.get_word_from_vec(self.space.current_pos)

            if self.number_of_words_to_trans > 0:
                self.space.__residual_vectors__()
                self.done = False
                reward = .5
            else:
                self.done = True
                reward = 1

            return self.space.current_pos, reward, self.done, info

        else:
            state = self.__move_robot__(action)
            return state, reward, self.done, info

    def __step_in_training__(self, action):

        # Save last position before taking action
        past_pos = self.space.current_pos

        # Take an action
        current_pos = self.__move_robot__(action)

        # Default info for game
        info = {"ale.lives": self.ale.lives()}

        # Check the robot is in the boundary
        if not self.space.is_robot_in_region:
            return past_pos, -1, False, info

        # Check that there is remaining words
        if self.__number_of_remain_words__() == 0:
            self.done = True
            return past_pos, 0, self.done, info

        # Check that the goal accomplished one time only
        if self.done:
            return current_pos, 0, True, info

        # Check if the goal had accomplished
        if len(self.goals_as_vectors) == 0:
            self.done = True
            return current_pos, 1, self.done, info

        # define difference between current position and current goal position
        difference = np.abs(current_pos - self.get_current_goal)
        # Check the distance between robot and check pick up action is taken
        if (difference <= self.epsilon).all() and (action == [0] or action == 0):
            # Robot capture something right
            if self.__number_of_remain_words__() > 1:
                # The robot pick-up a word correctly
                self.space.__residual_vectors__()
                self.__remove_first_vector_from_goal__()
                reward = .5
                self.done = False
                # Give a positive reward and stay in current location
                return current_pos, reward, self.done, info
            else:
                # The phrase end at right position
                reward = 1
                self.done = True
                self.__remove_first_vector_from_goal__()
                return current_pos, reward, self.done, info

        else:
            # Give a negative reward
            reward = -round(np.sqrt(np.sum(difference ** 2)), 5)
            self.done = False
            return current_pos, reward, self.done, info

    def __move_robot__(self, action):
        if isinstance(action, int) or isinstance(action, (np.ndarray, np.generic)):
            return self.space.move_robot(self.ACTION[int(action)])
        else:
            return self.space.move_robot(action)

    def __remove_first_vector_from_goal__(self):
        self.goals_as_vectors.pop(0)

    def __number_of_remain_words__(self):
        count = 0
        for i in self.goals_as_vectors:
            if sum(i) != 0:
                count += 1
        return count

    @property
    def env(self):
        return self

    @property
    def get_current_goal(self):
        return self.goals_as_vectors[0]


class EmbeddingEnvEvaluate(EmbeddingEnv):
    def __init__(self):
        super(EmbeddingEnvEvaluate, self).__init__()


if __name__ == "__main__":
    raise BaseException("Can't run the script \nTry to use Gym to run embedding_world environment")
