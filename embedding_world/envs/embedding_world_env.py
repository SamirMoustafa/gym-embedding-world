import re
import gym
import atari_py
import numpy as np
from gym import spaces
from gym.utils import seeding
from embedding_world.envs.embedding_world_handler import SpaceHandler


def normalize_arabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    return(text)


class EmbeddingEnv(gym.Env):

    phrase, target = None, None
    in_production_mood = False
    metadata = {'render.modes': ['human', "rgb_array"]}
    ACTION = [['pick-up']]
    done = False

    def __init__(self, embedding_from_file=None, embedding_to_file=None):

        # simulate environment to be like a game(pong)
        self.ale = atari_py.ALEInterface()
        self.game_path = atari_py.get_game_path('pong')

        if embedding_from_file and embedding_to_file:
            self.__set_paths__(embedding_to_file, embedding_from_file)
        else:
            pass

    def __set_sentences__(self, phrase, target):
        self.phrase, self.target = phrase, target

    def __set_paths__(self,embedding_from_file=None,embedding_to_file=None):

        if self.phrase is None or self.target == None:
            raise ValueError("use __set_sentnce__(phrase, target) to set your sentences")

        # load the corpus to gensim model as word to vector
        self.space = SpaceHandler(space_file_path_from=embedding_from_file,
                                  space_file_path_to=embedding_to_file,
                                  initial_words_list=self.phrase.lower().split(),
                                  goal_words_list=normalize_arabic(self.target).split())

        # get epsilon
        self.epsilon = self.space.get_epsilon()

        # get goal
        self.goals_as_vectors = self.space.get_goals()

        # get initial
        self.initial_as_vetors = self.space.get_initial()

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

        low_i = []
        high_i = []
        # observation is the n-space
        for i in range(self.emb_dim):
            low_i.append(np.zeros(len(self.n_dim_space), dtype='float64'))
            high_i.append(np.array(self.n_dim_space, dtype='float64') - np.ones(len(self.n_dim_space), dtype='float64'))

        self.observation_space = spaces.Box(np.array(low_i), np.array(high_i), dtype='float64')

        # initial condition
        self.state = None
        self.steps_beyond_done = None

        # Simulation related variables.
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.ale.loadROM(self.game_path)
        return [seed]

    def __step_in_production__(self, action):
        reward = 0
        # default info for game
        info = {"ale.lives": self.ale.lives()}
        if action == 0:
            reward = .5
            info['trans'] = self.space.get_word_from_vec(self.space.current_pos)
            try:
                self.space.residual_vectors()
            except IndexError:
                reward, self.done = 1, True
                self.space.stop()
        else:
            self.__move_robot(action)

        state = self.space.current_pos
        return state, reward, self.done, info

    def __step_in_training__(self, action):
        past_pos = self.space.current_pos
        self.__move_robot(action)

        # default info for game
        info = {"ale.lives": self.ale.lives()}

        if self.number_of_remain_words == 0:
            return past_pos, 0, True, info

        # define difference between current position and current goal position
        difference = np.abs(self.space.current_pos - self.get_current_goal)
        #                                           pick up action taken
        if (difference <= self.epsilon).all() and (action == [0] or action == 0):
            if self.number_of_remain_words == 1:
                # the phrase end
                reward = 1
                self.done = True
                self.__remove_first_vector_from_goal()
            else:
                reward = .5
                self.space.residual_vectors()
                self.__remove_first_vector_from_goal()
                self.done = False
            return self.space.current_pos.tolist(), reward, self.done, info

        else:
            reward = -round(self.emb_dim * np.sqrt(np.sum(difference ** 2)), 5)
            self.done = False

        self.state = self.space.current_pos

        if self.number_of_remain_words == 0:
            self.done = True

        return self.state.tolist(), reward, self.done, info



    def step(self, action):
        if self.in_production_mood is True:
            return self.__step_in_production__(action)
        else:
            return self.__step_in_training__(action)

    def __move_robot(self, action):
        if isinstance(action, int) or isinstance(action, (np.ndarray, np.generic)):
            self.space.move_robot(self.ACTION[int(action)])
        else:
            self.space.move_robot(action)

    def reset(self):
        self.space.reset_robot()
        self.state = self.space.get_goals()[0].tolist()
        self.done = False
        return self.state

    def render(self, mode='human', close=False):
        pass

    def close(self):
        del self.space

    def production_is_on(self):
        self.in_production_mood = True

    def __remove_first_vector_from_goal(self):
        self.goals_as_vectors.pop(0)

    @property
    def env(self):
        return self

    @property
    def get_current_goal(self):
        return self.goals_as_vectors[0].tolist()

    @property
    def number_of_remain_words(self):
        # Normalize the goal matrix and check if it's equal to zeros or not
        if (np.array(self.goals_as_vectors).ravel() == 0).all():
            return 0
        return len(self.goals_as_vectors)

class EmbeddingEnvEvaluate(EmbeddingEnv):
    def __init__(self):
        super(EmbeddingEnvEvaluate, self).__init__()


if __name__ == "__main__":
    raise BaseException("Can't run the script \nTry to use Gym to run embedding_world environment")