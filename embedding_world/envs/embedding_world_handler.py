import numpy as np
from gensim.models import KeyedVectors


def load_space(space_file_path):
    # import os for this scope only
    import os
    # handle all case to load the corpse
    if not space_file_path is None:
        return KeyedVectors.load_word2vec_format(str(space_file_path), binary=False)
    else:
        if not os.path.exists(space_file_path):
            dir_path = os.path.dirname(os.path.abspath(__file__))
            rel_path = os.path.join("%sworld_samples%s" % (dir_path, space_file_path))
            if os.path.exists(rel_path):
                space_file_path_to = rel_path
            else:
                raise FileExistsError("Cannot find %s." % space_file_path)
        return KeyedVectors.load_word2vec_format(str(space_file_path), binary=False)


class SpaceHandler:
    def __init__(self, space_file_path_from=None, space_file_path_to=None):

        # initialize goal and task finish or not
        self.initial_matrix = []
        self.__goal = 0
        self.__task_is_over = False
        self.COMPASS = {}
        self.is_robot_in_region = False

        # print message to make awareness that it might take some time
        print('Start loading the space from \'%s\', it might take several time.' %
              '/'.join(space_file_path_from.split('/')[0:-1]))

        # Load space to map from it
        self.__space_initial = load_space(space_file_path_from)

        # Load space to map to it
        self.__space_target = load_space(space_file_path_to)

        self.epsilon = self.__compute_epsilon__()

        # define the embedding dimension
        self.emb_dim = self.__space_initial.wv.vector_size

        # define the name for the space
        space_name = 'Space-%iD' % self.emb_dim

        # print message to make awareness that loading word2vec model finished
        print('Finish loading %s with epsilon equals %f' % (space_name, self.epsilon))

    def handle_initial_and_goal(self, initial_words_list, goal_words_list):
        max_length = max(len(initial_words_list), len(goal_words_list))
        self.set_initial(initial_words_list, max_length)
        self.set_goals(goal_words_list, max_length)

        # set the starting point
        self.__entrance = self.reset_robot()

        # create the moving robot
        self.__robot = self.entrance

        # define all available movement for the space
        self.__configuration__()

    def stop(self):
        try:
            self.__task_is_over = True
        except Exception:
            pass

    def move_robot(self, dir):
        if dir not in self.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s." % (str(dir), str(self.COMPASS.keys())))
        if self.__is_in_region__(dir, self.__robot[0]):
            # move the robot
            self.__robot[0] = self.__robot[0] + np.array(self.COMPASS[dir], dtype='float64')
        return self.__robot[0]

    def set_goals(self, goal_words_list, desired_length):
        self.goal_matrix = []
        for i in range(desired_length):
            if i < len(goal_words_list):
                self.goal_matrix.append(self.__space_target.wv[goal_words_list[i]])
            else:
                self.goal_matrix.append(np.zeros(self.emb_dim, dtype='float64'))

    def set_initial(self, initial_words_list, desired_length):
        for i in range(desired_length):
            if i < len(initial_words_list):
                self.initial_matrix.append(self.__space_initial.wv[initial_words_list[i]])
            else:
                self.initial_matrix.append(np.zeros(self.emb_dim, dtype='float64'))

    def __is_in_region__(self, dir, pos):
        # define the future step that the robot wants to move to it
        future_step = pos + self.COMPASS[dir]
        # check that future step in between 1 and -1
        if (1 < future_step).any() or (future_step < -1).any():
            self.is_robot_in_region = False
            return False
        self.is_robot_in_region = True
        return True

    def __residual_vectors__(self):
        # *** IMPORTANCE STEP
        # Add the matched word and the location of source second word to be dependant on each-other
        self.__robot[1] = self.__robot[1] + self.__robot[0]
        # Remove the matched word
        self.__robot.pop(0)

    def __compute_epsilon__(self):
        return round(1 / max(len(self.__space_initial.wv.vocab), len(self.__space_target.wv.vocab)), 5)

    def __configuration__(self):
        # initialize temp. with zeros (no movement)
        temp_tuple = [0] * self.emb_dim
        self.COMPASS['pick-up'] = tuple(temp_tuple)
        # make move two move for every dimension
        for i in range(self.emb_dim):
            # define increasing motion in dimension(i)
            up, up[i] = temp_tuple, self.epsilon
            self.COMPASS["dim(%s)+1" % i] = tuple(up)
            # define decreasing motion in dimension(i)
            down, down[i] = temp_tuple, -1 * self.epsilon
            self.COMPASS["dim(%s)-1" % i] = tuple(down)
            # reinitialize temp. with zeros
            temp_tuple = [0] * self.emb_dim

    def get_goals(self):
        return self.goal_matrix

    def get_initial(self):
        return self.initial_matrix

    def reset_robot(self):
        return self.get_initial()

    def set_robot(self, pos):
        self.__robot = np.array([pos])

    def get_epsilon(self):
        return self.epsilon

    def get_word_from_vec(self, vec):
        return self.__space_target.wv.most_similar(positive=[vec], topn=1)[0][0]

    @property
    def space(self):
        return self.__space_initial

    @property
    def robot(self):
        return self.__robot

    @property
    def current_pos(self):
        return self.__robot[0]

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal


if __name__ == "__main__":
    raise BaseException("Can't run the script Try to use Gym to run embedding_world environment")
