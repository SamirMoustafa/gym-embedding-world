import os
import numpy as np
from gensim.models import KeyedVectors


class SpaceHandler:

    COMPASS = {}

    def __init__(self, space_file_path_from=None, space_file_path_to=None,
                 initial_words_list=None, goal_words_list=None,
                 epslion=None):

        # print message to make awareness that it might take some time
        print('Start loading the space from \'%s\', it might take several time.' %
              '/'.join(space_file_path_from.split('/')[0:-1]))

        # initialize goal and task finish or not
        self.__goal = 0
        self.__task_is_over = False

        # Load a space
        if not space_file_path_from is None:
            self.__space_initial = KeyedVectors.load_word2vec_format(space_file_path_from, binary=False)
        else:
            if not os.path.exists(space_file_path_from):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "world_samples", space_file_path_from)
                if os.path.exists(rel_path):
                    space_file_path_from = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % space_file_path_from)
            self.__space_initial = KeyedVectors.load_word2vec_format(space_file_path_from, binary=False)


        if not space_file_path_to is None:
            self.__space_target = KeyedVectors.load_word2vec_format(space_file_path_to, binary=False)
        else:
            if not os.path.exists(space_file_path_to):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "world_samples", space_file_path_to)
                if os.path.exists(rel_path):
                    space_file_path_to = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % space_file_path_to)
            self.__space_target = KeyedVectors.load_word2vec_format(space_file_path_to, binary=False)


        # set the initial and goal
        self.set_initial(initial_words_list)
        self.set_goals(goal_words_list)


        # Load epsilon value
        if epslion is None or epslion >= 1:
            raise ValueError("Epsilon can't be %s" % epslion)
        else:
            self.space_size = int(epslion ** (-1))

        # define the embedding dimension
        self.emb_dim = self.__space_initial.wv.vector_size

        # define the name for the space
        space_name = 'Space-%iD' % self.emb_dim

        # set the starting point
        self.__entrance = self.reset_robot()

        # define all available movement for the space
        # initialize temp. with zeros (no movement)
        temp_tuple = [0] * self.emb_dim
        for i in range(self.emb_dim):
            # define increasing motion in dimension(i)
            up, up[i] = temp_tuple, epslion
            self.COMPASS["dim(%s)+1" % i] = tuple(up)
            # define decreasing motion in dimension(i)
            down, down[i] = temp_tuple, -1 * epslion
            self.COMPASS["dim(%s)-1" % i] = tuple(down)
            # reinitialize temp. with zeros
            temp_tuple = [0] * self.emb_dim

        # create the moving robot
        self.__robot = self.entrance

        # print message to make awareness that loading word2vec model finished
        print('Finish loading %s with epsilon equals %f' % (space_name, epslion))

    def stop(self):
        try:
            self.__task_is_over = True
        except Exception:
            pass

    def move_robot(self, dir):
        if dir not in self.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s." % (str(dir), str(self.COMPASS.keys())))

        if self.in_region(dir):
            # move the robot
            self.__robot += np.array(self.COMPASS[dir], dtype='float64')

    def in_region(self, dir):
        # define the future step that the robot wants to move to it
        future_step = self.__robot + np.array(self.COMPASS[dir])
        # check that future step in between 1 and -1
        if (1 < future_step).all() or (future_step < -1).all(): return False
        return True

    def set_goals(self, goal_words_list):
        self.goal_matrix = []
        for word in goal_words_list:
            self.goal_matrix.append(self.__space_target.wv[word])

    def set_initial(self, initial_words_list):
        self.initial_matrix = []
        for word in initial_words_list:
            self.initial_matrix.append(self.__space_initial.wv[word])
        print(self.initial_matrix)

    def get_goals(self):
        return self.goal_matrix

    def get_initial(self):
        return self.initial_matrix

    def reset_robot(self):
        self.__robot = np.array(self.initial_matrix, dtype='float64')

    @property
    def space(self):
        return self.__space_initial

    @property
    def robot(self):
        return self.__robot

    @property
    def entrance(self):
        return self.__entrance

    @property
    def goal(self):
        return self.__goal

    @property
    def stop(self):
        return self.__task_is_over
