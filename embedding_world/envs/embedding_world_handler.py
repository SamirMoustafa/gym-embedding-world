import os

import numpy as np
from gensim.models import KeyedVectors


class SpaceHandler:

    COMPASS = {}

    def __init__(self, space_name="SpaceND", space_file_path=None, epslion=None, goal=None):

        self.__goal = 0

        self.__task_is_over = False
        #asdasdasd
        # Load a maze
        if space_file_path is None:
            self.__space = KeyedVectors.load_word2vec_format(space_file_path, binary=False)
        else:
            if not os.path.exists(space_file_path):
                dir_path = os.path.dirname(os.path.abspath(__file__))
                rel_path = os.path.join(dir_path, "world_samples", space_file_path)
                if os.path.exists(rel_path):
                    space_file_path = rel_path
                else:
                    raise FileExistsError("Cannot find %s." % space_file_path)
            self.__space = KeyedVectors.load_word2vec_format(space_file_path, binary=False)

        if epslion:
            self.space_size = int(epslion ** (-1))
        else:
            raise ValueError("Epsilon can't be %s" % epslion)

        self.emb_dim = self.__space.wv.vector_size

        space_name = 'Space%iD' % self.emb_dim

        # Set the starting point
        self.__entrance = np.zeros(self.emb_dim, dtype='float64')

        if goal:
            if not (len(goal) == self.emb_dim):
                raise ValueError("Goal dimension must match with the space")
            else:
                # Set the Goal
                self.__goal = np.array(goal)
                # else:
                # raise ValueError("Goal can't be %s"%goal)

        temp_tuple = [0] * self.emb_dim
        for i in range(self.emb_dim):
            up, up[i] = temp_tuple, epslion
            self.COMPASS["dim(%s)+1" % i] = tuple(up)

            down, down[i] = temp_tuple, -1 * epslion
            self.COMPASS["dim(%s)-1" % i] = tuple(down)

            temp_tuple = [0] * self.emb_dim

        # for i, val in enumerate(self.COMPASS):
        #    print(list(self.COMPASS.keys())[i], self.COMPASS.get(val))

        # Create the Robot
        self.__robot = self.entrance

    def update(self, mode="human"):
        try:
            self.__controller_update()
        except Exception as e:
            self.__task_is_over = True
            self.stop()
            raise e
        else:
            return

    def stop(self):
        try:
            self.__task_is_over = True
        except Exception:
            pass

    def move_robot(self, dir):
        if dir not in self.COMPASS.keys():
            raise ValueError("dir cannot be %s. The only valid dirs are %s." % (str(dir), str(self.COMPASS.keys())))

        if self.in_region(self.__robot, dir):
            # move the robot
            self.__robot += np.array(self.COMPASS[dir],dtype='float64')


    def in_region(self, robot, dir):
        for i in self.__robot + np.array(self.COMPASS[dir]):
            if (1 < i or i < -1): return False
        return True

    def set_goals(self,words_list):
        self.phrase_matrix = []
        for word in words_list:
            self.phrase_matrix.append(self.__space.wv[word])

    def get_goals(self):
        return self.phrase_matrix

    def reset_robot(self):
        self.__robot = np.zeros(self.emb_dim, dtype=int)

    def __controller_update(self):
        pass

    @property
    def space(self):
        return self.__space

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
    def game_over(self):
        return self.__task_is_over


class Portal:
    def __init__(self, *locations):

        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError("location must be a list or a tuple.")

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations