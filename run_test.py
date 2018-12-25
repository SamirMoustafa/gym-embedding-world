import gym
import os
import numpy as np
import pandas as pd
import embedding_world
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt

if __name__ == '__main__':

    env = gym.make('embedding_worldNoFrameskip-v0')

    env.reset()

    state, reward, done, info = env.step(np.array([1],dtype='int64'))

    print(state, reward, done, info)

    env.close()
