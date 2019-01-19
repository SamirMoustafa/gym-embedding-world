import gym
import numpy as np
import embedding_world


if __name__ == '__main__':

    env = gym.make('embedding_worldNoFrameskip-v0')

    env.reset()

    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)

    env.close()