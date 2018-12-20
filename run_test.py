import gym
import numpy as np
import embedding_world

if __name__ == '__main__':

    env = gym.make('embedding_worldNoFrameskip-v0')

    env.reset()

    state, reward, done, info = env.step(np.array([20],dtype='int64'))

    print(state, reward, done, info)

    env.close()