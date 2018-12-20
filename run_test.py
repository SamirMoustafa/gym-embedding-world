import gym
import numpy as np
import embedding_world

if __name__ == '__main__':

    env = gym.make('embedding_worldNoFrameskip-v0')

    env.reset()

    state, reward, done, info = env.step('528')

    print(state, reward, done, info)

    env.close()