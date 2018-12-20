import gym
import numpy as np
import embedding_world

if __name__ == '__main__':

    env = gym.make('embedding_world-v0')

    env.reset()

    env.handle_goals("a")

    state, reward, done, info = env.step('dim(0)+1')

    print(state, reward, done, info)

    env.close()