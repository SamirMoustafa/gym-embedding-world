import gym
import numpy as np
import embedding_world


if __name__ == '__main__':

    env = gym.make('embedding_worldNoFrameskip-v0')

    env.__set_sentences__('grab', 'جرب ديه')

    env.__set_paths__(embedding_from_file="embedding_world/envs/world_sample/mini.wiki.multi.2.en.vec",
                        embedding_to_file="embedding_world/envs/world_sample/mini.wiki.multi.2.ar.vec")

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
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([2], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([2], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([2], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done)

    env.close()