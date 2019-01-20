import gym
import numpy as np
import embedding_world


if __name__ == '__main__':

    env = gym.make('embedding_world-v0')

    env.__set_sentences__('grab', 'جرب ديه')

    env.__set_paths__(embedding_from_file="embedding_world/envs/world_sample/mini.wiki.multi.2.en.vec",
                        embedding_to_file="embedding_world/envs/world_sample/mini.wiki.multi.2.ar.vec")
    env.reset()

    #env.production_is_on()

    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([3], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([3], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([3], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([0], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([4], dtype='int64'))
    print(state, reward, done, info)
    print("===================================")
    state, reward, done, info = env.step(np.array([1], dtype='int64'))
    print(state, reward, done, info)

    env.close()