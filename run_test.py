import re
import gym
import embedding_world
import numpy as np

normalize_arabic = lambda text: re.sub("[إأٱآا]", "ا", text)

normalize_english = lambda text: text.lower()

if __name__ == '__main__':
    env = gym.make('embedding_world-v0')

    env.set_paths(embedding_from_file="embedding_world/envs/world_sample/mini.wiki.multi.2.en.vec",
                  embedding_to_file="embedding_world/envs/world_sample/mini.wiki.multi.2.ar.vec")

    env.set_normalization(normalize_arabic, normalize_english)

    env.set_sentences('test1 test2', 'تجربه1 تجربه2')

    env.reset()

    # env.production_is_on()

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

    env.reset()

    env.close()
