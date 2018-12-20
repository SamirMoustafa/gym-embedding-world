from gym.envs.registration import register

register(
    id='embedding_world-v0',
    entry_point='embedding_world.envs:EmbeddingEnvExample',
)

register(
    id='embedding_worldNoFrameskip-v0',
    entry_point='embedding_world.envs:EmbeddingEnvExample',
    kwargs={'game': 'embedding_world', 'obs_type': 'ram', 'frameskip': 1}, # A frameskip of 1 means we get every frame
    max_episode_steps=1 * 100000,
    nondeterministic=False,
)
