from gym.envs.registration import register

register(
    id='embedding_world-v0',
    entry_point='embedding_world.envs:EmbeddingEnvExample',
)

register(
    id='embedding_world-NoFrameskip-v0',
    entry_point='embedding_world.envs:EmbeddingEnvExample',
)
