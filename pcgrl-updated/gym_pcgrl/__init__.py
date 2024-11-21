from gymnasium.envs.registration import register

register(
    id='sokoban-narrow-v0',
    entry_point='gym_pcgrl.envs.pcgrl_env:PcgrlEnv',
    max_episode_steps=1000,
    reward_threshold=100.0,
    # Removed 'render_modes' keyword
)
register(
    id='sokoban-wide-v0',
    entry_point='gym_pcgrl.envs.pcgrl_env:PcgrlEnv',
    kwargs={'prob': 'sokoban', 'rep': 'wide'},
    max_episode_steps=1000,
    reward_threshold=100.0,
)
register(
    id='sokoban-turtle-v0',
    entry_point='gym_pcgrl.envs.pcgrl_env:PcgrlEnv',
    kwargs={'prob': 'sokoban', 'rep': 'wide'},
    max_episode_steps=1000,
    reward_threshold=100.0,
)