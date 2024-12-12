import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Helper functions
get_action = lambda a: a.item() if hasattr(a, "item") else a
get_pcgrl_env = lambda env: env if "PcgrlEnv" in str(type(env)) else get_pcgrl_env(env.env)

class ToImage(gym.Wrapper):
    def __init__(self, env, names, **kwargs):
        env = gym.make(env) if isinstance(env, str) else env
        super().__init__(env)
        self.names = names
        depth = 0
        max_value = 0

        # Validate and calculate observation space dimensions
        for name in names:
            assert name in env.observation_space.spaces, f"Key '{name}' missing in observation space."
            shape = env.observation_space[name].shape
            depth += 1 if len(shape) == 2 else shape[-1]
            max_value = max(max_value, env.observation_space[name].high.max())

        # Define the transformed observation space
        height, width = env.observation_space[names[0]].shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=max_value, shape=(height, width, depth), dtype=np.uint8
        )

    def step(self, action):
        action = action.item() if hasattr(action, "item") else action
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.transform(obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.transform(obs), info

    def transform(self, obs):
        images = [obs[name].reshape(obs[name].shape[:2] + (-1,)) for name in self.names]
        return np.concatenate(images, axis=-1)


class OneHotEncoding(gym.Wrapper):
    # Converts a specified observation into a one-hot encoded representation.
    def __init__(self, env, name, **kwargs):
        super().__init__(env)
        self.name = name
        shape = env.observation_space[name].shape
        self.num_classes = env.observation_space[name].high.max() + 1

        self.observation_space = gym.spaces.Dict(env.observation_space.spaces)
        new_shape = shape + (self.num_classes,)
        self.observation_space.spaces[name] = gym.spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.uint8
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(get_action(action))
        obs[self.name] = self.to_one_hot(obs[self.name])
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs[self.name] = self.to_one_hot(obs[self.name])
        return obs, info

    def to_one_hot(self, array):
        one_hot = np.eye(self.num_classes)[array]
        return one_hot


class ActionMap(gym.Wrapper):
    # Represents actions as indices in a 2D map (height, width).
    def __init__(self, env, **kwargs):
        super().__init__(env)
        map_shape = env.observation_space["map"].shape
        self.height, self.width = map_shape
        self.num_tiles = env.observation_space["map"].high.max() + 1
        self.action_space = gym.spaces.Discrete(self.height * self.width * self.num_tiles)

    def step(self, action):
        y, x, v = np.unravel_index(action, (self.height, self.width, self.num_tiles))
        obs, reward, done, truncated, info = self.env.step((x, y, v))

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class Cropped(gym.Wrapper):
    # Crops and centers the observation around the agent's position.
    def __init__(self, env, crop_size, pad_value, name, **kwargs):
        super().__init__(env)
        self.name = name
        self.crop_size = crop_size
        self.pad_value = pad_value
        self.pad = crop_size // 2

        high = env.observation_space[name].high.max()
        self.observation_space = gym.spaces.Dict(env.observation_space.spaces)
        self.observation_space.spaces[name] = gym.spaces.Box(
            low=0, high=high, shape=(crop_size, crop_size), dtype=np.uint8
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(get_action(action))
        obs[self.name] = self.crop(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs[self.name] = self.crop(obs)
        return obs, info

    def crop(self, obs):
        map_ = obs[self.name]
        x, y = obs["pos"]
        padded = np.pad(map_, self.pad, constant_values=self.pad_value)
        return padded[y:y + self.crop_size, x:x + self.crop_size]


class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, env_name, crop_size, **kwargs):
        env = gym.make(env_name, **kwargs)

        base_env = env.unwrapped

        env = Cropped(env, crop_size, base_env.get_border_tile(), "map")
        env = OneHotEncoding(env, "map")

        super().__init__(ToImage(env, ["map"]))

        self.episode_rewards = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_rewards.append(0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_rewards[-1] += reward

        return obs, reward, terminated, truncated, info

    def get_attr(self, name, indices=None):
        if name == 'episode_rewards':
            return self.episode_rewards
        return super().get_attr(name, indices)
     
    def render(self, mode='human'):
        return self.env.render()


class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, env_name, **kwargs):
        env = gym.make(env_name, **kwargs)

        base_env = env.unwrapped

        env = ActionMap(env)
        env = OneHotEncoding(env, "map")

        super().__init__(ToImage(env, ["map"]))

        self.episode_rewards = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_rewards.append(0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.episode_rewards[-1] += reward

        return obs, reward, terminated, truncated, info

    def get_attr(self, name, indices=None):
        if name == 'episode_rewards':
            return self.episode_rewards
        return super().get_attr(name, indices)
    
    def render(self, mode='human'):
        return self.env.render()
        