import gym
import gym_sokoban  # Ensure gym-sokoban is installed and imported
import numpy as np
from gym import spaces

# Helper function to clean input actions
def get_action(a):
    return a.item() if hasattr(a, "item") else a

# Function to retrieve the Sokoban environment from wrapped environments
def get_sokoban_env(env):
    if "SokobanEnv" in str(type(env)):
        return env
    elif hasattr(env, 'env'):
        return get_sokoban_env(env.env)
    else:
        return None

class ToImage(gym.Wrapper):
    def __init__(self, env, names, **kwargs):
        sokoban_env = get_sokoban_env(env)
        if sokoban_env is not None and hasattr(sokoban_env, 'adjust_param'):
            sokoban_env.adjust_param(**kwargs)
        super(ToImage, self).__init__(env)

        self.names = names
        self.shape = None
        depth = 0
        max_value = 0

        for n in names:
            if isinstance(self.env.observation_space, spaces.Dict):
                assert n in self.env.observation_space.spaces.keys(), f'The wrapper requires "{n}" in observation_space.'
                new_shape = self.env.observation_space.spaces[n].shape
                high_val = self.env.observation_space.spaces[n].high.max()
            elif isinstance(self.env.observation_space, spaces.Box):
                assert n == 'map', f'The wrapper requires "map" for Box observation_space.'
                new_shape = self.env.observation_space.shape
                high_val = self.env.observation_space.high.max()
            else:
                raise NotImplementedError("Unsupported observation space type")

            if self.shape is None:
                self.shape = new_shape
            if len(new_shape) <= 2:
                depth += 1
            else:
                depth += new_shape[2]
            max_value = max(max_value, high_val)

        self.observation_space = spaces.Box(
            low=0, high=max_value, shape=(self.shape[0], self.shape[1], depth), dtype=np.float32
        )

        # Debugging Statement
        print(f"ToImage Wrapper Initialized: Observation Space Shape set to {self.observation_space.shape}")

    def step(self, action):
        action = get_action(action)
        step_output = self.env.step(action)

        if isinstance(step_output, tuple):
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
            elif len(step_output) == 4:
                obs, reward, done, info = step_output
                terminated, truncated = done, False
            else:
                raise ValueError("Unexpected number of values returned from step(). Expected 4 or 5 elements.")
        else:
            raise ValueError("Environment step did not return a tuple.")

        transformed_obs = self.transform(obs)
        return transformed_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
        else:
            obs = reset_output
            info = {}

        transformed_obs = self.transform(obs)
        return transformed_obs, info

    def transform(self, obs):
        final = None
        for n in self.names:
            if isinstance(self.env.observation_space, spaces.Dict):
                obs_n = obs[n].reshape(self.shape[0], self.shape[1], -1)
            elif isinstance(self.env.observation_space, spaces.Box):
                obs_n = obs.reshape(self.shape[0], self.shape[1], -1)
            else:
                raise NotImplementedError("Unsupported observation space type")

            if final is None:
                final = obs_n
            else:
                final = np.concatenate((final, obs_n), axis=2)

        # Debugging Statement
        print(f"ToImage.transform: Transformed observation shape={final.shape}")
        return final

class OneHotEncoding(gym.Wrapper):
    def __init__(self, env, name, **kwargs):
        sokoban_env = get_sokoban_env(env)
        if sokoban_env is not None and hasattr(sokoban_env, 'adjust_param'):
            sokoban_env.adjust_param(**kwargs)
        super(OneHotEncoding, self).__init__(env)

        self.name = name

        if isinstance(env.observation_space, spaces.Dict):
            obs_space = env.observation_space.spaces[self.name]
        else:
            obs_space = env.observation_space

        if isinstance(obs_space, spaces.Discrete):
            self.dim = obs_space.n
            new_shape = (self.dim,)
        elif isinstance(obs_space, spaces.Box):
            self.dim = int(obs_space.high.max()) + 1
            new_shape = (*obs_space.shape, self.dim)
        else:
            raise NotImplementedError("OneHotEncoding wrapper only supports Discrete and Box spaces.")

        new_space = spaces.Box(
            low=0, high=1, shape=new_shape, dtype=np.float32
        )

        if isinstance(env.observation_space, spaces.Dict):
            self.observation_space.spaces[self.name] = new_space
        else:
            self.observation_space = new_space

        # Debugging Statement
        print(f"OneHotEncoding Wrapper Initialized: New observation space for '{self.name}' with shape {new_shape}")

    def step(self, action):
        step_output = self.env.step(action)

        if isinstance(step_output, tuple):
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
            elif len(step_output) == 4:
                obs, reward, done, info = step_output
                terminated, truncated = done, False
            else:
                raise ValueError("Unexpected number of values returned from step(). Expected 4 or 5 elements.")
        else:
            raise ValueError("Environment step did not return a tuple.")

        obs = self.transform(obs)
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
        else:
            obs = reset_output
            info = {}

        obs = self.transform(obs)
        return obs, info

    def transform(self, obs):
        if isinstance(self.env.observation_space, spaces.Dict):
            old = obs[self.name]
            obs_space = self.env.observation_space.spaces[self.name]
        else:
            old = obs
            obs_space = self.env.observation_space

        if isinstance(obs_space, spaces.Discrete):
            one_hot = np.eye(self.dim, dtype=np.float32)[old]
        elif isinstance(obs_space, spaces.Box):
            one_hot = np.eye(self.dim, dtype=np.float32)[old.astype(int)]
        else:
            raise NotImplementedError("OneHotEncoding wrapper only supports Discrete and Box spaces.")

        if isinstance(obs, dict):
            obs[self.name] = one_hot
        else:
            obs = one_hot

        # Debugging Statement
        print(f"OneHotEncoding.transform: Transformed observation shape={obs.shape}")
        return obs

class ActionMap(gym.Wrapper):
    def __init__(self, env, **kwargs):
        sokoban_env = get_sokoban_env(env)
        if sokoban_env is not None and hasattr(sokoban_env, 'adjust_param'):
            sokoban_env.adjust_param(**kwargs)
        super(ActionMap, self).__init__(env)

        # Retrieve ACTION_LOOKUP from the Sokoban environment
        sokoban_env = get_sokoban_env(env)
        if sokoban_env is None:
            raise ValueError("Cannot find SokobanEnv")

        if not hasattr(sokoban_env, 'ACTION_LOOKUP'):
            raise AttributeError("SokobanEnv does not have ACTION_LOOKUP attribute")

        self.valid_actions = list(sokoban_env.ACTION_LOOKUP)  # Convert to list
        self.action_mapping = {i: action for i, action in enumerate(self.valid_actions)}
        self.reverse_action_mapping = {action: i for i, action in enumerate(self.valid_actions)}

        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Initialize old_obs
        self.old_obs = None

        # Debugging Statement
        print(f"ActionMap Wrapper Initialized: Action space set to {self.action_space}, Number of valid actions: {len(self.valid_actions)}")

    def step(self, action):
        # Map integer action to (x, y, v)
        if action not in self.action_mapping:
            raise ValueError(f"Invalid action index: {action}. It should be within [0, {len(self.valid_actions)-1}]")
        mapped_action = self.action_mapping[action]

        step_output = self.env.step(mapped_action)

        if isinstance(step_output, tuple):
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
            elif len(step_output) == 4:
                obs, reward, done, info = step_output
                terminated, truncated = done, False
            else:
                raise ValueError("Unexpected number of values returned from step(). Expected 4 or 5 elements.")
        else:
            raise ValueError("Environment step did not return a tuple.")

        self.old_obs = obs
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
        else:
            obs = reset_output
            info = {}

        self.old_obs = obs
        return obs, info

class FlattenedActionMap(gym.Wrapper):
    def __init__(self, env):
        super(FlattenedActionMap, self).__init__(env)

        if isinstance(env.observation_space, spaces.Dict):
            h, w = env.observation_space.spaces['map'].shape[:2]
        elif isinstance(env.observation_space, spaces.Box):
            h, w = env.observation_space.shape[:2]
        else:
            raise NotImplementedError("Unsupported observation space type")

        self.h, self.w, self.num_tiles = h, w, getattr(env, 'get_num_tiles', lambda: 4)()

        # Set a new, flattened action space for compatibility with PPO
        self.action_space = spaces.Discrete(self.h * self.w * self.num_tiles)

        # Debugging Statement
        print(f"FlattenedActionMap Wrapper Initialized: Action space set to {self.action_space}")

    def step(self, action):
        # Unflatten action to get (x, y, tile_type)
        y, x, tile_type = np.unravel_index(action, (self.h, self.w, self.num_tiles))
        # Pass the unflattened action to the environment's step as a tuple
        return self.env.step((x, y, tile_type))

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
        else:
            obs = reset_output
            info = {}

        return obs, info

class Cropped(gym.Wrapper):
    def __init__(self, env, crop_size, pad_value, name, **kwargs):
        sokoban_env = get_sokoban_env(env)
        if sokoban_env is not None and hasattr(sokoban_env, 'adjust_param'):
            sokoban_env.adjust_param(**kwargs)
        super(Cropped, self).__init__(env)

        self.name, self.size = name, crop_size
        self.pad = crop_size // 2
        self.pad_value = pad_value

        if isinstance(env.observation_space, spaces.Dict):
            assert 'pos' in env.observation_space.spaces.keys(), 'Pos key required'
            map_space = env.observation_space.spaces[self.name]
            self.h, self.w = map_space.shape[:2]
        elif isinstance(env.observation_space, spaces.Box):
            self.h, self.w = env.observation_space.shape[:2]
        else:
            raise NotImplementedError("Unsupported observation space type")

        self.observation_space = spaces.Box(
            low=0, high=self.pad_value, shape=(crop_size, crop_size), dtype=np.uint8
        )

        # Debugging Statement
        print(f"Cropped Wrapper Initialized: Crop size {crop_size}, Pad value {pad_value}")

    def step(self, action):
        action = get_action(action)
        step_output = self.env.step(action)

        if isinstance(step_output, tuple):
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
            elif len(step_output) == 4:
                obs, reward, done, info = step_output
                terminated, truncated = done, False
            else:
                raise ValueError("Unexpected number of values returned from step(). Expected 4 or 5 elements.")
        else:
            raise ValueError("Environment step did not return a tuple.")

        transformed_obs = self.transform(obs)
        return transformed_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        reset_output = self.env.reset(**kwargs)

        if isinstance(reset_output, tuple):
            if len(reset_output) == 2:
                obs, info = reset_output
            else:
                obs = reset_output
                info = {}
        else:
            obs = reset_output
            info = {}

        transformed_obs = self.transform(obs)
        return transformed_obs, info

    def transform(self, obs):
        if isinstance(obs, dict):
            map_ = obs[self.name]
            pos = obs['pos']
        elif isinstance(obs, np.ndarray):
            map_ = obs
            # Placeholder for position; adjust as needed
            pos = (self.w // 2, self.h // 2)
        else:
            raise NotImplementedError("Unsupported observation type")

        x, y = pos
        padded = np.pad(map_, self.pad, constant_values=self.pad_value)
        x += self.pad
        y += self.pad
        cropped_map = padded[y - self.pad:y + self.pad + 1, x - self.pad:x + self.pad + 1]

        if isinstance(obs, dict):
            obs[self.name] = cropped_map
            return obs, {}
        elif isinstance(obs, np.ndarray):
            return cropped_map, {}
        else:
            raise NotImplementedError("Unsupported observation type")

class CroppedImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, env, crop_size=5, **kwargs):
        border_tile = getattr(env, 'get_border_tile', lambda: 0)()
        env = Cropped(env, crop_size, border_tile, 'map', **kwargs)
        if 'binary' not in kwargs:
            env = OneHotEncoding(env, 'map')
        env = ToImage(env, ['map'])
        super(CroppedImagePCGRLWrapper, self).__init__(env)

        # Debugging Statement
        print("CroppedImagePCGRLWrapper Initialized")

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ActionMapImagePCGRLWrapper(gym.Wrapper):
    def __init__(self, env, **kwargs):
        env = ActionMap(env, **kwargs)
        if 'binary' not in kwargs:
            env = OneHotEncoding(env, 'map')
        env = ToImage(env, ['map'])
        super(ActionMapImagePCGRLWrapper, self).__init__(env)

        # Debugging Statement
        print("ActionMapImagePCGRLWrapper Initialized")

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
