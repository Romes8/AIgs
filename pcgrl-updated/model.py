import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy


def init_weights(module, init_scale=np.sqrt(2)):
    """
    Initialize weights using Kaiming Normal initialization for Conv2d and Linear layers.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FullyConv1Extractor(BaseFeaturesExtractor):
    """
    Feature extractor for narrow representation.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(FullyConv1Extractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]  # Assuming channels-last
        h, w = observation_space.shape[:2]

        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # c1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),                # c2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * h * w, features_dim),
            nn.ReLU()
        )
        self.conv.apply(init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Permute to (batch, channels, height, width) for PyTorch
        observations = observations.permute(0, 3, 1, 2)
        return self.conv(observations)


class FullyConv2Extractor(BaseFeaturesExtractor):
    """
    Feature extractor for wide or turtle representation.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(FullyConv2Extractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[2]
        h, w = observation_space.shape[:2]

        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # c1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),                # c2
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (h // 2) * (w // 2), features_dim),
            nn.ReLU()
        )
        self.conv.apply(init_weights)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Permute to (batch, channels, height, width)
        observations = observations.permute(0, 3, 1, 2)
        return self.conv(observations)


class FullyConvPolicy(ActorCriticCnnPolicy):
    """
    Custom policy using the custom feature extractors.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        if observation_space.shape[0] < 10:
            features_extractor_class = FullyConv1Extractor
            print(f"Using FullyConv1Extractor for observation space shape: {observation_space.shape}")
        else:
            features_extractor_class = FullyConv2Extractor
            print(f"Using FullyConv2Extractor for observation space shape: {observation_space.shape}")

        super(FullyConvPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=features_extractor_class,
            *args,
            **kwargs
        )

        print(f"Policy action space: {self.action_space}")
