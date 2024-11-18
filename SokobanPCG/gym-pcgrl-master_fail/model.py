import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from gym import spaces
from stable_baselines3.common.policies import ActorCriticCnnPolicy

class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor compatible with Stable Baselines3 and PyTorch.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # Adjusted CNN structure with padding=1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # Output size: same as input
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),                # Output size: same as input
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),                # Output size: same as input
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * observation_space.shape[1] * observation_space.shape[2], features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.cnn(observations)


class FullyConvPolicySmallMap(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicySmallMap, self).__init__(
            *args, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512), **kwargs
        )


class FullyConvPolicyBigMap(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicyBigMap, self).__init__(
            *args, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512), **kwargs
        )


class CustomPolicyBigMap(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicyBigMap, self).__init__(
            *args, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512), **kwargs
        )


class CustomPolicySmallMap(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicySmallMap, self).__init__(
            *args, features_extractor_class=CustomCNN, features_extractor_kwargs=dict(features_dim=512), **kwargs
        )
