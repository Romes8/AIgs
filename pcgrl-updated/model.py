# model.py

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch as th
import torch.nn as nn

# Define a custom feature extractor
class FullyConvExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that applies convolutional layers to the observation.
    Assumes observations are images with shape (height, width, channels).
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # Ensure the observation space has the expected shape
        assert len(observation_space.shape) == 3, "Observation space must have shape (height, width, channels)"
        
        super(FullyConvExtractor, self).__init__(observation_space, features_dim)
        
        # Extract the number of input channels
        n_input_channels = observation_space.shape[2]  # channels last
        
        # Define CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # (H, W) -> (H, W)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (H, W) -> (H, W)
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate the number of features after convolutional layers
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            n_flatten = self.conv(sample_input).shape[1]
        
        # Define the linear layer
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: th.Tensor) -> th.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations (th.Tensor): Batch of observations with shape (batch, height, width, channels)
        
        Returns:
            th.Tensor: Extracted features with shape (batch, features_dim)
        """
        # Permute to (batch, channels, height, width) for PyTorch
        observations = observations.permute(0, 3, 1, 2)
        conv_out = self.conv(observations)
        return self.linear(conv_out)

# Define a custom policy using the extractor
class FullyConvPolicySmallMap(ActorCriticCnnPolicy):
    """
    Custom policy that uses the FullyConvExtractor as its feature extractor.
    Inherits from ActorCriticCnnPolicy provided by Stable-Baselines3.
    """
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicySmallMap, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=FullyConvExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )
