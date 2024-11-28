# model.py

import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np  # Ensure numpy is imported
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy

def init_weights(module, init_scale=np.sqrt(2)):
    """
    Initialize weights using Kaiming Normal initialization for Conv2d and Linear layers.
    """
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class FullyConv1Extractor(BaseFeaturesExtractor):
    """
    Equivalent to the old FullyConv1 in TensorFlow.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_tools: int = 1):
        print(f"Initializing FullyConv1Extractor with n_tools={n_tools}, features_dim={features_dim}")
        # Ensure the observation space has the expected shape
        assert len(observation_space.shape) == 3, "Observation space must have shape (H, W, C)"
        super(FullyConv1Extractor, self).__init__(observation_space, features_dim)

        # Extract the number of input channels
        n_input_channels = observation_space.shape[2]  # Assuming channels-last

        # Define the convolutional layers as per FullyConv1
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # c1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),               # c2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c6
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c7
            nn.ReLU(),
            nn.Conv2d(64, n_tools, kernel_size=3, stride=1, padding=1),          # c8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_tools * observation_space.shape[0] * observation_space.shape[1], 512),  # fc1
            nn.ReLU()
        )
        self.conv.apply(init_weights)

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
        return self.conv(observations)

class FullyConv2Extractor(BaseFeaturesExtractor):
    """
    Equivalent to the old FullyConv2 in TensorFlow.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_tools: int = 1):
        print(f"Initializing FullyConv2Extractor with n_tools={n_tools}, features_dim={features_dim}")
        # Ensure the observation space has the expected shape
        assert len(observation_space.shape) == 3, "Observation space must have shape (H, W, C)"
        super(FullyConv2Extractor, self).__init__(observation_space, features_dim)

        # Extract the number of input channels
        n_input_channels = observation_space.shape[2]  # Assuming channels-last

        # Define the convolutional layers as per FullyConv2
        self.conv_shared = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # c1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),               # c2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),               # c3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # c6
            nn.ReLU(),
            nn.Conv2d(64, n_tools, kernel_size=3, stride=1, padding=1),          # c8
            nn.ReLU()
        )
        self.conv_shared.apply(init_weights)

        # Dynamically compute the size of the flattened output
        with th.no_grad():
            # Sample a dummy input to compute the output size
            sample_input = th.zeros((1, n_input_channels, *observation_space.shape[:2]))
            conv_output = self.conv_shared(sample_input)
            flat_size = conv_output.numel()  # Total elements in the output tensor

        # Define the final fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, features_dim),  # Use dynamically calculated size
            nn.ReLU()
        )
        self.fc.apply(init_weights)

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
        conv_out = self.conv_shared(observations)
        return self.fc(conv_out)


class FullyConvPolicyBigMap(ActorCriticCnnPolicy):
    """
    Custom policy for BigMap using FullyConv2Extractor.
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr_schedule, **kwargs):
        super(FullyConvPolicyBigMap, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FullyConv2Extractor,
            **kwargs  # features_extractor_kwargs passed via policy_kwargs
        )

class FullyConvPolicySmallMap(ActorCriticCnnPolicy):
    """
    Custom policy for SmallMap using FullyConv1Extractor.
    """
    def __init__(self, observation_space: gym.Space, action_space: gym.Space, lr_schedule, **kwargs):
        super(FullyConvPolicySmallMap, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=FullyConv1Extractor,
            **kwargs  # features_extractor_kwargs passed via policy_kwargs
        )
