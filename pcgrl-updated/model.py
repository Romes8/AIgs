# model.py

import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch as th
import torch.nn as nn

class FullyConvExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor with a shared convolutional base for policy and value functions.
    Matches architecture in PCGRL paper with 6 convolutional layers.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        # Ensure the observation space has the expected shape
        assert len(observation_space.shape) == 3, "Observation space must have shape (H, W, C)"
        super(FullyConvExtractor, self).__init__(observation_space, features_dim)

        # Extract the number of input channels
        n_input_channels = observation_space.shape[2]  # Assuming channels-last

        # Shared convolutional base with 6 layers
        self.conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),  # Layer 1
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),               # Layer 2
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # Layer 3
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # Layer 4
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # Layer 5
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),               # Layer 6
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate the number of features after the convolutional layers
        with th.no_grad():
            sample_input = th.as_tensor(observation_space.sample()[None]).permute(0, 3, 1, 2).float()
            conv_out_dim = self.conv(sample_input).shape[1]

        # Linear layer to match the desired features_dim
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim, features_dim),
            nn.ReLU()
        )

        # Initialize weights similar to the original TensorFlow initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

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
        features = self.fc(conv_out)
        return features

class FullyConvPolicy(ActorCriticCnnPolicy):
    """
    Custom policy using FullyConvExtractor for PCGRL environments.
    """
    def __init__(self, *args, **kwargs):
        super(FullyConvPolicy, self).__init__(
            *args,
            **kwargs,
            features_extractor_class=FullyConvExtractor
        )
