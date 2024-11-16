import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class SokobanNCA(nn.Module):
    def __init__(self, hidden_channels=16):
        super().__init__()
        
        # Input channels: 5 (one-hot encoding for each type: empty, wall, player, box, goal)
        # Output channels: 5 (same as input)
        self.perception = nn.Conv2d(5, hidden_channels, 3, padding=1)
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 5, 1)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, steps=10):
        # Run the NCA for multiple steps
        for _ in range(steps):
            dx = self.perception(x)
            dx = self.update(dx)
            x = self.softmax(x + dx)
        return x

def load_levels(directory):
    """Load and convert Sokoban levels to tensor format"""
    levels = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                level = f.read().splitlines()
                # Convert to one-hot encoded tensor
                tensor = char_level_to_tensor(level)
                levels.append(tensor)
    return levels

def char_level_to_tensor(level):
    """Convert character-based level to one-hot encoded tensor"""
    height = len(level)
    width = len(level[0])
    # 5 channels: empty, wall, player, box, goal
    tensor = torch.zeros((5, height, width))
    
    for i, row in enumerate(level):
        for j, char in enumerate(row):
            if char == ' ':
                tensor[0, i, j] = 1  # empty
            elif char == '#':
                tensor[1, i, j] = 1  # wall
            elif char == '@':
                tensor[2, i, j] = 1  # player
            elif char == '$':
                tensor[3, i, j] = 1  # box
            elif char == '.':
                tensor[4, i, j] = 1  # goal
            elif char == '+':  # player on goal
                tensor[2, i, j] = 1
                tensor[4, i, j] = 1
            elif char == '*':  # box on goal
                tensor[3, i, j] = 1
                tensor[4, i, j] = 1
    
    return tensor

def tensor_to_char_level(tensor):
    """Convert tensor back to character-based level"""
    _, height, width = tensor.shape
    level = []
    
    # Get the index of max value along channel dimension
    indices = torch.argmax(tensor, dim=0)
    
    for i in range(height):
        row = ''
        for j in range(width):
            idx = indices[i, j].item()
            if idx == 0:
                row += ' '  # empty
            elif idx == 1:
                row += '#'  # wall
            elif idx == 2:
                if tensor[4, i, j] > 0.5:  # check if also goal
                    row += '+'  # player on goal
                else:
                    row += '@'  # player
            elif idx == 3:
                if tensor[4, i, j] > 0.5:  # check if also goal
                    row += '*'  # box on goal
                else:
                    row += '$'  # box
            elif idx == 4:
                row += '.'  # goal
        level.append(row)
    
    return level

def train_nca(model, levels, num_epochs=100, batch_size=32):
    """Train the NCA model"""
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for epoch in range(num_epochs):
        total_loss = 0
        # Randomly sample batch_size levels
        batch_indices = np.random.choice(len(levels), batch_size)
        
        for idx in batch_indices:
            target = levels[idx].to(device)
            # Start with random noise
            x = torch.rand_like(target).to(device)
            
            # Forward pass
            output = model(x)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/batch_size:.4f}')

def generate_level(model, size=(10, 10)):
    """Generate a new Sokoban level"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        x = torch.rand(5, size[0], size[1]).to(device)
        output = model(x)
        level = tensor_to_char_level(output.cpu())
    return level

if __name__ == "__main__":
    # Parameters
    level_dir = 'data/sokoban_levels/'
    hidden_channels = 16
    num_epochs = 100
    
    # Load levels
    print("Loading levels...")
    levels = load_levels(level_dir)
    print(f"Loaded {len(levels)} levels")
    
    # Create and train model
    model = SokobanNCA(hidden_channels)
    print("Training model...")
    train_nca(model, levels, num_epochs)
    
    # Generate some example levels
    print("\nGenerating example levels...")
    for i in range(3):
        level = generate_level(model)
        print(f"\nGenerated level {i+1}:")
        print('\n'.join(level))