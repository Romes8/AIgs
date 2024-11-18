import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SokobanNCA(nn.Module):
    def __init__(self, hidden_channels=64):  # Increased hidden channels
        super().__init__()
        
        # Enhanced perception layer
        self.perception = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        # Enhanced update network
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 5, 1),
            nn.BatchNorm2d(5)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, steps=25):  # Increased steps
        batch_size = x.shape[0] if len(x.shape) > 3 else 1
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            
        original_x = x  # Store original input
        for _ in range(steps):
            dx = self.perception(x)
            dx = self.update(dx)
            x = self.softmax(original_x + 0.1 * dx)  # Reduced update rate
            
        return x.squeeze(0) if batch_size == 1 else x

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

def bfs_pathfinding(level, start, targets):
    """
    Breadth-first search to find path from start to all targets
    Returns True if all targets are reachable
    """
    height = len(level)
    width = len(level[0])
    
    # Valid moves (up, right, down, left)
    moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Keep track of visited positions
    visited = set()
    visited.add(start)
    
    # Queue for BFS
    queue = deque([start])
    found_targets = set()
    
    while queue and len(found_targets) < len(targets):
        current = queue.popleft()
        
        # Check if current position is a target
        if current in targets:
            found_targets.add(current)
        
        # Try all possible moves
        for dx, dy in moves:
            new_x = current[0] + dx
            new_y = current[1] + dy
            
            # Check if move is valid
            if (0 <= new_x < height and 
                0 <= new_y < width and 
                level[new_x][new_y] != '#' and 
                (new_x, new_y) not in visited):
                
                visited.add((new_x, new_y))
                queue.append((new_x, new_y))
    
    return len(found_targets) == len(targets)

def check_solvable(level_chars):
    """
    Check if a Sokoban level is potentially solvable:
    1. Player can reach all boxes
    2. All boxes can reach all goals
    3. No boxes are stuck in corners
    """
    # Find player, boxes, and goals positions
    player_pos = None
    boxes = set()
    goals = set()
    
    for i, row in enumerate(level_chars):
        for j, char in enumerate(row):
            if char in '@+':  # Player or player on goal
                player_pos = (i, j)
            if char in '$*':  # Box or box on goal
                boxes.add((i, j))
            if char in '.*+':  # Goal, box on goal, or player on goal
                goals.add((i, j))
    
    if not player_pos or not boxes or not goals:
        return False
    
    # Check if player can reach all boxes
    if not bfs_pathfinding(level_chars, player_pos, boxes):
        return False
    
    # Check if boxes can reach goals
    for box in boxes:
        if not bfs_pathfinding(level_chars, box, goals):
            return False
    
    # Check for boxes stuck in corners
    for box in boxes:
        i, j = box
        if is_corner(level_chars, i, j):
            return False
    
    return True

def is_corner(level, i, j):
    """Check if position is a corner where boxes can get stuck"""
    # Check for wall patterns that create corners
    if (level[i][j] in '$*' and  # Box position
        ((level[i-1][j] == '#' and level[i][j-1] == '#') or  # Top-left corner
         (level[i-1][j] == '#' and level[i][j+1] == '#') or  # Top-right corner
         (level[i+1][j] == '#' and level[i][j-1] == '#') or  # Bottom-left corner
         (level[i+1][j] == '#' and level[i][j+1] == '#'))):  # Bottom-right corner
        return True
    return False

def validate_level(tensor):
    """Validate if the level meets basic requirements"""
    # Count elements
    player_count = torch.sum(tensor[2] > 0.5).item()  # Player channel
    box_count = torch.sum(tensor[3] > 0.5).item()     # Box channel
    goal_count = torch.sum(tensor[4] > 0.5).item()    # Goal channel
    
    # Basic validation rules
    valid = (
        player_count == 1 and        # Exactly one player
        1 <= box_count <= 4 and      # 1-4 boxes
        box_count == goal_count and  # Equal number of boxes and goals
        box_count > 0                # At least one box-goal pair
    )
    
    return valid

def level_constraints_loss(output, target):
    """Enhanced loss function with better weighted constraints"""
    # Base loss with higher weight on important features
    mse_loss = nn.MSELoss()(output, target) * 2.0
    
    # Penalties for constraint violations
    player_count = torch.sum(output[:, 2], dim=(1,2))
    box_count = torch.sum(output[:, 3], dim=(1,2))
    goal_count = torch.sum(output[:, 4], dim=(1,2))
    
    # Stronger penalties
    player_penalty = torch.mean(torch.abs(player_count - 1)) * 0.5
    box_goal_diff_penalty = torch.mean(torch.abs(box_count - goal_count)) * 0.5
    count_penalty = torch.mean(torch.relu(box_count - 4)) * 0.3
    
    # Spatial coherence penalty
    wall_penalty = torch.mean(torch.abs(
        output[:, 1, 1:-1, 1:-1] - output[:, 1, :-2, 1:-1]
    )) * 0.2
    
    # Element distribution penalty
    distribution_penalty = torch.mean(torch.abs(
        output[:, 2:, 1:-1, 1:-1].sum(dim=1) - 1
    )) * 0.3
    
    # Combine all penalties into a single scalar value
    total_loss = (mse_loss + 
                 player_penalty + 
                 box_goal_diff_penalty + 
                 count_penalty + 
                 wall_penalty + 
                 distribution_penalty)
    
    return total_loss

def generate_structured_noise(shape):
    """Generate more structured initial noise"""
    noise = torch.zeros(shape)
    
    # Base noise with lower variance
    noise = torch.rand(shape) * 0.05
    
    # Add walls with high probability at borders
    noise[1, 0, :] = torch.rand(shape[2]) * 0.3 + 0.7  # Top wall
    noise[1, -1, :] = torch.rand(shape[2]) * 0.3 + 0.7  # Bottom wall
    noise[1, :, 0] = torch.rand(shape[1]) * 0.3 + 0.7  # Left wall
    noise[1, :, -1] = torch.rand(shape[1]) * 0.3 + 0.7  # Right wall
    
    # Add some structured internal patterns
    for _ in range(np.random.randint(2, 5)):
        x = np.random.randint(1, shape[1]-2)
        y = np.random.randint(1, shape[2]-2)
        size = np.random.randint(1, 3)
        noise[1, x:x+size, y:y+size] = torch.rand(size, size) * 0.5 + 0.5
    
    # Add hints for player and box positions
    center_x, center_y = shape[1] // 2, shape[2] // 2
    noise[2, center_x, center_y] = 0.3  # Player hint
    noise[3, center_x+1, center_y] = 0.2  # Box hint
    
    return noise

def train_nca(model, levels, num_epochs=300, batch_size=64):
    """Enhanced training with learning rate scheduling and gradient clipping"""
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Convert levels to tensor dataset for better batch processing
    level_tensors = torch.stack(levels)
    dataset = torch.utils.data.TensorDataset(level_tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_levels = 0
        batch_count = 0
        
        for batch in dataloader:
            target = batch[0].to(device)
            batch_size = target.shape[0]
            
            # Generate different noise for each sample
            x = torch.stack([generate_structured_noise(target.shape[1:]) 
                           for _ in range(batch_size)]).to(device)
            
            # Forward pass with random steps for robustness
            steps = np.random.randint(20, 30)
            output = model(x, steps=steps)
            loss = level_constraints_loss(output, target)
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            
            # Track valid levels
            for i in range(batch_size):
                if validate_level(output[i]):
                    valid_levels += 1
        
        avg_loss = total_loss / batch_count
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_sokoban_nca.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, '
                  f'Valid Levels: {valid_levels}/{len(level_tensors)}, '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
def generate_level(model, size=(10, 10), max_attempts=20):
    """Generate a new Sokoban level with solvability check"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    for attempt in range(max_attempts):
        with torch.no_grad():
            x = generate_structured_noise((5, size[0], size[1])).to(device)
            output = model(x)
            
            if validate_level(output):
                level_chars = tensor_to_char_level(output.cpu())
                if check_solvable(level_chars):
                    print(f"Found valid and solvable level after {attempt + 1} attempts")
                    return level_chars
    
    print("Failed to generate a valid and solvable level")
    return None

if __name__ == "__main__":
    level_dir = 'data/sokoban_levels/'
    hidden_channels = 32
    num_epochs = 200
    
    print("Loading levels...")
    levels = load_levels(level_dir)
    print(f"Loaded {len(levels)} levels")
    
    # Filter training levels to include only solvable ones
    print("Filtering solvable levels...")
    solvable_levels = []
    for level_tensor in levels:
        level_chars = tensor_to_char_level(level_tensor)
        if check_solvable(level_chars):
            solvable_levels.append(level_tensor)
    print(f"Found {len(solvable_levels)} solvable levels out of {len(levels)}")
    
    model = SokobanNCA(hidden_channels)
    print("Training model...")
    train_nca(model, solvable_levels, num_epochs)
    
    print("\nGenerating example levels...")
    attempts = 0
    generated_count = 0
    print("\nGenerating example levels...")
    for i in range(5):  # Generate 5 different levels
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # Add more randomness to the initial noise
            noise = generate_structured_noise((5, 10, 10))
            noise += torch.randn_like(noise) * 0.2  # Add random perturbations
            x = noise.to(device)
            
            # Add random number of steps for more variety
            steps = np.random.randint(10, 20)
            output = model(x, steps=steps)
            level_chars = tensor_to_char_level(output.cpu())
            
            print(f"\nGenerated level {i+1}:")
            print('\n'.join(level_chars))
            
            # Check and print if it's valid and solvable
            is_valid = validate_level(output)
            is_solvable = check_solvable(level_chars) if is_valid else False
            print(f"Valid: {is_valid}")
            print(f"Solvable: {is_solvable}")
            
            # Save the level
            with open(f'generated_level_{i+1}.txt', 'w') as f:
                f.write('\n'.join(level_chars))
        
        attempts += 1