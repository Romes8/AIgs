from csv import writer
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard visualization

# Define the Neural Cellular Automata (NCA) model
class SokobanNCA(nn.Module):
    def __init__(self, hidden_channels=32):
        super().__init__()
        
        # Perception layer
        self.perception = nn.Conv2d(5, hidden_channels, 3, padding=1)
        
        # Update rule with increased capacity and layers
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 5, 1)
        )
        
        # Activation function using Softmax for mutually exclusive channels
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, steps=15):
        for _ in range(steps):
            dx = self.perception(x)
            dx = self.update(dx)
            x = self.softmax(x + dx)  # Apply Softmax for mutually exclusive channels
        return x

# Weight initialization using He Initialization
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

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
            # Check for overlapping goal channel
            goal_present = tensor[4, i, j] > 0.5
            if idx == 0:
                row += ' '  # empty
            elif idx == 1:
                row += '#'  # wall
            elif idx == 2:
                if goal_present:
                    row += '+'  # player on goal
                else:
                    row += '@'  # player
            elif idx == 3:
                if goal_present:
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
    height = len(level)
    width = len(level[0])
    
    # Ensure we don't go out of bounds
    if i <= 0 or i >= height-1 or j <= 0 or j >= width-1:
        return False  # Cannot form a corner if on the edge
    
    # Check for wall patterns that create corners
    if (level[i][j] in '$*' and  # Box position
        (((level[i-1][j] == '#' and level[i][j-1] == '#') or  # Top-left corner
         (level[i-1][j] == '#' and level[i][j+1] == '#') or  # Top-right corner
         (level[i+1][j] == '#' and level[i][j-1] == '#') or  # Bottom-left corner
         (level[i+1][j] == '#' and level[i][j+1] == '#')))):  # Bottom-right corner
        return True
    return False

def validate_level(tensor):
    # Count elements in each relevant channel
    player_count = torch.sum(tensor[2] > 0.5).item()  # Player channel
    box_count = torch.sum(tensor[3] > 0.5).item()     # Box channel
    goal_count = torch.sum(tensor[4] > 0.5).item()    # Goal channel

    # Check validation rules
    is_valid = (
        player_count == 1 and                 # Exactly one player
        1 <= box_count <= 4 and               # Between one and four boxes
        box_count == goal_count and           # Equal number of boxes and goals
        box_count > 0                          # At least one box
    )

    # Print a confirmation only when the level is valid
    if is_valid:
        print("Level is valid")

    return is_valid

def level_constraints_loss(output, target):
    # Base MSE loss
    mse_loss = nn.MSELoss()(output, target)
    
    # Penalties for constraint violations based on **hard counts**
    player_count = torch.sum(output[2] > 0.5)
    box_count = torch.sum(output[3] > 0.5)
    goal_count = torch.sum(output[4] > 0.5)
    
    # **Substantially Increased Penalty Weights**
    player_penalty = torch.abs(player_count - 1) * 10.0  # Strong penalty for player count ≠ 1
    box_goal_diff_penalty = torch.abs(box_count - goal_count) * 10.0  # Strong penalty for mismatched boxes/goals
    count_penalty = torch.clamp(box_count - 4, min=0) * 5.0  # Moderate penalty for too many boxes
    
    # Adjusted spacing penalty to balance overall loss
    spacing_penalty = torch.mean(output[:, 1:-1, 1:-1]) * 0.05  # Encourages sparsity
    
    # Total enhanced loss
    total_loss = mse_loss + player_penalty + box_goal_diff_penalty + count_penalty + spacing_penalty
    
    return total_loss

def generate_structured_noise(shape):
    """
    Generate structured noise as initial state with 100% walls on the edges and enhanced initial presence of player and boxes.
    """
    noise = torch.zeros(shape)
    
    # **100% Walls on the Edges**
    # Top and Bottom Walls
    noise[1, 0, :] = 1.0  # Top wall
    noise[1, -1, :] = 1.0  # Bottom wall
    
    # Left and Right Walls
    noise[1, :, 0] = 1.0  # Left wall
    noise[1, :, -1] = 1.0  # Right wall
    
    # Add random interior walls (20% chance)
    interior_walls = torch.rand((shape[1]-2, shape[2]-2)) < 0.2
    noise[1, 1:-1, 1:-1] = interior_walls.float()
    
    # **Enhanced Initial Presence of Player, Boxes, and Goals**
    # Increase the initial probabilities
    noise[2, :, :] = torch.rand((shape[1], shape[2])) * 0.3  # Player channel increased from 0.1 to 0.3
    noise[3, :, :] = torch.rand((shape[1], shape[2])) * 0.3  # Box channel increased from 0.1 to 0.3
    noise[4, :, :] = torch.rand((shape[1], shape[2])) * 0.3  # Goal channel increased from 0.1 to 0.3
    
    return noise

def tensor_to_image(tensor):
    """Convert tensor to an RGB image for TensorBoard visualization"""
    _, height, width = tensor.shape
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Define colors for each channel
    colors = {
        0: [1.0, 1.0, 1.0],  # Empty - White
        1: [0.0, 0.0, 0.0],  # Wall - Black
        2: [0.0, 0.0, 1.0],  # Player - Blue
        3: [1.0, 0.0, 0.0],  # Box - Red
        4: [0.0, 1.0, 0.0],  # Goal - Green
    }
    
    # Assign colors based on the channel with the highest activation
    indices = torch.argmax(tensor, dim=0)
    for i in range(height):
        for j in range(width):
            channel = indices[i, j].item()
            image[i, j] = colors[channel]
    
    return image.transpose(2, 0, 1)  # Convert to (C, H, W)

def train_nca(model, levels, num_epochs=1000, batch_size=32):
    """Train the NCA model with constraints"""
    # Adjusted Learning Rate and Added Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Further reduced LR
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)  # Halve LR every 200 epochs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/sokoban_nca')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        valid_levels = 0
        batch_indices = np.random.choice(len(levels), batch_size)
        
        for idx in batch_indices:
            target = levels[idx].to(device)
            # Start with structured noise
            x = generate_structured_noise(target.shape).to(device)
            
            # Forward pass with random number of steps for variety
            steps = np.random.randint(10, 20)
            output = model(x, steps=steps)
            loss = level_constraints_loss(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check if generated level is valid
            if validate_level(output):
                valid_levels += 1
        
        # Step the scheduler
        scheduler.step()
        
        avg_loss = total_loss / batch_size
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Valid Levels', valid_levels, epoch)
        
        # Log gradients for monitoring
        for name, param in model.named_parameters():
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad.clone().cpu().numpy(), epoch)
        
        # Periodically visualize a generated level
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                sample_idx = np.random.randint(len(levels))
                sample = levels[sample_idx].to(device).unsqueeze(0)  # Add batch dimension
                generated = model(sample, steps=15).squeeze(0)
                image = tensor_to_image(generated.cpu())
                writer.add_image('Generated_Level', torch.tensor(image), epoch)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Valid Levels: {valid_levels}/{batch_size}')
    
    writer.close()

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
    level_dir = 'data/sokoban_levels/'  # Update this path to your levels directory
    hidden_channels = 32
    num_epochs = 200
    batch_size = 32  # Define batch size for training
    
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
    
    # Initialize the model and apply weight initialization
    model = SokobanNCA(hidden_channels)
    initialize_weights(model)
    
    print("Training model...")
    train_nca(model, solvable_levels, num_epochs, batch_size)
    
    print("\nGenerating example levels...")
    generated_levels = 0
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
        
        if is_valid and is_solvable:
            generated_levels += 1
    
    print(f"\nSuccessfully generated {generated_levels} valid and solvable levels out of 5 attempts.")