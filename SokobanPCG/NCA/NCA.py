import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from collections import deque
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

class SokobanNCA(nn.Module):
    def __init__(self, hidden_channels=64):
        super().__init__()
        
        self.perception = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels*2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels*2, 3, padding=1),
            nn.BatchNorm2d(hidden_channels*2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels*2, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 5, 1)
        )
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, steps=25):
        batch_size = x.shape[0] if len(x.shape) > 3 else 1
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        original_x = x
        for _ in range(steps):
            dx = self.perception(x)
            dx = self.update(dx)
            
            # Smaller update step
            new_state = original_x + 0.05 * dx
            
            # Lower temperature softmax
            x = self.softmax(new_state * 4.0)
        
        return x.squeeze(0) if batch_size == 1 else x

def load_levels(directory):
    """Load and convert Sokoban levels to tensor format"""
    levels = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r') as f:
                level = f.read().splitlines()
                tensor = char_level_to_tensor(level)
                levels.append(tensor)
    return levels

def filter_levels_by_complexity(levels, max_boxes):
    """Filter levels based on number of boxes"""
    filtered = []
    for level in levels:
        box_count = torch.sum(level[3] > 0.5).item()
        if box_count <= max_boxes:
            filtered.append(level)
    return filtered

def char_level_to_tensor(level):
    """Convert character-based level to one-hot encoded tensor"""
    height = len(level)
    width = len(level[0])
    tensor = torch.zeros((5, height, width))
    
    for i, row in enumerate(level):
        for j, char in enumerate(row):
            if char == ' ':
                tensor[0, i, j] = 1
            elif char == '#':
                tensor[1, i, j] = 1
            elif char == '@':
                tensor[2, i, j] = 1
            elif char == '$':
                tensor[3, i, j] = 1
            elif char == '.':
                tensor[4, i, j] = 1
            elif char == '+':
                tensor[2, i, j] = 1
                tensor[4, i, j] = 1
            elif char == '*':
                tensor[3, i, j] = 1
                tensor[4, i, j] = 1
    
    return tensor

def tensor_to_char_level(tensor):
    """Convert tensor back to character-based level"""
    _, height, width = tensor.shape
    level = []
    
    indices = torch.argmax(tensor, dim=0)
    
    for i in range(height):
        row = ''
        for j in range(width):
            idx = indices[i, j].item()
            if idx == 0:
                row += ' '
            elif idx == 1:
                row += '#'
            elif idx == 2:
                if tensor[4, i, j] > 0.3:
                    row += '+'
                else:
                    row += '@'
            elif idx == 3:
                if tensor[4, i, j] > 0.3:
                    row += '*'
                else:
                    row += '$'
            elif idx == 4:
                row += '.'
        level.append(row)
    
    return level

def validate_level(tensor, training=False):
    """Modified validation with more lenient training thresholds"""
    player_count = torch.sum(tensor[2] > 0.3).item()
    box_count = torch.sum(tensor[3] > 0.3).item()
    goal_count = torch.sum(tensor[4] > 0.3).item()
    
    overlapping = torch.sum(tensor[2:], dim=0) > 1.3
    has_overlaps = overlapping.any().item()
    
    if training:
        valid = (
            0.5 <= player_count <= 1.5 and
            abs(box_count - goal_count) < 1.0 and
            0.5 <= box_count <= 4.5 and
            not has_overlaps
        )
    else:
        valid = (
            0.9 <= player_count <= 1.1 and
            abs(box_count - goal_count) < 0.1 and
            1 <= box_count <= 4 and
            not has_overlaps
        )
    
    return valid

def level_constraints_loss(output, target):
    """Stabilized loss function"""
    # Base reconstruction loss with clipping
    mse_loss = torch.clamp(nn.MSELoss()(output, target), max=10.0)
    
    # Get counts with stable thresholding
    player_count = torch.sum(output[:, 2] > 0.3, dim=(1,2)).float()
    box_count = torch.sum(output[:, 3] > 0.3, dim=(1,2)).float()
    goal_count = torch.sum(output[:, 4] > 0.3, dim=(1,2)).float()
    
    # Clamped penalties to prevent explosion
    excess_player_penalty = torch.mean(torch.clamp(torch.pow(torch.relu(player_count - 1), 2), max=5.0))
    no_player_penalty = torch.mean(torch.clamp(torch.pow(torch.relu(1 - player_count), 2), max=5.0))
    
    # Box-goal matching with clamping
    box_goal_diff_penalty = torch.mean(torch.clamp(torch.abs(box_count - goal_count), max=3.0))
    
    # Constrained penalties
    count_penalty = torch.mean(torch.clamp(torch.relu(box_count - 4), max=2.0))
    min_count_penalty = torch.mean(torch.clamp(torch.relu(1 - box_count), max=2.0))
    
    # Controlled overlap penalty
    overlap_penalty = torch.mean(torch.clamp(
        torch.relu(torch.sum(output[:, 2:], dim=1) - 1.0), max=2.0))
    
    total_loss = (mse_loss + 
                 excess_player_penalty + 
                 no_player_penalty + 
                 box_goal_diff_penalty + 
                 count_penalty + 
                 min_count_penalty + 
                 overlap_penalty)
    
    return torch.clamp(total_loss, max=10.0)  # Final clamping for stability

def generate_structured_noise(shape):
    """Improved structured noise generation with better initialization"""
    noise = torch.zeros(shape)
    
    # Definite walls at borders with some randomness
    border_strength = 0.9 + torch.rand(1).item() * 0.1
    noise[1, 0, :] = border_strength
    noise[1, -1, :] = border_strength
    noise[1, :, 0] = border_strength
    noise[1, :, -1] = border_strength
    
    # Clear middle area for gameplay
    noise[0, 2:-2, 2:-2] = 0.8  # Strong bias for empty space in middle
    
    # Add player with high certainty
    px, py = shape[1]//2, shape[2]//2
    noise[2, px, py] = 0.9
    noise[0, px, py] = 0.0
    
    # Add initial box-goal pairs with high certainty
    num_pairs = np.random.randint(1, 3)
    for i in range(num_pairs):
        offset = i + 1
        # Box placement
        bx, by = px + offset, py
        noise[3, bx, by] = 0.9
        noise[0, bx, by] = 0.0
        
        # Goal placement
        gx, gy = bx + 1, by
        noise[4, gx, gy] = 0.9
        noise[0, gx, gy] = 0.0
    
    # Add minimal random noise
    noise += torch.rand(shape) * 0.05
    
    return noise

def is_valid_pos(level_chars, pos):
    """Check if a position is valid and not a wall"""
    return (0 <= pos[0] < len(level_chars) and 
            0 <= pos[1] < len(level_chars[0]) and 
            level_chars[pos[0]][pos[1]] != '#')

def get_player_moves(level_chars, pos, boxes):
    """Get all possible player positions considering boxes"""
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    visited = {pos}
    positions = {pos}
    queue = [pos]
    
    while queue:
        current = queue.pop(0)
        for dx, dy in moves:
            new_pos = (current[0] + dx, current[1] + dy)
            if (is_valid_pos(level_chars, new_pos) and 
                new_pos not in visited and 
                new_pos not in boxes):
                visited.add(new_pos)
                positions.add(new_pos)
                queue.append(new_pos)
    return positions

def get_state_hash(player, boxes):
    """Create a unique hash for the current game state"""
    return (player, frozenset(boxes))

def solve_level(level_chars, player_pos, boxes, goals, max_states=10000, timeout=2.0):
    """Try to solve the level using BFS with state limit"""
    start_time = time.time()
    initial_state = get_state_hash(player_pos, boxes)
    visited = {initial_state}
    queue = [(player_pos, boxes, 0)]  # Include depth for limiting search
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    
    while queue and len(visited) < max_states:
        if time.time() - start_time > timeout:
            return False  # Timeout reached
        player, current_boxes, depth = queue.pop(0)
        
        # Check if we've reached a solution
        if all(box in goals for box in current_boxes):
            return True
        
        # Get all positions the player can reach
        player_positions = get_player_moves(level_chars, player, current_boxes)
        
        # Try to push each box
        for box in current_boxes:
            for dx, dy in moves:
                # Position player needs to be in to push the box
                push_pos = (box[0] - dx, box[1] - dy)
                
                # New position for the box after pushing
                new_box_pos = (box[0] + dx, box[1] + dy)
                
                # Check if push is possible
                if (push_pos in player_positions and 
                    is_valid_pos(level_chars, new_box_pos) and 
                    new_box_pos not in current_boxes):
                    
                    # Create new box configuration
                    new_boxes = set(current_boxes)
                    new_boxes.remove(box)
                    new_boxes.add(new_box_pos)
                    
                    # Check if this state has been seen
                    new_state = get_state_hash(box, new_boxes)
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((box, new_boxes, depth + 1))
    
    return False

def check_solvable(level_chars):
    """Advanced Sokoban solvability checker with proper pushing mechanics"""
    # Find player, boxes, and goals
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

    # Basic validation
    if not player_pos or not boxes or not goals:
        return False
    if len(boxes) != len(goals):
        return False

    # Quick check for obviously stuck boxes
    for box in boxes:
        i, j = box
        if box not in goals and (  # Only check boxes not already on goals
            (level_chars[i-1][j] == '#' and level_chars[i][j-1] == '#') or  # Top-left corner
            (level_chars[i-1][j] == '#' and level_chars[i][j+1] == '#') or  # Top-right corner
            (level_chars[i+1][j] == '#' and level_chars[i][j-1] == '#') or  # Bottom-left corner
            (level_chars[i+1][j] == '#' and level_chars[i][j+1] == '#')):   # Bottom-right corner
            return False

    # Try to solve the level
    return solve_level(level_chars, player_pos, boxes, goals)

def train_nca(model, levels, num_epochs=300, batch_size=16):
    """Enhanced training function with stability measures"""
    # Initialize optimizer with conservative learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=15, 
        verbose=True, 
        min_lr=1e-6
    )
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Prepare dataset
    level_tensors = torch.stack(levels)
    dataset = torch.utils.data.TensorDataset(level_tensors)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize tracking variables
    best_loss = float('inf')
    best_valid_rate = 0
    patience_counter = 0
    best_model_state = None
    last_save_epoch = 0
    min_loss_for_save = float('inf')
    
    print(f"Starting training on device: {device}")
    print(f"Total levels: {len(levels)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    
    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            epoch_valid_levels = 0
            epoch_solvable_levels = 0
            total_generated = 0
            batch_count = 0
            
            # Training loop
            for batch in dataloader:
                try:
                    target = batch[0].to(device)
                    batch_size = target.shape[0]
                    total_generated += batch_size
                    
                    # Generate noise for each sample
                    x = torch.stack([generate_structured_noise(target.shape[1:]) 
                                   for _ in range(batch_size)]).to(device)
                    
                    # Forward pass with fixed steps for stability
                    output = model(x, steps=25)
                    loss = level_constraints_loss(output, target)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(f"Warning: NaN loss detected in batch {batch_count}, skipping...")
                        continue
                    
                    # Gradient computation and update
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Clip gradients for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                    
                    # Periodic validation during training
                    if batch_count % 10 == 0:
                        model.eval()
                        with torch.no_grad():
                            test_noise = generate_structured_noise((5, 10, 10))
                            test_output = model(test_noise.to(device), steps=25)
                            if validate_level(test_output):
                                print(f"Valid level generated during training at batch {batch_count}")
                        model.train()
                    
                    # Check valid and solvable levels
                    for i in range(batch_size):
                        if validate_level(output[i], training=True):
                            epoch_valid_levels += 1
                            level_chars = tensor_to_char_level(output[i].cpu())
                            if check_solvable(level_chars):
                                epoch_solvable_levels += 1
                
                except RuntimeError as e:
                    print(f"Error in batch {batch_count}: {str(e)}")
                    continue
            
            # Skip epoch statistics if no successful batches
            if batch_count == 0:
                print(f"Epoch {epoch + 1} had no successful batches, continuing...")
                continue
            
            # Compute epoch statistics
            avg_loss = total_loss / batch_count
            valid_rate = epoch_valid_levels / total_generated if total_generated > 0 else 0
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save best model
            if valid_rate > best_valid_rate or (valid_rate == best_valid_rate and avg_loss < best_loss):
                best_loss = avg_loss
                best_valid_rate = valid_rate
                best_model_state = model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'valid_rate': valid_rate,
                }, 'best_model.pth')
                print(f"New best model saved with valid rate: {valid_rate:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Periodic progress update
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Loss: {avg_loss:.4f}, '
                      f'Valid Rate: {valid_rate:.4f}, '
                      f'Solvable: {epoch_solvable_levels}/{epoch_valid_levels}, '
                      f'LR: {current_lr:.6f}')
            
            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'valid_rate': valid_rate,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping check
            if patience_counter >= 20:
                print("No improvement for 20 epochs. Restoring best model and stopping...")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
            
            # Learning rate minimum check
            if current_lr <= scheduler.min_lrs[0]:
                print("Learning rate reached minimum. Stopping training...")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current state...")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss if 'avg_loss' in locals() else None,
        }, 'interrupted_training.pth')
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
    finally:
        # Ensure we're using the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"Training completed. Best valid rate: {best_valid_rate:.4f}")
        else:
            print("Training completed but no best model was saved.")

    return model

if __name__ == "__main__":
    level_dir = 'data/sokoban_levels/'
    hidden_channels = 128  # Increased channel count for more capacity
    
    # Create directories for saving generated levels
    if not os.path.exists('generated_levels'):
        os.makedirs('generated_levels')
        os.makedirs('generated_levels/valid')
        os.makedirs('generated_levels/solvable')
    
    # Load training levels
    print("Loading training levels...")
    all_levels = load_levels(level_dir)
    print(f"Loaded {len(all_levels)} training levels")
    
    # Create model
    model = SokobanNCA(hidden_channels)
    
    # Single phase training with longer duration
    print("\nTraining model...")
    print(f"Training on {len(all_levels)} levels")
    train_nca(model, all_levels, num_epochs=300)  # Increased epochs for better learning
    
    # Generate new levels
    print("\nGenerating new levels...")
    model.eval()
    num_attempts = 100
    valid_count = 0
    solvable_count = 0
    
    for i in range(num_attempts):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            # Generate multiple attempts per iteration
            for _ in range(5):  # Try 5 times per attempt
                noise = generate_structured_noise((5, 10, 10))
                x = noise.to(device)
                output = model(x, steps=30)  # Increased steps for generation
                level_chars = tensor_to_char_level(output.cpu())
                
                is_valid = validate_level(output)
                if is_valid:
                    valid_count += 1
                    valid_filename = f'generated_levels/valid/level_{valid_count}.txt'
                    with open(valid_filename, 'w') as f:
                        f.write('\n'.join(level_chars))
                    
                    if check_solvable(level_chars):
                        solvable_count += 1
                        solvable_filename = f'generated_levels/solvable/level_{solvable_count}.txt'
                        with open(solvable_filename, 'w') as f:
                            f.write('\n'.join(level_chars))
                        print(f"\nFound solvable level (#{solvable_count}):")
                        print('\n'.join(level_chars))
                        break  # Move to next attempt if we found a solvable level
        
        if (i + 1) % 10 == 0:
            print(f"\nProgress: {i+1}/{num_attempts}")
            print(f"Valid levels: {valid_count}")
            print(f"Solvable levels: {solvable_count}")
            
        # Early success check
        if solvable_count >= 10:  # If we've found enough good levels
            print("\nFound sufficient solvable levels, stopping generation.")
            break
    
    print("\nGeneration completed:")
    print(f"Total attempts: {i+1}")
    print(f"Total valid levels: {valid_count}")
    print(f"Total solvable levels: {solvable_count}")