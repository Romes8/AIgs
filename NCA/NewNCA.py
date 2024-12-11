import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import random
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# ------------------- LOADING LEVELS ----------------------
class SokobanDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.grid_size = (10, 10)
        self.char_to_index = {
            '#': 0,  # wall
            ' ': 1,  # floor
            '@': 2,  # player
            '$': 3,  # box
            '.': 4,  # goal
        }
        
        self.levels = self.load_levels()
        self.processed_levels = self._preprocess_levels()
    
    def load_levels(self):
        levels = []
        for file_path in self.data_dir.glob('*'):
            with open(file_path, 'r') as f:
                level_data = f.read().strip()
                levels.append(level_data)
        return levels
    
    def _preprocess_levels(self):
        processed = []
        for level in self.levels:
            tensor_level = torch.zeros((5, *self.grid_size))
            
            rows = level.split('\n')
            for i, row in enumerate(rows):
                for j, char in enumerate(row):
                    if char in self.char_to_index:
                        channel = self.char_to_index[char]
                        tensor_level[channel, i, j] = 1.0
                        
                        if char in '@$.':
                            tensor_level[1, i, j] = 1.0
            
            processed.append(tensor_level)
        
        return processed
    
    def __len__(self):
        return len(self.processed_levels)
    
    def __getitem__(self, idx):
        return self.processed_levels[idx]

def analyze_level(tensor_level):
    stats = {
        'walls': tensor_level[0].sum().item(),
        'floors': tensor_level[1].sum().item(),
        'players': tensor_level[2].sum().item(),
        'boxes': tensor_level[3].sum().item(),
        'goals': tensor_level[4].sum().item()
    }
    return stats

# -------------------------- TRAINING FUNCTIONS ---------------------------
class SokobanNCA(nn.Module):
    def __init__(self, hidden_channels=128, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.input_channels = 5
        self.grid_size = (10, 10)
        self.device = device
        
        self.perception = nn.Sequential(
            nn.Conv2d(self.input_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU()
        )
        
        self.update = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),  
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.input_channels, 1)
        )

    def forward(self, x, steps=50):
        batch = x.shape[0]
        
        if x is None:
            x = torch.zeros(batch, self.input_channels, *self.grid_size).to(self.device)
            x[:, 0, 0, :] = 1.0
            x[:, 0, -1, :] = 1.0
            x[:, 0, :, 0] = 1.0
            x[:, 0, :, -1] = 1.0
            x[:, :, 1:-1, 1:-1] = torch.rand(batch, self.input_channels, 8, 8).to(self.device) * 0.1
        
        states = []
        
        for step in range(steps):
            dx = self.perception(x)
            dx = self.update(dx)
            
            x = x + dx
            
            temperature = max(1.0 - step/steps, 0.5)
            x = torch.softmax(x / temperature, dim=1)
            
            states.append(x.clone())
        
        return x, states

class SokobanLoss(nn.Module):
    def __init__(self, target_boxes=3):
        super().__init__()
        self.target_boxes = target_boxes
        self.w_recon = 1.0
        self.w_count = 0.1  
        self.w_border = 0.5
        self.w_diversity = 2.0 
        
        self.eps = 1e-8
        
        self.target_props = {
            'walls': 0.35,    # ~35% walls
            'floors': 0.45,   # ~45% floors
            'player': 0.02,   # 1 player
            'boxes': 0.08,    # ~3 boxes
            'goals': 0.10     # ~3 goals
        }
    
    def count_objects(self, x):
        temp = 10.0
        x_temp = x * temp
        soft_x = F.softmax(x_temp, dim=1)
        
        counts = {
            'walls': soft_x[:, 0].sum(dim=(1,2)),
            'floors': soft_x[:, 1].sum(dim=(1,2)),
            'players': soft_x[:, 2].sum(dim=(1,2)),
            'boxes': soft_x[:, 3].sum(dim=(1,2)),
            'goals': soft_x[:, 4].sum(dim=(1,2))
        }
        return counts
    
    def diversity_loss(self, x):
        soft_x = F.softmax(x, dim=1)
        batch_size = x.shape[0]
        
        props = soft_x.mean(dim=(2,3))  # [batch_size, 5]
        
        target = torch.tensor([
            self.target_props['walls'],
            self.target_props['floors'],
            self.target_props['player'],
            self.target_props['boxes'],
            self.target_props['goals']
        ]).to(x.device).expand(batch_size, -1)
        
        div_loss = F.mse_loss(props, target)
        
        wall_excess_penalty = F.relu(props[:, 0] - self.target_props['walls']).mean()
        
        floor_deficit_penalty = F.relu(self.target_props['floors'] - props[:, 1]).mean()
        
        return div_loss + wall_excess_penalty + floor_deficit_penalty
    
    def forward(self, pred, target):
        recon_loss = F.mse_loss(pred + self.eps, target + self.eps)
        
        pred_counts = self.count_objects(pred)
        target_counts = self.count_objects(target)
        
        count_loss = (
            F.mse_loss(pred_counts['players'], torch.ones_like(pred_counts['players'])) * 2.0 +  # Emphasize player count
            F.mse_loss(pred_counts['boxes'], pred_counts['goals']) +
            F.mse_loss(pred_counts['boxes'], self.target_boxes * torch.ones_like(pred_counts['boxes']))
        )
        
        border_loss = F.mse_loss(pred[:, 0, 0, :], torch.ones_like(pred[:, 0, 0, :])) + \
                     F.mse_loss(pred[:, 0, -1, :], torch.ones_like(pred[:, 0, -1, :])) + \
                     F.mse_loss(pred[:, 0, :, 0], torch.ones_like(pred[:, 0, :, 0])) + \
                     F.mse_loss(pred[:, 0, :, -1], torch.ones_like(pred[:, 0, :, -1]))
        
        div_loss = self.diversity_loss(pred)
        
        total_loss = (
            self.w_recon * recon_loss + 
            self.w_count * count_loss + 
            self.w_border * border_loss +
            self.w_diversity * div_loss
        )
        
        return total_loss, {
            'recon_loss': recon_loss.item(),
            'count_loss': count_loss.item(),
            'border_loss': border_loss.item(),
            'diversity_loss': div_loss.item(),
            'weighted_total': total_loss.item(),
            'raw_total': recon_loss.item() + count_loss.item() + border_loss.item() + div_loss.item()
        }
    
class SokobanTrainer:
    def __init__(self, model, dataset, device, 
                 batch_size=32, 
                 learning_rate=1e-4,
                 save_dir='checkpoints'):
        self.device = device
        self.model = model.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.loss_fn = SokobanLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        self.history = {
            'epoch_loss': [],
            'recon_loss': [],
            'count_loss': [],
            'border_loss': [],
            'diversity_loss': []
        }
    
    def train_epoch(self):
        self.model.train()
        epoch_losses = []
        epoch_components = {
            'recon_loss': [], 
            'count_loss': [], 
            'border_loss': [],
            'diversity_loss': []
        }
        
        for batch in tqdm(self.dataloader, desc='Training'):
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            output, _ = self.model(batch)
            loss, components = self.loss_fn(output, batch)
            
            loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(components['weighted_total'])
            for k in epoch_components.keys():
                epoch_components[k].append(components[k])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_components = {k: sum(v) / len(v) for k, v in epoch_components.items()}
        
        self.history['epoch_loss'].append(avg_loss)
        for k, v in avg_components.items():
            self.history[k].append(v)
            
        return avg_loss, avg_components
    
    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, self.save_dir / f'checkpoint_epoch_{epoch}.pt')
    
    def generate_sample(self, steps=50):
        self.model.eval()
        with torch.no_grad():
            x = torch.rand(1, 5, 10, 10).to(self.device)
            output, states = self.model(x, steps=steps)
            return output[0], states
    
    def visualize_sample(self, sample, threshold=0.25):
        probs = F.softmax(sample, dim=0)
        
        chars = {0: '#', 1: ' ', 2: '@', 3: '$', 4: '.'}
        level_str = ''
        debug_str = ''
        
        for i in range(10):
            for j in range(10):
                max_prob = torch.max(probs[:, i, j])
                max_idx = torch.argmax(probs[:, i, j])
                
                if max_prob > threshold:
                    level_str += chars[max_idx.item()]
                else:
                    level_str += '?'
                    
                debug_str += f"({i},{j}): "
                for k in range(5):
                    debug_str += f"{chars[k]}:{probs[k,i,j]:.2f} "
                debug_str += '\n'
                
            level_str += '\n'
        
        return level_str, debug_str

def test_model(model, dataset, device):
    """Test if the model can process a single batch correctly"""
    try:
        batch = torch.stack([dataset[i] for i in range(4)]).to(device)
        output, states = model(batch)
        
        print("\nModel test results:")
        print(f"Input shape: {batch.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Number of intermediate states: {len(states)}")
        print(f"State shape: {states[0].shape}")
        
        print(f"\nOutput value range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        loss_fn = SokobanLoss()
        loss, components = loss_fn(output, batch)
        print("\nLoss components:", components)
        
        return True
        
    except Exception as e:
        print(f"Error during model test: {str(e)}")
        return False

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SokobanNCA(device=device)
    dataset = SokobanDataset('data/sokoban_levels')
    trainer = SokobanTrainer(model, dataset, device)
    
    # Training loop
    num_epochs = 50
    save_every = 50
    visualize_every = 10
    
    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            avg_loss, components = trainer.train_epoch()
            
            print(f"Average loss: {avg_loss:.4f}")
            for k, v in components.items():
                print(f"{k}: {v:.4f}")
            
            if (epoch + 1) % visualize_every == 0:
                print("\n" + "="*50)
                print(f"Generated Level at Epoch {epoch+1}:")
                sample, _ = trainer.generate_sample(steps=75)
                level_str, debug_str = trainer.visualize_sample(sample, threshold=0.3)
                print(level_str)
                
                probs = F.softmax(sample, dim=0)
                top2_values, top2_indices = torch.topk(probs, k=2, dim=0)
                uncertainty = top2_values[0] - top2_values[1]
                uncertain_positions = torch.where(uncertainty < 0.2)
                
                if len(uncertain_positions[0]) > 0:
                    print("\nUncertain positions:")
                    for y, x in zip(*uncertain_positions):
                        print(f"Position ({y},{x}): ", end="")
                        for k in range(5):
                            print(f"{['Wall', 'Floor', 'Player', 'Box', 'Goal'][k]}: {probs[k,y,x]:.2f} ", end="")
                        print()
                
                print("="*50 + "\n")
            
            if (epoch + 1) % save_every == 0:
                trainer.save_checkpoint(epoch + 1)
        
        # Save final model
        trainer.save_checkpoint('final')
        
        # Generate 5 levels after training
        print("\nGenerating 5 final levels from the trained model:")
        print("-" * 50)
        for i in range(5):
            sample, _ = trainer.generate_sample(steps=100)
            level_str, _ = trainer.visualize_sample(sample, threshold=0.3)
            print(f"\nFinal Generated Level {i+1}:")
            print(level_str)
            print("-" * 50)
        
        # Plot training history
        plt.figure(figsize=(15, 8))
        for key, values in trainer.history.items():
            plt.plot(values, label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final state...")
        trainer.save_checkpoint('interrupted')
        
        print("\nGenerating 5 levels from the current model state:")
        print("-" * 50)
        for i in range(5):
            sample, _ = trainer.generate_sample(steps=100)
            level_str, _ = trainer.visualize_sample(sample, threshold=0.3)
            print(f"\nGenerated Level {i+1}:")
            print(level_str)
            print("-" * 50)
        
        print("Final state saved.")