import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.image as mpimg  
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.decomposition import PCA

# Define the object type mappings for encoding levels
OBJECT_TYPES = {
    'empty': 0,
    'wall': 1,
    'target': 2,
    'agent': 3,
    'box': 4
}

# Load the assets (images)
assets = {
    'empty': np.zeros((32, 32, 3), dtype=np.uint8),
    'wall': mpimg.imread('assets/wall.png'),
    'target': mpimg.imread('assets/box_target.png'),
    'agent': mpimg.imread('assets/agent.png'),
    'box': mpimg.imread('assets/box.png'),
}


# Function to encode a level into a 3D tensor
def encode_level(grid):
    encoded = jnp.zeros((10, 10, 5), dtype=jnp.uint8)  # 10x10 grid, 5 channels

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j, 1] == 1:  # Wall
                encoded = encoded.at[i, j, OBJECT_TYPES['wall']].set(1)
            elif grid[i, j, 1] == 2:  # Target
                encoded = encoded.at[i, j, OBJECT_TYPES['target']].set(1)

            if grid[i, j, 0] == 3:  # Player
                encoded = encoded.at[i, j, OBJECT_TYPES['agent']].set(1)
            elif grid[i, j, 0] == 4:  # Box
                encoded = encoded.at[i, j, OBJECT_TYPES['box']].set(1)

    return encoded

def encode_multiple_levels(num_levels, env, rng):
    encoded_levels = []
    for i in tqdm(range(num_levels)):
        rng, subkey = jax.random.split(rng)
        state, timestep = env.reset(subkey)
        encoded_level = encode_level(timestep.observation.grid)
        encoded_levels.append(encoded_level)

    return jnp.stack(encoded_levels)

def map_to_rgb(level_classes):
    color_map = {
        0: [255, 255, 255],  # empty -> white
        1: [139, 69, 19],    # wall -> brown
        2: [255, 215, 0],    # target -> yellow
        3: [0, 255, 0],      # agent -> green
        4: [255, 0, 0]       # box -> red
    }
    rgb_image = jnp.zeros((level_classes.shape[0], level_classes.shape[1], 3), dtype=jnp.uint8)
    for object_type, color in color_map.items():
        mask = (level_classes == object_type)
        for channel in range(3):
            rgb_image = rgb_image.at[mask, channel].set(color[channel])
    return rgb_image

def map_level_to_image(level_classes):
    image_grid = np.zeros((level_classes.shape[0] * 16, level_classes.shape[1] * 16, 3), dtype=np.uint8)
    
    for i in range(level_classes.shape[0]):
        for j in range(level_classes.shape[1]):
            object_type = level_classes[i, j]
            asset_image = assets[list(OBJECT_TYPES.keys())[object_type]]
            
            image_grid[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16, :] = asset_image[:, :, :3] 
    
    return image_grid

# Visualization of Decoded Levels
def visualize_decoded_level(model, params, original_level, original_shape):
    original_level = original_level.reshape(1, *original_shape)
    reconstructed = model.apply({'params': params}, original_level).squeeze()

    original_level_classes = jnp.argmax(original_level.squeeze(), axis=-1)
    reconstructed_level_classes = jnp.argmax(reconstructed, axis=-1)
    original_rgb = map_to_rgb(original_level_classes)
    reconstructed_rgb = map_to_rgb(reconstructed_level_classes)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(original_rgb)
    axs[0].set_title("Original Level")
    axs[0].axis('off')
    axs[1].imshow(reconstructed_rgb)
    axs[1].set_title("Reconstructed Level")
    axs[1].axis('off')
    plt.show()


# Visualization of Decoded Levels using assets
def visualize_decoded_level_with_assets(model, params, original_level, original_shape, is_vae=False):
    original_level = original_level.reshape(1, *original_shape)

    if is_vae:
        reconstructed, _, _ = model.apply({'params': params}, original_level) 
        reconstructed = reconstructed.squeeze()
    else:
        reconstructed = model.apply({'params': params}, original_level).squeeze()

    original_level_classes = jnp.argmax(original_level.squeeze(), axis=-1)
    reconstructed_level_classes = jnp.argmax(reconstructed, axis=-1)

    # Plot the original and reconstructed levels using assets
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    original_image = map_level_to_image(original_level_classes)
    reconstructed_image = map_level_to_image(reconstructed_level_classes)

    axs[0].imshow(original_image)
    axs[0].set_title("Original Level")
    axs[0].axis('off')

    axs[1].imshow(reconstructed_image)
    axs[1].set_title("Reconstructed Level")
    axs[1].axis('off')

    plt.show()

def resize_image(image, new_size=(16, 16)):
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  

    if image.shape[-1] == 4:
        image = image[:, :, :3]  # Drop the alpha channel

    # Resize the image using PIL and return as a numpy array
    return np.array(Image.fromarray(image).resize(new_size))

# Sampling new levels from the latent space and decoding them
def generate_new_levels(model, params, latent_dim, method):
    num_levels=50
    grid_size=(5, 10)
    seed = random.randint(0, int(1e6))
    rng = jax.random.PRNGKey(seed) 
    rng, subkey = jax.random.split(rng)
    
    # Sample random points in the latent space using the new subkey
    latent_samples = jax.random.normal(subkey, (num_levels, latent_dim))

    # Decode the latent samples into new levels
    generated_levels = model.apply({'params': params}, latent_samples, method=method)

    # Set up the plot grid
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(20, 10))
    axs = axs.ravel()  # Flatten the grid of axes for easy iteration

    # Visualize the generated levels in a grid
    for i in tqdm(range(num_levels)):
        generated_level = generated_levels[i].squeeze()
        generated_level_classes = jnp.argmax(generated_level, axis=-1)
        generated_image = map_level_to_image(generated_level_classes)  # Map to assets

        axs[i].imshow(generated_image)
        axs[i].axis('off')
        axs[i].set_title(f"Level {i + 1}")

    for i in range(num_levels, grid_size[0] * grid_size[1]):
        fig.delaxes(axs[i])

    plt.tight_layout()
    plt.show()


def visualize_latent_space(model, params, batch, method, num_components=2, is_vae=False):
    # Pass levels through the encoder to get latent representations
    latent_vectors = model.apply({'params': params}, batch, method=method)

    # If VAE, latent_vectors might be a tuple (mu, log_var), so we use mu
    if is_vae:
        latent_vectors = latent_vectors[0] 

    if len(latent_vectors.shape) > 2:
        latent_vectors = latent_vectors.reshape((latent_vectors.shape[0], -1))
        
    # Reduce to 2D using PCA
    pca = PCA(n_components=num_components)
    latent_2d = pca.fit_transform(latent_vectors)

    # Plot the 2D latent space
    plt.figure(figsize=(8, 6))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c='blue', label='Sokoban Levels')
    plt.title(f'{num_components}D Visualization of Latent Space')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()
    plt.show()