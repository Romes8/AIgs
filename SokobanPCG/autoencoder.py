import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import matplotlib.pyplot as plt
import jumanji
from tqdm import tqdm
from utils import encode_multiple_levels, visualize_latent_space, assets, generate_new_levels, resize_image, visualize_decoded_level_with_assets

# Initialize RNG and environment
rng = jax.random.PRNGKey(0)
env = jumanji.make("Sokoban-v0")

#  Encoder
class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = x.reshape((x.shape[0], -1))  # Flatten
        latent = nn.Dense(features=self.latent_dim)(x)
        return latent

#  Decoder
class Decoder(nn.Module):
    latent_dim: int
    original_shape: tuple

    @nn.compact
    def __call__(self, latent):
        batch_size = latent.shape[0]
        x = nn.Dense(features=128 * (self.original_shape[0] // 4) * (self.original_shape[1] // 4))(latent)
        x = x.reshape((batch_size, self.original_shape[0] // 4, self.original_shape[1] // 4, 128))

        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)

        x = nn.ConvTranspose(features=self.original_shape[2], kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        return x

# Autoencoder combining Encoder and Decoder
class Autoencoder(nn.Module):
    latent_dim: int
    original_shape: tuple

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.latent_dim, self.original_shape)

    def __call__(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    def encode(self, latent):
        return self.encoder(latent)
    def decode(self, latent):
        return self.decoder(latent)

# MSE Loss
# cross-entropy loss function
def compute_loss(params, model, batch):
    # Forward pass through the model
    reconstructions = model.apply({'params': params}, batch)
    # Use softmax cross-entropy for classification
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=reconstructions, 
        labels=jnp.argmax(batch, axis=-1)
    )
    return jnp.mean(loss)

# Training step
@jax.jit
def train_step(params, opt_state, batch):
    loss, grads = jax.value_and_grad(compute_loss)(params, model, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Train the model
def train_autoencoder(num_epochs, batch):
    global params, opt_state
    for epoch in tqdm(range(num_epochs)):
        params, opt_state, loss = train_step(params, opt_state, batch)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')



# Initialize model, optimizer, and parameters
encoded_levels = encode_multiple_levels(100, env, rng)
learning_rate = 1e-3
optimizer = optax.adam(learning_rate)
original_shape = (10, 10, 5)
latent_dim = 64  
model = Autoencoder(latent_dim=latent_dim, original_shape=original_shape)
rng = jax.random.PRNGKey(0)
params = model.init(rng, jnp.ones((1, *original_shape)))['params']
opt_state = optimizer.init(params)

for key in assets.keys():
    assets[key] = resize_image(assets[key])

batch = encoded_levels.reshape((-1, *original_shape))
train_autoencoder(500, batch) 

# visualize_latent_space(model, params, batch, Autoencoder.encode)
visualize_decoded_level_with_assets(model, params, encoded_levels[-1], original_shape)
generate_new_levels(model, params, latent_dim = latent_dim, method = Autoencoder.decode)





