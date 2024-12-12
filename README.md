Final project - Artificial Intelligence for Games and Simulations 2024

# Generating Sokoban levels using different content generation methods

## Autoencoders 

Required python version: 3.10 or higher

To run autoencoder, install required packages, then inside of Autencoders folder run training using 
`python autoencoder.py`
or
`python variational_autoencoder.py`

## Neural Cellural Automata - NCA

Required python version 3.10 or higher

To run NCA, install required packages, then inside of NCA folder run
`python NewNCA.py`

## PCG-RL

Original implementation:
https://github.com/amidos2006/gym-pcgrl/tree/master

Required python version 3.10

To run start new training install required packages, then inside of pcgrl-updated folder run
`python train.py`
you can change representation type by setting `representation` argument in main function. 
To continue training, update `load_dir` to the path of a model you want to continue training.

To generate new levels, run
`python generator.py`
You can adjust model used for generation, number of generated levels and representation inside of generator.py