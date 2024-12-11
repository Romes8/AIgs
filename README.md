Final project - Artificial Intelligence for Games and Simulations 2024

## Generating Sokoban levels using different content generation methods

# Neural Cellural Automata - NCA

Inspiration for implementing an NCA for Generating new Sokoban levels was taken from the article “Illuminating Diverse Neural Cellular Automata for Level Generation” which talks about a “method of generating diverse collections of neural cellular automata (NCA) to design video game levels”. This research was trying to build a much simpler NCA model and still be able to generate new levels. 
The model that was built uses a relatively shallow NCA with 64 hidden channels and a single perception layer. This simplicity is less demanding on computational efficiency and makes the training faster but may lack the ability to understand the complexities of Sokoban Levels. As for the activation function, Softmax was used which enforces mutual exclusivity among elements making it possible to have each cell as a separate game block (wall, floor, player, box, goal). The model was enforced to work based on certain constraints which are required for Sokoban levels such as having walls around the border. Box bot being spawned in the corners or where 2 sides of the box are touching the wall. Such constraints act as the rule set for the level of training, however they introduce less diversity in the model behavior which lowers the success of generating valid levels. 
In order to have a valid level, the model needs to place exactly one player and the same amount of boxes and goals into the level. These parameters ensure only validity but not solvability. To tackle the solvability Breadth-First Search (BFS) solver was applied to the training stage to find the optimal path for the player to move all the boxes onto goals if possible. The benefit of this was to make sure that the level was not split into “two sections” and that all the floor tiles were connected. Mentioning training, the model was trained on a dataset of 1000 already made Sokoban levels marked as “medium” difficulty with at least 3 boxes and goals per level and a high density of walls. 


## Results
The current model does not achieve foundational success in making levels valid or solvable due to the Sokoban level complexity and logic that needs to be applied during the training process. Generated levels were carrying signs on repetitive patterns when placing boxes and other elements into the level which supports the statement from found research that NCAs could grow complex 2D patterns, however their lack of global context and and logical constraints makes it challenging to be applied for Sokoban level generation. NCAs in general are not efficient when it comes to deliberate structure, strategic element placement, and global coherence which is essential when making Sokoban levels.

### How to run: 
1. Required python 3.12
2. 
```
cd NCA
python NewNCA.py
```