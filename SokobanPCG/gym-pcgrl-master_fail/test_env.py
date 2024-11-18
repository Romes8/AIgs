import gym
import gym_sokoban
from gym_pcgrl.wrappers import ActionMapImagePCGRLWrapper
import numpy as np

def main():
    # Step 1: Create the base Sokoban environment
    base_env = gym.make('Sokoban-v0')
    
    # Step 2: Wrap the environment with your custom wrapper
    env = ActionMapImagePCGRLWrapper(base_env)
    
    # Step 3: Reset the environment and get the initial observation and info
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        if len(reset_output) == 2:
            obs, info = reset_output
        else:
            obs = reset_output
            info = {}
    else:
        obs = reset_output
        info = {}
    
    # Step 4: Print initial observation details
    if isinstance(obs, np.ndarray):
        print("Initial Observation Shape:", obs.shape)
    elif isinstance(obs, dict):
        print("Initial Observation Keys:", list(obs.keys()))
    else:
        print("Initial Observation Type:", type(obs))
    
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    
    done = False
    step_count = 0  # To prevent infinite loops
    
    while not done and step_count < 100:  # Limit to 100 steps for testing
        action = env.action_space.sample()  # Sample a random action
        step_output = env.step(action)
        if isinstance(step_output, tuple):
            if len(step_output) == 5:
                obs, reward, terminated, truncated, info = step_output
            elif len(step_output) == 4:
                obs, reward, done, info = step_output
                terminated, truncated = done, False
            else:
                raise ValueError("Unexpected number of values returned from step.")
        else:
            raise ValueError("Environment step did not return a tuple.")
        
        # Render the environment
        env.render()
        
        # Check if the episode is done
        done = terminated or truncated
        step_count += 1
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
