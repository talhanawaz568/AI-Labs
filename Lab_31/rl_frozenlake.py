import gymnasium as gym # Modern version of gym
import os

# --- Task 2: Setting Up the Environment ---
print("Task 2: Initializing FrozenLake-v1...")

# is_slippery=False means if you move right, you actually go right (no sliding)
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="ansi")

# Reset the environment to start
# In newer gym versions, reset() returns (state, info)
state, info = env.reset()

print(f"Initial State: {state}")
print("Environment Map:")
print(env.render()) # Shows the 4x4 grid (S=Start, F=Frozen, H=Hole, G=Goal)

# --- Task 3: Document Actions and Rewards ---
print("\nTask 3: Simulating Agent Actions...")

# Action Space: 0: Left, 1: Down, 2: Right, 3: Up
for step in range(10):  # Let's try 10 steps
    # 3.1 Sample a random action
    action = env.action_space.sample()
    
    # 3.2 Apply the action
    # Modern Gym returns: obs, reward, terminated, truncated, info
    new_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    print(f"Step {step + 1}:")
    print(f"  Action taken: {action} (0:L, 1:D, 2:R, 3:U)")
    print(f"  New State:    {new_state}")
    print(f"  Reward:       {reward}")
    print(f"  Episode Done: {done}")
    
    # Optional: Print the visual map after the move
    print(env.render())

    if done:
        if reward == 1:
            print("✨ SUCCESS! The agent reached the Goal (G)!")
        else:
            print("💀 GAME OVER! The agent fell into a Hole (H).")
        break

env.close()
