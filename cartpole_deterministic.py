
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

n_episode = 50
n_timestep = 50

for episode in range(n_episode):
    state, info = env.reset()
    total_reward = 0
    
    for timestep in range(n_timestep):
        pole_angle = state[2]
        
        if pole_angle > 0:
            action = 1
        else:
            action = 0
            
        
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        
        if terminated or truncated:
            break
            
    print(f"Episode {episode + 1} finished with Total Reward: {total_reward}")


env.close()
