import numpy as np
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO


def train_model(env):
    """
    Train new model for new policy and save.
    """
    model = PPO(CnnPolicy, env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    model.learn(total_timesteps=2000000)
    model.save('new_policy')

def new_policy(observation, action_space):
    """
    Simple rule based policy based on day or night time
    """
    # Todo: Train the model - model = PPO.load(“policy”)
    model = ""
    action  = model.predict(observation, deterministic=True)[0]
    
    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action

class BasicPPOAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}
    
    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        self.action_space[agent_id] = action_space
        return new_policy(observation, action_space)

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return new_policy(observation, self.action_space[agent_id])