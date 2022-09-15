import os, sys
sys.path.append('./agents/')
from citylearn.citylearn import CityLearnEnv

from SAC.sac_torch import Agent


class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'



def sac_policy(observation, action_space):
    pass


class SACAgent():
    
    def __init__(self):

        env = CityLearnEnv(schema=Constants.schema_path)
        
        self.agent = Agent(
            input_dims=env.observation_space[0].shape,
            env=env,
            n_actions=env.action_space[0].shape[0],
            alpha=0.003,
            beta=0.003
        )
        print(f"The number of observations are: {env.observation_space[0].shape}")
        print(f"The number of actions are: {env.action_space[0].shape[0]}")
        self.action_space = {}
    
    def register_reset(self, observation, action_space, agent_id):
        """Get the first observation after env.reset, return action"""
        #self.action_space[agent_id] = action_space
        return self.agent.choose_action(observation)

    def compute_action(self, observations, reward, observations_, agent_id, done):
        """Get observation return action"""
        action = self.agent.choose_action(observations)
        print(action)
        action = action.reshape(1, 5)
        print('************************************')
        print(action)
        self.agent.remember(observations[agent_id], action, reward, observations_[agent_id], done)
        self.agent.learn()
        return action


        # ## Should return actions
        # return sac_policy(observation, self.action_space[agent_id])

    