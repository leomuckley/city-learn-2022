import numpy as np
#from citylearn.citylearn import CityLearnEnv
#import supersuit as ss
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.test import api_test
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
#import gym

#######

from citylearn.citylearn_pettingzoo import CityLearnPettingZooEnv


"""

input : 

    observations (per agent) = [ (28,) (28,) (28,) (28,) (28,)]



    for observation in agent[observations] 



"""



#env.observations




#########

class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
    central_agent_schema_path = './data/citylearn_challenge_2022_phase_1/central_agent_schema.json'


def train_model(model_name='new_policy'):
    """
    Train new model for new policy and save.
    """
    env = CityLearnPettingZooEnv(schema=Constants.schema_path)
    #model = PPO('MlpPolicy', env, verbose=3, gamma=0.95, n_steps=256, ent_coef=0.0905168, learning_rate=0.00062211, vf_coef=0.042202, max_grad_norm=0.9, gae_lambda=0.99, n_epochs=5, clip_range=0.3, batch_size=256)
    # multiprocess environment
    #env = make_vec_env('CartPole-v1', n_envs=4)

    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("ppo2_pz_env")

# del model # remove to demonstrate saving and loading

# model = PPO2.load("ppo2_cartpole")

# # Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()


if __name__ == '__main__':
    train_model()

